"""
The downstream axis: does the denoiser help the STT, or hurt it?

This is the metric that decides the question the plugin actually exists to
answer, and it is the one most likely to disagree with the others. Suppression is
tuned to please human listeners, and the things that please a listener — a silent
background, a smooth spectrum — are not what an acoustic model needs. Fricatives
and stop bursts are low-energy, noise-like, and the first thing an aggressive
suppressor removes. A model can gain a point of PESQ and lose WER doing it. If
the tables in this benchmark ever disagree, this is the column to believe.

**What the reference is.** Neither VoiceBank+DEMAND nor the DNS dev set ships
transcripts, so the reference here is the STT's *own transcript of the clean
signal*, and WER is measured against that. This is a deliberate choice, not a
workaround, and it changes the interpretation: the number is not "how accurate is
the STT", it is "how much of the clean-audio result does this denoiser preserve".
That is the right question — a denoiser cannot be blamed for words the STT would
have missed anyway, and pooling both effects into one number would hide which is
which. `passthrough` gives the degradation the noise alone causes; anything that
does not beat that row is losing to doing nothing.

A real transcript column is used instead when the corpus has one.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass

import numpy as np

__all__ = ["AsrClient", "WerAccumulator", "WhisperClient", "is_degenerate"]


def _wav_bytes(x: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    pcm = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 32768.0), -32768, 32767).astype(
        np.int16
    )
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _multipart(fields: dict[str, str], filename: str, payload: bytes) -> tuple[bytes, str]:
    boundary = f"----nc-benchmark-{uuid.uuid4().hex}"
    out = bytearray()
    for k, v in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    out += f"--{boundary}\r\n".encode()
    out += (
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode()
    out += payload + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


@dataclass
class AsrClient:
    """
    Thin client for this repo's own `POST /audio/transcriptions`.

    Deliberately `urllib` rather than a client library: the benchmark's optional
    extras should not drag an HTTP stack into a package whose whole selling point
    is that it runs inside an agent process without pulling the world in.
    """

    base_url: str = "http://127.0.0.1:8000"
    language: str | None = None
    timeout: float = 300.0

    def transcribe(self, x: np.ndarray, rate: int) -> str:
        fields = {"response_format": "json"}
        if self.language:
            fields["language"] = self.language
        body, content_type = _multipart(fields, "audio.wav", _wav_bytes(x, rate))
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/audio/transcriptions",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"STT returned {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"cannot reach STT at {self.base_url} ({e.reason}); start it or pass --asr-url"
            ) from None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip()
        if isinstance(data, dict):
            if isinstance(data.get("text"), str):
                return data["text"]
            segs = data.get("segments")
            if isinstance(segs, list):
                return " ".join(
                    s.get("text", "") for s in segs if isinstance(s, dict)
                ).strip()
        return str(data)

    def check(self) -> None:
        """Fail fast, before a long run, if the STT is not actually there."""
        self.transcribe(np.zeros(16000, dtype=np.float32), 16000)


@dataclass
class WhisperClient:
    """
    Local Whisper, for when there is no `stt-api` to point at — and the only way
    to say anything trustworthy about the generative restorers.

    DNSMOS cannot settle that class on its own, and relying on it would be a
    serious mistake. A vocoder that resynthesises confident, fluent speech scores
    *well* on a reference-free perceptual metric whether or not the words it
    produced are the words that were said. Hallucination is this class's
    characteristic failure and DNSMOS is blind to it by construction; WER is the
    only axis here that catches it.

    `large-v3` by default because that is what CallEnhancer's published CER table
    used, so the numbers are comparable to theirs rather than to nothing.
    """

    model: str = "openai/whisper-large-v3"
    device: str = "cuda"
    language: str | None = "en"
    _pipe: object = None

    def _ensure(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from transformers import pipeline

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device=self.device,
        )
        return self._pipe

    def transcribe(self, x: np.ndarray, rate: int) -> str:
        import numpy as _np

        pipe = self._ensure()
        if rate != 16000:
            from .audio import StreamResampler

            x = StreamResampler(rate, 16000).push(x)
        kw = {"language": self.language} if self.language else {}
        out = pipe(
            {"raw": _np.asarray(x, dtype=_np.float32), "sampling_rate": 16000},
            generate_kwargs=kw,
        )
        return (out or {}).get("text", "").strip()

    def check(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32), 16000)


def is_degenerate(ref: str, hyp: str, *, expansion: float = 3.0, unique_ratio: float = 0.35) -> bool:
    """
    Is this hypothesis a Whisper repetition loop rather than a transcript?

    This guard is not fussiness, it is the difference between a true and a false
    conclusion. Pooled WER is edit distance over *reference* words, so insertions
    are unbounded: one hypothesis that loops a phrase forty times produced 407
    words against a 20-word reference — 2035 % WER for that item alone, and about
    three quarters of all errors over a 100-item corpus. Left in, it turned a
    model whose median per-item WER was 0 % into an apparent 70 % catastrophe.

    Repetition looping is a known decoder failure of autoregressive ASR on
    degraded input. It is a property of the ASR, not of the enhancer being
    scored, so it is counted and reported separately rather than being allowed to
    dominate — and separately rather than silently dropped, because a model that
    provokes many loops is genuinely producing audio the ASR cannot handle.

    Flagged when the hypothesis is both much longer than the reference and made
    of very few distinct 4-grams.
    """
    rw, hw = ref.split(), hyp.split()
    if len(hw) <= max(12, int(expansion * max(len(rw), 1))):
        return False
    grams = [" ".join(hw[i : i + 4]) for i in range(len(hw) - 3)]
    if not grams:
        return False
    return (len(set(grams)) / len(grams)) < unique_ratio


class WerAccumulator:
    """
    Pools per-utterance edit counts into a corpus WER.

    Pooled counts, not a mean of per-utterance rates: a two-word utterance and a
    forty-word one do not deserve equal weight, and averaging rates is how a
    single short clip with one error ends up dominating the table. This mirrors
    what `stt_api.evaluation.corpus_metrics` does for the repo's own scoring.

    Pooling has one sharp edge, though, and `is_degenerate` exists for it: because
    insertions are unbounded, a single ASR repetition loop can outweigh the whole
    corpus. Those are excluded from the pool and counted, and the median
    per-item rate is reported next to the pooled one so the two disagreeing is
    visible rather than silent.
    """

    def __init__(self) -> None:
        from stt_api.evaluation import Metrics

        self._total = Metrics()
        self.pairs: list[tuple[str, str, str]] = []
        self.per_item: list[float] = []
        self.degenerate: list[str] = []

    def add(self, item_id: str, ref: str, hyp: str) -> None:
        from stt_api.evaluation import score_one

        if is_degenerate(ref, hyp):
            self.degenerate.append(item_id)
            return
        m = score_one(ref, hyp)
        self._total = self._total + m
        self.per_item.append(m.word_dist / max(m.ref_words, 1))
        self.pairs.append((item_id, ref, hyp))

    @property
    def median_wer(self) -> float:
        if not self.per_item:
            return float("nan")
        import statistics

        return statistics.median(self.per_item)

    @property
    def n_degenerate(self) -> int:
        return len(self.degenerate)

    @property
    def wer(self) -> float:
        return self._total.wer

    @property
    def cer(self) -> float:
        return self._total.cer

    @property
    def ref_words(self) -> int:
        return self._total.ref_words
