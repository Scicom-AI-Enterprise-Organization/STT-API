"""
The models behind the detector: one local, one remote, one interface.

A *semantic* VAD decides whether a speaker has finished from the **waveform** —
prosody, final-syllable lengthening, intonation contour — rather than from a
transcript. That distinction is the whole point of this plugin. The text
detectors in `../turn_detector/` cannot fire until the STT has produced words,
which on a production stack measured a median 1.5 s after the speaker stopped;
an audio-native model answers from the audio the agent already has.

Three backends, because the useful deployments are genuinely different:

* `ScicomEoT` — `Scicom-intl/semantic-vad-eot-whisper-{tiny,base,small}`,
  Apache-2.0, trained on real Malaysian call-centre telephony. **The one to use
  on `ms`.** int8 ONNX, 24-145 ms on one CPU thread depending on size.
* `SmartTurnV3` — `pipecat-ai/smart-turn-v3`, BSD-2. Same architecture, 23
  languages, no Malay.
* `RemoteEoT` — a plain HTTP POST to a model too large to co-locate, such as a
  Qwen2-Audio classifier on a GPU. Deliberately plain JSON: LiveKit's own cloud
  path speaks protobuf over a websocket, but nothing forces a *self-hosted*
  detector to reimplement that server. See the README.

All three satisfy `Backend`, so the transport does not know or care which it has.
The first two share `_WhisperWindowEoT`: same recipe, different checkpoints.
"""

from __future__ import annotations

import base64
import json
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["Backend", "RemoteEoT", "ScicomEoT", "SmartTurnV3"]

SAMPLE_RATE = 16000


@runtime_checkable
class Backend(Protocol):
    """One pause worth of audio in, one probability out."""

    #: Seconds of trailing audio the model wants. The transport sizes its ring
    #: buffer from this, so a backend that needs more context simply asks.
    window_seconds: float

    def predict(self, pcm: np.ndarray) -> float:
        """`pcm` is mono float32 in [-1, 1] at 16 kHz. Returns p(turn complete)."""
        ...


class _WhisperWindowEoT:
    """
    Shared implementation for the Whisper-encoder-plus-head EoT family.

    `pipecat-ai/smart-turn-v3` and Scicom's `semantic-vad-eot-whisper-*` are the
    same recipe and the same input contract: an 80 x 800 log-mel of a fixed 8 s
    window, left-padded, into a graph whose single output is **already a
    probability**. Only three things differ between them — the checkpoint, the
    window length, and whether the feature extractor normalises — so they differ
    by configuration here rather than by implementation.

    Two details are load-bearing and silent when wrong:

    * **Left padding.** The model is trained with the decision point at the *end*
      of its window. Letting the feature extractor right-pad puts the speech at
      the start followed by seconds of silence, which it never saw in training
      and scores with confident nonsense rather than an error.
    * **`do_normalize` is per-checkpoint.** smart-turn-v3 wants it on; the Scicom
      models ship `do_normalize: false` in `eot_window.json`. Measured on the
      same tone, the flag moves p(eot) from 0.39 to 0.53 — a plausible number
      either way, so nothing tells you it is wrong.
    """

    window_seconds: float = 8.0
    do_normalize: bool = True

    def __init__(
        self,
        *,
        model_path: str,
        window_seconds: float | None = None,
        do_normalize: bool | None = None,
        num_threads: int = 1,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        if window_seconds is not None:
            self.window_seconds = window_seconds
        if do_normalize is not None:
            self.do_normalize = do_normalize

        opts = ort.SessionOptions()
        # One thread by default: this runs inside the agent process alongside the
        # STT, the VAD and whatever else shares the core, and a thread pool for
        # single-digit-millions of parameters costs more in contention than it
        # saves in latency.
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        self._sess = ort.InferenceSession(
            model_path, opts, providers=providers or ["CPUExecutionProvider"]
        )
        self._input = self._sess.get_inputs()[0].name

        # chunk_length is what makes this 800 frames instead of Whisper's usual
        # 3000-frame 30 s grid. Getting it wrong is a shape error at the first
        # inference, not a silent quality loss, which is the good kind of wrong.
        self._fe = WhisperFeatureExtractor(chunk_length=int(self.window_seconds))

    @staticmethod
    def _fit_window(pcm: np.ndarray, n: int) -> np.ndarray:
        """Take the last `n` samples, **left**-padding when short."""
        if len(pcm) > n:
            return pcm[-n:]
        if len(pcm) < n:
            return np.pad(pcm, (n - len(pcm), 0), mode="constant")
        return pcm

    def predict(self, pcm: np.ndarray) -> float:
        if pcm.size == 0:
            return 0.0
        n = int(self.window_seconds * SAMPLE_RATE)
        window = self._fit_window(np.asarray(pcm, dtype=np.float32), n)
        feats = self._fe(
            window,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=n,
            truncation=True,
            do_normalize=self.do_normalize,
        )["input_features"].astype(np.float32)
        # The exported graphs already apply the sigmoid — the single output *is*
        # p(complete); smart-turn's is unnamed, the Scicom ones name it
        # `probability`. Applying another sigmoid here squashed every score toward
        # 0.73 and made the model look like it was ignoring its input.
        return float(self._sess.run(None, {self._input: feats})[0].reshape(-1)[0])


class SmartTurnV3(_WhisperWindowEoT):
    """
    `pipecat-ai/smart-turn-v3` — an open semantic VAD, BSD-2-Clause.

    Whisper-Tiny encoder into a shallow linear classifier: 8 M parameters, 8 MB
    quantised. Reported 92.6 % accuracy over 31,527 samples across 23 languages
    (FPR 4.7 %, FNR 2.6 %).

    **Malay is not among those 23 languages.** Indonesian is, and is a useful
    prior, but for `ms` prefer `ScicomEoT`, which is trained on Malaysian
    call-centre telephony.
    """

    window_seconds = 8.0
    do_normalize = True

    _REPO = "pipecat-ai/smart-turn-v3"
    _DEFAULT_FILE = "smart-turn-v3.2-cpu.onnx"

    def __init__(self, *, model_file: str | None = None, **kw) -> None:
        from huggingface_hub import hf_hub_download

        super().__init__(
            model_path=hf_hub_download(self._REPO, model_file or self._DEFAULT_FILE), **kw
        )


class ScicomEoT(_WhisperWindowEoT):
    """
    `Scicom-intl/semantic-vad-eot-whisper-{tiny,base,small}` — Apache-2.0.

    Same architecture as smart-turn-v3, trained on **real Malaysian call-centre
    telephony** (Malay and English, both channels) where turn ends are *observed*
    — the other party took the floor — rather than inferred from alignment gaps.
    That makes this the one to reach for on `ms`, which smart-turn-v3 does not
    cover.

    Sizes, on the publishers' own eot-bench run over 300 private telephony turns:

        tiny   AUC 0.78   ~30 ms int8, 10 MB
        base   AUC 0.84
        small  AUC 0.86   best of the three, still CPU-viable

    `int8` by default: roughly half the latency of `fp32` for a reported mean
    absolute output shift of 0.068. That is a real difference — if you are
    calibrating a threshold near a decision boundary, measure `fp32` too rather
    than assuming the quantisation is free.

    Window and normalisation are read from the checkpoint's own
    `eot_window.json` rather than hardcoded, so a re-trained model that changes
    either keeps working.
    """

    SIZES = ("tiny", "base", "small")

    def __init__(
        self,
        size: str = "base",
        *,
        repo: str | None = None,
        quantized: bool = True,
        **kw,
    ) -> None:
        import json

        from huggingface_hub import hf_hub_download

        if repo is None:
            if size not in self.SIZES:
                raise ValueError(f"size must be one of {self.SIZES}, got {size!r}")
            repo = f"Scicom-intl/semantic-vad-eot-whisper-{size}"
        self.repo = repo

        model_path = hf_hub_download(
            repo, "onnx/model.int8.onnx" if quantized else "onnx/model.fp32.onnx"
        )
        # Trust the checkpoint over any default: `do_normalize` is false for these
        # and true for smart-turn, and the wrong one produces a plausible score.
        cfg = {}
        for candidate in ("onnx/eot_window.json", "eot_window.json"):
            try:
                cfg = json.load(open(hf_hub_download(repo, candidate)))
                break
            except Exception:  # noqa: BLE001 - fall through to the next candidate
                continue
        kw.setdefault("window_seconds", float(cfg.get("window_seconds", 8.0)))
        kw.setdefault("do_normalize", bool(cfg.get("do_normalize", False)))
        super().__init__(model_path=model_path, **kw)


class RemoteEoT:
    """
    A self-hosted audio EoT model behind a plain HTTP endpoint.

    For models too large to run in the agent — a Qwen2-Audio classifier, say.
    The request is deliberately boring:

        POST {url}
        {"audio": "<base64 PCM s16le>", "sample_rate": 16000}
        -> {"probability": 0.83}

    JSON and base64 rather than LiveKit's protobuf-over-websocket protocol,
    because implementing that server buys nothing here. LiveKit's transport is a
    seven-method Protocol; once you implement it in-process (which this plugin
    does) you own the wire and can use whatever the model server already speaks.
    The cost is one HTTP round trip per pause, which must fit inside the 1.0 s
    prediction timeout along with inference.

    `urllib` on a worker thread, not aiohttp: the transport already calls this
    from `asyncio.to_thread`, and a blocking client there is simpler than a
    second async stack.
    """

    def __init__(
        self,
        url: str,
        *,
        window_seconds: float = 8.0,
        timeout: float = 0.9,
        headers: dict[str, str] | None = None,
        probability_key: str = "probability",
    ) -> None:
        self.url = url
        self.window_seconds = window_seconds
        # Default under LiveKit's 1.0 s DEFAULT_PREDICTION_TIMEOUT: if the server
        # is slow, failing our own way leaves a usable error in the log, whereas
        # letting the client time out looks like a detector that never answers.
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.probability_key = probability_key

    def predict(self, pcm: np.ndarray) -> float:
        import urllib.request

        window = pcm[-int(self.window_seconds * SAMPLE_RATE) :]
        s16 = np.clip(np.rint(window * 32768.0), -32768, 32767).astype("<i2")
        body = json.dumps(
            {
                "audio": base64.b64encode(s16.tobytes()).decode("ascii"),
                "sample_rate": SAMPLE_RATE,
            }
        ).encode()
        req = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return float(data[self.probability_key])
