"""
The models behind the detector: one local, one remote, one interface.

A *semantic* VAD decides whether a speaker has finished from the **waveform** —
prosody, final-syllable lengthening, intonation contour — rather than from a
transcript. That distinction is the whole point of this plugin. The text
detectors in `../turn_detector/` cannot fire until the STT has produced words,
which on a production stack measured a median 1.5 s after the speaker stopped;
an audio-native model answers from the audio the agent already has.

Two backends, because the two useful deployments are genuinely different:

* `SmartTurnV3` — 8 M parameters of Whisper-Tiny encoder plus a linear head,
  8 MB of int8 ONNX, ~12 ms on a CPU core. Small enough to sit inside the agent
  process, which means no network hop and no service to operate.
* `RemoteEoT` — a plain HTTP POST to a model too large to co-locate, such as a
  Qwen2-Audio classifier on a GPU. Deliberately plain JSON: LiveKit's own cloud
  path speaks protobuf over a websocket, but nothing forces a *self-hosted*
  detector to reimplement that server. See the README.

Both satisfy `Backend`, so the transport does not know or care which it has.
"""

from __future__ import annotations

import base64
import json
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["Backend", "RemoteEoT", "SmartTurnV3"]

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


class SmartTurnV3:
    """
    `pipecat-ai/smart-turn-v3` — an open semantic VAD, BSD-2-Clause.

    Whisper-Tiny encoder into a shallow linear classifier: 8 M parameters, 8 MB
    quantised. Reported 92.6 % accuracy over 31,527 samples across 23 languages
    (FPR 4.7 %, FNR 2.6 %).

    **Malay is not in its training languages.** Indonesian is, and is close
    enough to be a useful prior, but for `ms` this should be benchmarked before
    it is trusted — see the README. That is exactly the gap a locally trained
    model fills, which is why `RemoteEoT` exists alongside it.

    The model takes a fixed 8-second window: 80 mel bins x 800 frames. Shorter
    pauses are padded by the feature extractor, longer context is truncated to
    the most recent 8 seconds, because that is what it was trained on.
    """

    #: 800 frames at hop 160 on 16 kHz is exactly 8 s — the model's fixed input.
    window_seconds = 8.0

    _REPO = "pipecat-ai/smart-turn-v3"
    _DEFAULT_FILE = "smart-turn-v3.2-cpu.onnx"

    def __init__(
        self,
        *,
        model_file: str | None = None,
        num_threads: int = 1,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        path = hf_hub_download(self._REPO, model_file or self._DEFAULT_FILE)

        opts = ort.SessionOptions()
        # One thread by default: this runs inside the agent process alongside the
        # STT, the VAD and whatever else shares the core, and a thread pool for
        # 8 M parameters costs more in contention than it saves in latency.
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        self._sess = ort.InferenceSession(
            path, opts, providers=providers or ["CPUExecutionProvider"]
        )
        self._input = self._sess.get_inputs()[0].name

        # chunk_length=8 is what makes this 800 frames instead of Whisper's usual
        # 3000-frame 30 s grid. Getting it wrong is a shape error at the first
        # inference, not a silent quality loss, which is the good kind of wrong.
        self._fe = WhisperFeatureExtractor(chunk_length=8)

    @staticmethod
    def _fit_window(pcm: np.ndarray, n: int) -> np.ndarray:
        """
        Take the last `n` samples, **left**-padding when short.

        The padding side is not cosmetic. The model is trained with the decision
        point at the *end* of its window, so a short pause must be pushed to the
        right with leading zeros. Letting the feature extractor right-pad instead
        puts the speech at the start followed by seconds of silence — a signal
        the model never saw in training, and one it happily scores with
        confident nonsense rather than an error.
        """
        if len(pcm) > n:
            return pcm[-n:]
        if len(pcm) < n:
            return np.pad(pcm, (n - len(pcm), 0), mode="constant")
        return pcm

    def predict(self, pcm: np.ndarray) -> float:
        if pcm.size == 0:
            return 0.0
        window = self._fit_window(
            np.asarray(pcm, dtype=np.float32), int(self.window_seconds * SAMPLE_RATE)
        )
        feats = self._fe(
            window,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=int(self.window_seconds * SAMPLE_RATE),
            truncation=True,
            do_normalize=True,
        )["input_features"].astype(np.float32)
        # The exported graph already applies the sigmoid — its single output *is*
        # p(complete). Applying another one here squashed every score toward
        # 0.73 and made the model look like it was ignoring its input.
        return float(self._sess.run(None, {self._input: feats})[0].reshape(-1)[0])


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
