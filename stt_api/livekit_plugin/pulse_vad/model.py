"""
The PulseVAD checkpoints: two sizes, two precisions, one output contract.

PulseVAD (https://github.com/AydinAdnan/PulseVAD, MIT) is a *frame* VAD — is
there speech in these 200 ms — not an end-of-turn model. It is the same job
silero does in `livekit-plugins-silero`, at 2,118 parameters instead of 1.8 M.
Nothing here overlaps with `../semantic_vad/` or `../turn_detector/`, which
answer "has the speaker finished"; see this package's README for which is which.

The weights are vendored under `resources/` rather than taken from the `pulsevad`
PyPI package on purpose. That package hard-depends on **scipy and soundfile** for
one file-reading helper this plugin never calls, and declares
`requires-python >= 3.11` while this repo supports 3.10. Dragging both into an
agent image for 12 KB of weights is exactly the extra-boundary violation
`CLAUDE.md` warns about; the front end is 90 lines of numpy and is reimplemented
in `frontend.py`, bit-exact.

Three traps, all measured on this host over 12,098 windows of `test_audio/` plus
synthetic noise:

**1. The output is logits, not a probability.** `logits` is `(B, 2)` ordered
`[non_speech, speech]`, and p = sigmoid(logits[1] - logits[0]). This is the
opposite of the smart-turn graphs in `../semantic_vad/backends.py`, whose single
output is already a probability — applying a second sigmoid there pinned every
score to ~0.73. Here, *forgetting* the sigmoid is the mirror-image bug.

**2. p(speech) saturates well below 1.0, and a threshold above the ceiling makes
the VAD permanently silent.** Measured ceilings:

    2.1k fp32   p in [0.002, 0.711]
    2.1k int8   p in [0.005, 0.708]
    81k  fp32   p in [0.006, 0.895]

At `activation_threshold=0.75` the 2.1k model fires on **0.0 %** of windows — the
agent never hears anyone, and nothing errors. Since `activation_threshold=0.8` is
an ordinary thing to write when coming from silero (which saturates at 1.0),
`PulseVAD.load` refuses a threshold at or above the ceiling rather than shipping
a deaf agent.

**3. int8 is not faster here, and the file is bigger.** Measured at batch 1 on
one CPU thread: fp32 **0.018 ms**, int8 **0.019 ms** — and 26.8 KB of QDQ graph
against 12 KB of fp32. Quantisation pays on the microcontrollers PulseVAD targets;
in an agent process it costs a mean |Δp| of 0.008 (max 0.052) for nothing. So
`precision` defaults to `"fp32"` here even though upstream's `load_pulsevad`
defaults to `quantized=True`. `"int8"` stays available for parity testing against
an edge deployment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from .frontend import WINDOW_SAMPLES, log_mel

__all__ = ["PulseVADModel", "CEILINGS"]

Variant = Literal["2.1k", "81k"]
Precision = Literal["fp32", "int8"]

_FILES: dict[tuple[str, str], str] = {
    ("2.1k", "fp32"): "pulsevad_2.1k.onnx",
    ("2.1k", "int8"): "pulsevad_2.1k_int8.onnx",
    ("81k", "fp32"): "pulsevad_teacher_81k.onnx",
    # No int8 teacher is published upstream; ("81k", "int8") raises below.
}

CEILINGS: dict[tuple[str, str], float] = {
    ("2.1k", "fp32"): 0.711,
    ("2.1k", "int8"): 0.708,
    ("81k", "fp32"): 0.895,
}
"""Largest p(speech) observed over 12,098 windows of `test_audio/` plus synthetic
noise, at 1 dp of headroom. Used to reject a threshold that can never be crossed
— see trap 2 in the module docstring. p99.9 sits within 0.0005 of each of these,
so the saturation is flat, not a tail."""


def _resource(filename: str) -> str:
    path = Path(__file__).parent / "resources" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"PulseVAD weights missing from the package at {path}. Reinstall stt-api, "
            f"or pass model_path= pointing at a copy of {filename}."
        )
    return str(path)


@lru_cache(maxsize=8)
def _session(path: str, num_threads: int, providers: tuple[str, ...]):
    """
    One ORT session per (file, threads, providers), shared across streams.

    The graph is stateless and ORT sessions are safe to call concurrently, so
    every participant in a room can share one. Single-threaded by default: the
    model is 0.018 ms of work, and a thread pool for 2,118 parameters loses more
    to contention with the STT sharing the core than it could ever win.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = num_threads
    return ort.InferenceSession(path, opts, providers=list(providers))


class PulseVADModel:
    """
    A loaded PulseVAD checkpoint. Windows of 3,200 samples in, p(speech) out.

        model = PulseVADModel()                      # 2.1k fp32, the default
        model = PulseVADModel("81k")                 # the 81 k teacher
        model = PulseVADModel("2.1k", precision="int8")

        p = model(pcm_3200)                          # one window  -> float
        ps = model.probabilities(windows)            # (B, 3200)   -> (B,)

    Usable without livekit-agents installed — `benchmark_pulse_vad.py` and the
    tests both drive it directly.
    """

    def __init__(
        self,
        model: Variant = "2.1k",
        *,
        precision: Precision = "fp32",
        num_threads: int = 1,
        providers: list[str] | None = None,
        model_path: str | None = None,
    ) -> None:
        if model not in ("2.1k", "81k"):
            raise ValueError(f"model must be '2.1k' or '81k', got {model!r}")
        if precision not in ("fp32", "int8"):
            raise ValueError(f"precision must be 'fp32' or 'int8', got {precision!r}")
        if (model, precision) not in _FILES:
            raise ValueError(
                f"no {precision} build of the {model} model is published upstream. "
                f"Available: {sorted(_FILES)}. Use precision='fp32' for '81k'."
            )

        self.model = model
        self.precision = precision
        self.ceiling = CEILINGS[(model, precision)]
        """Largest p(speech) this checkpoint is known to emit. A threshold at or
        above it can never fire — `PulseVAD.load` checks against this."""

        path = (
            model_path
            or os.environ.get("PULSEVAD_ONNX_PATH")
            or _resource(_FILES[(model, precision)])
        )
        if model_path or os.environ.get("PULSEVAD_ONNX_PATH"):
            if not os.path.exists(path):
                raise FileNotFoundError(f"PulseVAD model not found at {path}")
        self.path = path
        self._session = _session(
            path, num_threads, tuple(providers or ["CPUExecutionProvider"])
        )

    def probabilities(self, windows: np.ndarray) -> np.ndarray:
        """`(B, 3200)` float32 at 16 kHz -> `(B,)` p(speech)."""
        logits = self._session.run(None, {"log_mel": log_mel(windows)})[0]
        # [non_speech, speech]: the sigmoid is NOT in the graph. See trap 1.
        return 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))

    def __call__(self, window: np.ndarray) -> float:
        """One 3,200-sample window -> p(speech)."""
        if window.shape[-1] != WINDOW_SAMPLES:
            raise ValueError(
                f"PulseVAD needs exactly {WINDOW_SAMPLES} samples (200 ms at 16 kHz), "
                f"got {window.shape[-1]}"
            )
        return float(self.probabilities(np.asarray(window).reshape(1, -1))[0])

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"PulseVADModel({self.model!r}, precision={self.precision!r})"
