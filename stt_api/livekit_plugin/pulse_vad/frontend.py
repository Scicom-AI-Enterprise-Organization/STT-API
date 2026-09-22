"""
PulseVAD's log-mel front end, vectorised.

The model is a 2,118-parameter CNN over a 64 x 21 log-mel patch, and the patch is
the expensive part: upstream's `pulsevad.frontend_np` builds it with a Python
loop over the 21 STFT frames, one `np.fft.rfft` call each. Measured on this host
at batch 1 that is **0.163 ms** against **0.018 ms** for the model itself — 90 %
of the per-window cost spent assembling the input to a model that is nearly free.

This is the same shape of trap as `sequence_1d` in `nemo_speaker_vector.py`: the
tensor work was never the problem, the Python around it was. One
`sliding_window_view` and one batched `rfft` bring the front end to **0.079 ms**,
so a window costs ~0.097 ms end to end instead of ~0.181 ms.

**Bit-exact with upstream**, deliberately — `tests/test_pulse_vad.py` asserts it
against a transcription of `frontend_np.py`. That is why the power spectrum is
written `np.abs(spec) ** 2` and not the marginally more accurate
`spec.real**2 + spec.imag**2`: the sqrt-then-square round trip is what upstream
does, it costs nothing here, and matching it exactly means a future divergence
shows up as a test failure rather than as a small unexplained shift in p(speech).

The recipe, fixed by the checkpoint: pre-emphasis 0.97 -> whole-window z-norm ->
400-sample periodic Hann centred in a 512-point FFT, hop 160 -> 64 mel bins ->
log -> per-bin z-norm across the 21 frames.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["WINDOW_SAMPLES", "SAMPLE_RATE", "N_MELS", "N_FRAMES", "log_mel"]

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 3200
"""200 ms. Not a tunable: the model's input is a fixed 64 x 21 patch."""

N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 64
N_FRAMES = 21
PREEMPHASIS_ALPHA = 0.97
EPS = 1e-5

_PAD_WIN = (N_FFT - WIN_LENGTH) // 2
"""56 zeros each side: torchaudio centres a 400-sample window in a 512-point FFT."""

_HANN = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN_LENGTH) / WIN_LENGTH)).astype(
    np.float32
)
"""Periodic Hann, matching `torch.hann_window(400, periodic=True)`. `np.hanning`
is the *symmetric* window and would be a different front end."""


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    """The (257, 64) filterbank that ships with the checkpoint, loaded once."""
    path = Path(__file__).parent / "resources" / "mel_filterbank.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"PulseVAD mel filterbank missing from the package at {path}. "
            "Reinstall stt-api."
        )
    return np.load(path).astype(np.float32)


def log_mel(waveform: np.ndarray) -> np.ndarray:
    """
    `(3200,)` or `(B, 3200)` of float32 PCM at 16 kHz -> `(B, 64, 21)` float32.

    Mono, in [-1, 1]. The per-window z-norm makes the result scale-invariant, so
    input gain does not shift p(speech) — but it also means every window is
    normalised against *itself*, which is why a digital-silence window is a
    well-defined input rather than a divide-by-zero (it scores 0.008).
    """
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2 or x.shape[1] != WINDOW_SAMPLES:
        raise ValueError(
            f"expected (B, {WINDOW_SAMPLES}) or ({WINDOW_SAMPLES},) samples, got {x.shape}"
        )

    # 1. pre-emphasis, y[0] = x[0]
    pre = np.empty_like(x)
    pre[:, 0] = x[:, 0]
    pre[:, 1:] = x[:, 1:] - PREEMPHASIS_ALPHA * x[:, :-1]

    # 2. whole-window z-norm (population std, matching torch's default)
    pre = (pre - pre.mean(-1, keepdims=True)) / (pre.std(-1, keepdims=True) + EPS)

    # 3. centred STFT: reflect-pad, then 21 overlapping Hann frames in one view
    padded = np.pad(pre, ((0, 0), (N_FFT // 2, N_FFT // 2)), mode="reflect")
    offset = padded[:, _PAD_WIN:]
    frames = np.lib.stride_tricks.sliding_window_view(offset, WIN_LENGTH, axis=-1)
    frames = frames[:, ::HOP_LENGTH][:, :N_FRAMES]  # (B, 21, 400)

    buf = np.zeros((x.shape[0], N_FRAMES, N_FFT), dtype=np.float32)
    buf[:, :, _PAD_WIN : _PAD_WIN + WIN_LENGTH] = frames * _HANN
    spec = np.fft.rfft(buf, n=N_FFT, axis=-1)
    power = np.abs(spec) ** 2  # see module docstring: matches upstream exactly

    # 4. mel, log, then per-bin z-norm across the 21 frames
    mel = np.log((power @ mel_filterbank()).transpose(0, 2, 1) + EPS)  # (B, 64, 21)
    mel = (mel - mel.mean(-1, keepdims=True)) / (mel.std(-1, keepdims=True) + EPS)
    return mel.astype(np.float32)
