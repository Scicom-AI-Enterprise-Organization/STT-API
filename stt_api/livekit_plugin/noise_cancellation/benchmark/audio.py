"""
Audio primitives shared by the harness: framing, streaming resampling, alignment.

Nothing here is benchmark-specific arithmetic — it is the plumbing that has to be
right before any metric means anything. In particular `estimate_delay`: every
metric that compares against a clean reference (PESQ, STOI, SI-SDR) collapses if
the two signals are misaligned by even a few milliseconds, and each enhancer here
imposes a different algorithmic delay. Getting that number wrong does not produce
an obviously broken benchmark, it produces a plausible one that ranks the models
by how well their latency happens to match.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

__all__ = [
    "StreamResampler",
    "align",
    "estimate_delay",
    "iter_frames",
    "mulaw_roundtrip",
    "rms",
    "rms_db",
    "telephony_degrade",
]


def iter_frames(x: np.ndarray, n: int) -> Iterator[np.ndarray]:
    """
    Yield fixed-length frames, zero-padding the final partial one.

    LiveKit hands a filter whatever the SFU delivers, always the same size for a
    given stream (`frame_size_ms=50` by default), so a fixed frame is the honest
    shape to drive with. The tail is padded rather than dropped so that the output
    length matches the input length exactly and the metrics do not silently score
    a truncated signal.
    """
    for i in range(0, len(x), n):
        chunk = x[i : i + n]
        if len(chunk) < n:
            chunk = np.concatenate((chunk, np.zeros(n - len(chunk), dtype=np.float32)))
        yield chunk


class StreamResampler:
    """
    Stateful sample-rate conversion, for enhancers whose native rate is not the
    rate the pipeline runs at.

    Stateful is the point. Resampling each frame independently would restart the
    polyphase filter at every boundary and stitch discontinuities into the output
    at the frame rate — audible as a buzz, and enough to move PESQ by a full
    point. `soxr.ResampleStream` carries its filter state across calls.

    Quality is chosen by *direction*, not by taste, following the same reasoning
    the GTCRN plugin documents for its own resamplers — and here it is a fairness
    requirement, not just a tuning nicety. Resampler group delay lands in the
    measured latency of whichever enhancer needs the conversion, so picking a
    uniformly high quality would charge the 48 kHz models (RNNoise,
    DeepFilterNet) tens of milliseconds that belong to this class rather than to
    them. Measured with `HQ` in both directions, RNNoise reported 90 ms of delay,
    the large majority of it soxr's.

    * **Upsampling** cannot alias — the signal is already band-limited below the
      new Nyquist — so `QQ` is free.
    * **Downsampling** needs a real anti-alias filter, or everything above the new
      Nyquist folds back into the band the model is trying to clean. `LQ` filters
      properly at a fraction of `HQ`'s delay.
    """

    def __init__(self, in_rate: int, out_rate: int, quality: str | None = None) -> None:
        import soxr

        self.in_rate = in_rate
        self.out_rate = out_rate
        self.quality = quality or ("QQ" if out_rate >= in_rate else "LQ")
        self._passthrough = in_rate == out_rate
        self._rs = (
            None
            if self._passthrough
            else soxr.ResampleStream(
                in_rate, out_rate, 1, dtype="float32", quality=self.quality
            )
        )

    def push(self, x: np.ndarray) -> np.ndarray:
        if self._passthrough:
            return x.astype(np.float32, copy=False)
        return self._rs.resample_chunk(x.astype(np.float32, copy=False))  # type: ignore[union-attr]


def mulaw_roundtrip(x: np.ndarray) -> np.ndarray:
    """
    G.711 µ-law encode/decode, the quantisation an 8 kHz phone call carries.

    Implemented here rather than through `audioop` because that module was
    removed in Python 3.13. This is the standard ITU-T G.711 µ-law curve with
    µ=255: a logarithmic companding that gives roughly 8 bits of perceptual
    resolution over a 14-bit range, so quiet detail is coarsened far more than
    loud detail. Combined with the 8 kHz band limit it is most of what makes
    telephony audio hard for both denoisers and ASR.
    """
    MU = 255.0
    x = np.clip(x.astype(np.float32), -1.0, 1.0)
    # encode: companding curve, then quantise to 8 bits
    mag = np.log1p(MU * np.abs(x)) / np.log1p(MU)
    q = np.round(np.clip(mag, 0.0, 1.0) * 127.0) / 127.0
    # decode: inverse companding
    return (np.sign(x) * (np.expm1(q * np.log1p(MU)) / MU)).astype(np.float32)


def telephony_degrade(x: np.ndarray, rate: int) -> np.ndarray:
    """
    Put a signal through a narrowband phone path: band-limit to 8 kHz, µ-law, back.

    This is the condition inbound SIP audio actually arrives in, and it is the
    domain the call-centre restorers are trained for. Evaluating them on
    wideband studio speech instead measures how gracefully they handle
    out-of-domain input, which is a different and much less useful question —
    CallEnhancer scores *below* unprocessed on wideband VoiceBank for exactly
    this reason.
    """
    if rate == 8000:
        return mulaw_roundtrip(x)
    down = StreamResampler(rate, 8000).push(x)
    coded = mulaw_roundtrip(down)
    up = StreamResampler(8000, rate).push(coded)
    # Resamplers shift length slightly; hold the original length so the clean
    # reference stays aligned and the metrics compare like with like.
    if len(up) < len(x):
        up = np.concatenate((up, np.zeros(len(x) - len(up), dtype=np.float32)))
    return up[: len(x)]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(max(rms(x), 1e-12))


def estimate_delay(ref: np.ndarray, test: np.ndarray, max_lag: int) -> int:
    """
    Integer sample delay of `test` relative to `ref`, searched over 0..max_lag.

    Two estimators are run and the more trustworthy answer wins:

    * **GCC-PHAT** — whitens both spectra before correlating, so the peak is
      driven by phase alignment alone. That is what makes it right for this job:
      a denoiser changes the *magnitude* spectrum drastically by design, which
      drags a plain correlation peak toward whichever lag best matches the new
      spectral tilt rather than the true delay.
    * **Plain normalised cross-correlation** — the fallback, because PHAT's
      whitening amplifies bins that hold nothing but noise, and on heavily
      suppressed output (a good denoiser in a pause) that can produce a confident
      peak at the wrong lag.

    Both candidates are then scored by the *same* plain normalised correlation and
    the better one is returned, so the two estimators can disagree without the
    caller having to care which was used.

    Only non-negative lags are searched. A causal filter can delay audio; it
    cannot advance it, and admitting negative lags would let a bad estimate
    "improve" a score by shifting the reference instead.
    """
    n = min(len(ref), len(test))
    if n == 0:
        return 0
    a = ref[:n].astype(np.float64)
    b = test[:n].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    max_lag = max(0, min(max_lag, n - 1))
    if max_lag == 0 or not np.any(a) or not np.any(b):
        return 0

    size = 1 << int(np.ceil(np.log2(2 * n)))
    A = np.fft.rfft(a, size)
    B = np.fft.rfft(b, size)

    cross = np.conj(A) * B
    phat = np.fft.irfft(cross / np.maximum(np.abs(cross), 1e-12), size)[: max_lag + 1]
    plain = np.fft.irfft(cross, size)[: max_lag + 1]

    candidates = {int(np.argmax(phat)), int(np.argmax(plain))}

    def score(lag: int) -> float:
        m = n - lag
        if m <= 0:
            return -np.inf
        x, y = a[:m], b[lag : lag + m]
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return float(np.dot(x, y) / denom) if denom > 0 else -np.inf

    return max(candidates, key=score)


def align(ref: np.ndarray, test: np.ndarray, delay: int) -> tuple[np.ndarray, np.ndarray]:
    """Drop `delay` samples off the front of `test`, then trim both to one length."""
    t = test[delay:] if delay else test
    n = min(len(ref), len(t))
    return ref[:n], t[:n]
