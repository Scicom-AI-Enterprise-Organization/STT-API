"""
Reference-based quality metrics: PESQ, STOI, ESTOI, SI-SDR.

All four compare the enhanced signal against the clean reference, and all four
assume the two are sample-aligned. They are not, before the harness compensates
for each enhancer's algorithmic delay — see `audio.estimate_delay`. A 32 ms
misalignment is enough to drag wideband PESQ down by roughly a point, which is
larger than the gap between any two models here. Alignment is not a refinement;
without it the ranking is noise.

What each one is actually good for, since they disagree often and it matters
which you believe:

* **PESQ (wideband, ITU-T P.862.2)** — perceptual quality. Well correlated with
  listening tests for codecs and additive noise, which is what VoiceBank+DEMAND
  is. Its blind spot is the artefacts neural suppressors invent; it does not
  reliably punish musical noise.
* **STOI / ESTOI** — short-time objective *intelligibility*, i.e. how much of the
  message survives, not how pleasant it sounds. ESTOI is the later variant, and
  the one to prefer for modulated and babble noise where plain STOI saturates.
* **SI-SDR** — scale-invariant signal-to-distortion ratio. A pure waveform-fidelity
  number with no perceptual model at all, which is exactly why it is worth having:
  it is the one metric here that a model cannot flatter by making a pleasant-
  sounding signal that is not the original speech.

None of them predicts WER, which is why `asr.py` exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["Quality", "score_quality"]


@dataclass(frozen=True)
class Quality:
    pesq: float
    stoi: float
    estoi: float
    si_sdr: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """
    Scale-invariant SDR in dB.

    The projection onto `ref` removes any overall gain difference before measuring
    error, so a model is neither rewarded nor punished for changing the level —
    which several of these do, and which is trivially fixable downstream.
    """
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    ref = ref - ref.mean()
    est = est - est.mean()
    energy = float(np.dot(ref, ref))
    if energy <= 0:
        return float("nan")
    target = (float(np.dot(est, ref)) / energy) * ref
    noise = est - target
    num, den = float(np.dot(target, target)), float(np.dot(noise, noise))
    if num <= 0 or den <= 0:
        return float("nan")
    return 10.0 * np.log10(num / den)


def score_quality(ref: np.ndarray, est: np.ndarray, rate: int) -> Quality:
    """
    Score an already-aligned pair. Any individual metric that cannot be computed
    comes back as NaN rather than as a substituted value, so a broken metric shows
    up as a hole in the table instead of silently averaging into the result.
    """
    n = min(len(ref), len(est))
    ref, est = ref[:n].astype(np.float32), est[:n].astype(np.float32)

    try:
        from pesq import pesq as _pesq

        # Wideband PESQ is defined only at 16 kHz; narrowband at 8 kHz. There is
        # no 48 kHz mode, so anything else has to be resampled rather than
        # silently scored with the wrong mode.
        if rate == 16000:
            pesq_v = float(_pesq(16000, ref, est, "wb"))
        elif rate == 8000:
            pesq_v = float(_pesq(8000, ref, est, "nb"))
        else:
            from .audio import StreamResampler

            r = StreamResampler(rate, 16000).push(ref)
            e = StreamResampler(rate, 16000).push(est)
            m = min(len(r), len(e))
            pesq_v = float(_pesq(16000, r[:m], e[:m], "wb"))
    except Exception:
        # pesq raises on silent or degenerate input (NoUtterancesError), which is
        # a real condition on aggressive suppression, not a bug to crash on.
        pesq_v = float("nan")

    try:
        from pystoi import stoi as _stoi

        stoi_v = float(_stoi(ref, est, rate, extended=False))
        estoi_v = float(_stoi(ref, est, rate, extended=True))
    except Exception:
        stoi_v = estoi_v = float("nan")

    return Quality(pesq=pesq_v, stoi=stoi_v, estoi=estoi_v, si_sdr=si_sdr(ref, est))
