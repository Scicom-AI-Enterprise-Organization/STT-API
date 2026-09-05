"""
Does the filter protect turn-taking? — the axis WER cannot see.

The agent's turn boundary is decided by `audio -> VAD -> ASR -> EOT(text)`. The
end-of-turn model in `turn_detector/multilingual.py` reads a ChatML prompt built
from the *transcript*, so anything the VAD admits and the ASR transcribes lands
in the text that decides when the user stopped speaking. A competing voice in the
background therefore corrupts turn boundaries even when the target speaker's own
words come through perfectly — and WER on the target speaker, the metric
`asr.py` reports, is completely blind to it.

That makes background *speech* the failure mode that matters, and it is not the
one the models in `enhancers.py` were built for. GTCRN, DTLN and RNNoise are
trained to separate speech from **non-speech** noise. Background speech is
speech: there is no reason for them to remove it, and the measurements here say
they largely do not. Removing it is a different capability — target-speaker
extraction, what Krisp markets as BVC — which needs either speaker enrolment or a
model trained for it.

Two numbers, and the second is the one to read:

* **suppression** — how far the interferer's level drops in regions where it is
  talking and the target is not.
* **selectivity** — that suppression *minus* the suppression applied to the
  target's own speech. This is the honest figure. A filter that simply turns
  everything down scores well on suppression and zero on selectivity, and a
  filter that scores zero here has not distinguished the two voices at all,
  whatever it did to the level.

Plus the operational consequence: **VAD false-trigger rate**, the fraction of
interferer-only audio that Silero still calls speech, at this repo's own
production thresholds. That is the number that turns into a spurious turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .audio import rms

__all__ = ["CrosstalkScore", "score_crosstalk", "vad_speech_mask"]

# This repo's production VAD settings, from stt_api/main.py. Using anything else
# would measure a VAD nobody runs.
_VAD_THRESHOLD = 0.5
_MIN_SPEECH_MS = 250
_MIN_SILENCE_MS = 400


@dataclass(frozen=True)
class CrosstalkScore:
    suppression_db: float
    """Interferer level change where it speaks alone. Negative is suppression."""
    target_db: float
    """Target level change where the target speaks alone. Negative is damage."""
    selectivity_db: float
    """`target_db - suppression_db`. Positive means the interferer was attenuated
    more than the target — i.e. the filter actually told the voices apart."""
    vad_false_trigger: float
    """Fraction of interferer-only audio Silero still calls speech, after."""
    vad_false_trigger_before: float
    """The same before enhancement, so the delta is readable."""


@lru_cache(maxsize=1)
def _silero():
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=True)


def vad_speech_mask(x: np.ndarray, rate: int) -> np.ndarray:
    """Per-sample boolean speech mask from Silero, at production thresholds."""
    import torch
    from silero_vad import get_speech_timestamps

    mask = np.zeros(len(x), dtype=bool)
    if len(x) < rate // 100:
        return mask
    ts = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        _silero(),
        sampling_rate=rate,
        threshold=_VAD_THRESHOLD,
        min_speech_duration_ms=_MIN_SPEECH_MS,
        min_silence_duration_ms=_MIN_SILENCE_MS,
    )
    for seg in ts:
        mask[int(seg["start"]) : int(seg["end"])] = True
    return mask


def _active(x: np.ndarray, rate: int, rel_db: float = -30.0) -> np.ndarray:
    """
    Energy-based activity mask, per 20 ms frame, relative to the signal's own peak
    frame. Deliberately not a VAD: this defines *ground truth* about where each
    source has energy, and using a neural VAD here would make the reference
    depend on the very thing being tested.
    """
    win = max(int(0.02 * rate), 1)
    n = len(x) // win * win
    if n < win:
        return np.zeros(len(x), dtype=bool)
    frames = x[:n].reshape(-1, win)
    energy = 20.0 * np.log10(np.maximum(np.sqrt((frames.astype(np.float64) ** 2).mean(1)), 1e-12))
    live = energy > (energy.max() + rel_db)
    mask = np.repeat(live, win)
    return np.concatenate((mask, np.zeros(len(x) - n, dtype=bool)))


def _level_change_db(before: np.ndarray, after: np.ndarray, region: np.ndarray) -> float:
    if region.sum() < 100:
        return float("nan")
    n = min(len(before), len(after), len(region))
    r = region[:n]
    b, a = rms(before[:n][r]), rms(after[:n][r])
    if b <= 0:
        return float("nan")
    return 20.0 * np.log10(max(a, 1e-12) / b)


def score_crosstalk(
    noisy: np.ndarray,
    enhanced: np.ndarray,
    clean: np.ndarray,
    interferer: np.ndarray,
    rate: int,
    delay: int = 0,
) -> CrosstalkScore:
    """
    Compare regions where only the interferer speaks against regions where only
    the target does.

    `delay` compensates the enhancer's algorithmic latency, without which the
    regions would be offset against the output and every number here would be a
    blend of the two conditions.
    """
    est = enhanced[delay:] if delay else enhanced
    n = min(len(noisy), len(est), len(clean), len(interferer))
    noisy, est = noisy[:n], est[:n]
    clean, interferer = clean[:n], interferer[:n]

    tgt_on = _active(clean, rate)
    int_on = _active(interferer, rate)
    interferer_only = int_on & ~tgt_on
    target_only = tgt_on & ~int_on

    supp = _level_change_db(noisy, est, interferer_only)
    tgt = _level_change_db(noisy, est, target_only)
    sel = (tgt - supp) if np.isfinite(supp) and np.isfinite(tgt) else float("nan")

    def false_trigger(sig: np.ndarray) -> float:
        if interferer_only.sum() < rate // 10:  # under 100 ms, not worth a rate
            return float("nan")
        speech = vad_speech_mask(sig, rate)
        m = min(len(speech), len(interferer_only))
        return float((speech[:m] & interferer_only[:m]).sum() / interferer_only[:m].sum())

    return CrosstalkScore(
        suppression_db=supp,
        target_db=tgt,
        selectivity_db=sel,
        vad_false_trigger=false_trigger(est),
        vad_false_trigger_before=false_trigger(noisy),
    )
