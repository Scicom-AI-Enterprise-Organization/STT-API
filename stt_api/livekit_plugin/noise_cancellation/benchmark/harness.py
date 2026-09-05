"""
The measurement loop.

One rule governs the whole file: every enhancer is measured under identical
conditions, and the conditions are the ones a LiveKit agent actually imposes.
Same frames, same order, same corpus items, same alignment procedure, same
scoring code. Where an enhancer cannot meet those conditions — `dfn3` — it is
reported separately rather than quietly given a different deal.

The cost numbers deserve a note, because the obvious one is the wrong one. RTF
(total CPU over total audio) is what gets published, but it is not what breaks a
call. `_process` runs inline on the audio read loop, so what matters is the
*worst* frame, not the average: a filter with an excellent RTF that occasionally
takes 60 ms on a 20 ms frame will stutter, and the mean will never show it. Hence
p50/p95/p99/max per frame, and a `real_time_budget` figure that compares p99
against the frame duration it had to fit inside.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field

import numpy as np

from .audio import align, estimate_delay, iter_frames, rms_db
from .corpus import Item
from .enhancers import Enhancer
from .metrics import Quality, score_quality

__all__ = ["ItemResult", "Summary", "run_item", "summarize"]

_MAX_LAG_MS = 250
"""Widest delay the aligner will consider. Comfortably past every candidate here
(the worst measured is RNNoise at 55 ms) while staying far short of the point
where a spurious correlation peak on a periodic signal becomes plausible."""


@dataclass
class ItemResult:
    enhancer: str
    item_id: str
    source: str
    audio_seconds: float
    delay_samples: int | None
    rate: int
    quality: Quality | None = None
    dnsmos: object | None = None
    frame_times_ms: np.ndarray | None = None
    noise_floor_db: float | None = None
    level_change_db: float | None = None
    crosstalk: object | None = None
    enhanced: np.ndarray | None = None
    error: str | None = None

    @property
    def delay_ms(self) -> float:
        # None, not 0.0, when there is no clean reference to align against.
        # Reporting an unmeasured delay as zero would read as "adds no latency",
        # which is the opposite of what not knowing means.
        if self.delay_samples is None:
            return float("nan")
        return self.delay_samples / self.rate * 1000.0


def _noise_floor_db(clean: np.ndarray, est: np.ndarray, rate: int) -> float | None:
    """
    Residual level in the quietest tenth of the reference, relative to input.

    Aggregate SNR hides the thing users complain about. A denoiser can post a fine
    SNR while leaving an audible hiss between words, and it is that hiss — not the
    in-speech noise — that makes a call sound unprocessed and that a VAD trips on.
    Measuring where the reference is quietest isolates it.
    """
    win = max(int(0.02 * rate), 1)
    n = min(len(clean), len(est)) // win * win
    if n < win * 10:
        return None
    frames = np.abs(clean[:n]).reshape(-1, win).mean(axis=1)
    quiet = np.argsort(frames)[: max(1, len(frames) // 10)]
    seg = est[:n].reshape(-1, win)[quiet]
    return rms_db(seg.reshape(-1))


def run_item(
    enh: Enhancer,
    item: Item,
    *,
    frame_ms: int = 20,
    dnsmos=None,
    keep_audio: bool = False,
) -> ItemResult:
    """
    Push one corpus item through one enhancer and score it.

    `enh.reset()` first, always: recurrent state carried over from the previous
    utterance would let a model arrive already adapted to the noise, which is a
    real advantage in a long call and a fabricated one on a benchmark of
    independent clips.
    """
    res = ItemResult(
        enhancer=enh.name,
        item_id=item.id,
        source=item.source,
        audio_seconds=len(item.noisy) / item.rate,
        delay_samples=None,
        rate=item.rate,
    )
    try:
        enh.reset()
        n = item.rate * frame_ms // 1000

        if enh.streaming:
            times = np.empty(0, dtype=np.float64)
            out: list[np.ndarray] = []
            frames = list(iter_frames(item.noisy, n))
            times = np.empty(len(frames), dtype=np.float64)
            gc_was_on = gc.isenabled()
            gc.disable()  # a collection landing mid-frame lands in the p99
            try:
                for i, f in enumerate(frames):
                    t0 = time.perf_counter()
                    y = enh.process(f, item.rate)
                    times[i] = (time.perf_counter() - t0) * 1000.0
                    out.append(y)
            finally:
                if gc_was_on:
                    gc.enable()
            enhanced = np.concatenate(out)[: len(item.noisy)] if out else np.zeros(0, np.float32)
            res.frame_times_ms = times
        else:
            enhanced = enh.process_all(item.noisy, item.rate)

        if enhanced.size == 0:
            res.error = "produced no audio"
            return res

        res.level_change_db = rms_db(enhanced) - rms_db(item.noisy)

        if item.clean is not None:
            max_lag = int(_MAX_LAG_MS * item.rate / 1000)
            res.delay_samples = estimate_delay(item.clean, enhanced, max_lag)
            ref, est = align(item.clean, enhanced, res.delay_samples)
            res.quality = score_quality(ref, est, item.rate)
            res.noise_floor_db = _noise_floor_db(ref, est, item.rate)

        if item.interferer is not None and item.clean is not None:
            from .turntaking import score_crosstalk

            res.crosstalk = score_crosstalk(
                item.noisy, enhanced, item.clean, item.interferer, item.rate,
                delay=res.delay_samples or 0,
            )

        if dnsmos is not None:
            res.dnsmos = dnsmos(enhanced, item.rate)
        if keep_audio:
            res.enhanced = enhanced
    except Exception as e:  # noqa: BLE001 - one bad item must not abort a long run
        res.error = f"{type(e).__name__}: {e}"
    return res


@dataclass
class Summary:
    """Per-enhancer aggregate. NaNs are dropped, not zero-filled."""

    enhancer: str
    items: int = 0
    failed: int = 0
    audio_seconds: float = 0.0
    quality: dict[str, float] = field(default_factory=dict)
    dnsmos: dict[str, float] = field(default_factory=dict)
    crosstalk: dict[str, float] = field(default_factory=dict)
    delay_ms: float = float("nan")
    rtf: float = float("nan")
    frame_ms: dict[str, float] = field(default_factory=dict)
    noise_floor_db: float = float("nan")
    level_change_db: float = float("nan")
    streaming: bool = True
    generative: bool = False
    wer: float | None = None
    wer_median: float | None = None
    asr_loops: int = 0
    errors: list[str] = field(default_factory=list)


def _mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def summarize(
    results: list[ItemResult],
    *,
    streaming: bool = True,
    generative: bool = False,
    frame_ms: int = 20,
) -> Summary:
    name = results[0].enhancer if results else "?"
    ok = [r for r in results if r.error is None]
    bad = [r for r in results if r.error is not None]
    s = Summary(
        enhancer=name,
        items=len(ok),
        failed=len(bad),
        audio_seconds=sum(r.audio_seconds for r in ok),
        streaming=streaming,
        generative=generative,
        errors=sorted({r.error for r in bad if r.error})[:3],
    )
    if not ok:
        return s

    for key in ("pesq", "stoi", "estoi", "si_sdr"):
        s.quality[key] = _mean([getattr(r.quality, key) for r in ok if r.quality])
    if any(r.dnsmos for r in ok):
        for key in ("sig", "bak", "ovrl", "p808"):
            s.dnsmos[key] = _mean([getattr(r.dnsmos, key) for r in ok if r.dnsmos])

    if any(r.crosstalk for r in ok):
        for key in (
            "suppression_db",
            "target_db",
            "selectivity_db",
            "vad_false_trigger",
            "vad_false_trigger_before",
        ):
            s.crosstalk[key] = _mean([getattr(r.crosstalk, key) for r in ok if r.crosstalk])

    s.delay_ms = _mean([r.delay_ms for r in ok])
    s.noise_floor_db = _mean([r.noise_floor_db for r in ok])
    s.level_change_db = _mean([r.level_change_db for r in ok])

    times = [r.frame_times_ms for r in ok if r.frame_times_ms is not None]
    if times:
        allt = np.concatenate(times)
        total_ms = float(allt.sum())
        s.rtf = total_ms / 1000.0 / s.audio_seconds if s.audio_seconds else float("nan")
        s.frame_ms = {
            "p50": float(np.percentile(allt, 50)),
            "p95": float(np.percentile(allt, 95)),
            "p99": float(np.percentile(allt, 99)),
            "max": float(allt.max()),
            # Fraction of the frame's own duration consumed at p99. Above 1.0 the
            # filter cannot keep up with its worst frames and the call stutters;
            # anything above ~0.3 leaves no headroom for the STT, VAD and LLM
            # sharing the same core.
            "budget_p99": float(np.percentile(allt, 99)) / frame_ms,
        }
    return s
