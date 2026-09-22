"""
Scoring: what changes for the agent if silero is swapped for PulseVAD.

Three families of number, never mixed in the report, in descending order of how
much they can be trusted:

**Real ground truth — clip level.** Production turns carry a Whisper transcript.
A clip whose transcript is empty contains no speech, so *any* speech reported in
it is a false fire; a clip with a transcript contains speech, so reporting none
is a miss. `neg_fire_rate` and `pos_detect_rate` are those two, and they are the
numbers that decide the swap: a VAD that fires on silence makes an agent
interrupt itself, and one that misses turns makes it deaf.

**Real ground truth — frame level, guarded.** `pad_fp_rate` counts firing inside
the constructed pads, *excluding a guard band* after speech. That exclusion is
not a fudge: `min_silence_duration` keeps a VAD "speaking" for 0.55 s after the
last word by design, and counting that as a false positive rates the *setting*
rather than the model — it made both detectors look ~20 % wrong in an earlier
draft of this file.

**No truth at all.** Agreement with silero. Silero is the incumbent, not an
oracle; `agreement` answers "would the agent behave differently", which is the
real question when someone proposes a swap.

Two methodological points decide whether any of it is valid:

* **Each model is scored at its own best threshold, never a shared one.**
  PulseVAD's p(speech) saturates at 0.711 and silero's at ~1.0, so scoring both
  at 0.5 compares silero's mid-range against PulseVAD's 70th percentile and
  reports the difference as accuracy. That is a measurement of output scale.
* **`segment` replays the streaming state machine offline** instead of
  re-running inference per threshold, mirroring `vad.py` exactly. `verify_replay`
  asserts it reproduces the live stream's turns at the default threshold; if
  that ever fails, every swept threshold but the streamed one is fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Scores",
    "agreement",
    "wilson",
    "balanced_score",
    "paired_timing",
    "score_run",
    "segment",
    "sweep",
    "verify_replay",
]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    95 % Wilson interval for a proportion.

    The clip-level rates here run on small denominators — 180 no-speech clips
    means one clip is 0.56 pp — so a difference of "1.67 pp" is three clips and
    well inside noise. Printing the interval next to the rate is the cheapest
    way to stop that being read as a real gap. Wilson rather than the normal
    approximation because these proportions sit near 0 and 1, where the normal
    interval runs off the end of the scale.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def segment(
    times: np.ndarray,
    probs: np.ndarray,
    *,
    activation: float,
    deactivation: float | None = None,
    min_speech: float = 0.05,
    min_silence: float = 0.55,
) -> list[tuple[float, float]]:
    """
    Replay the VADStream state machine -> speaking intervals `[(start, end), ...]`.

    Mirrors `vad.py::_main_task` and silero's equivalent: hysteresis between
    `activation` and `deactivation`, then a duration gate in each direction.
    """
    if len(times) == 0:
        return []
    deact = deactivation if deactivation is not None else max(activation - 0.1, 0.01)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.032

    out: list[tuple[float, float]] = []
    speaking = False
    speech_acc = silence_acc = 0.0
    start = 0.0
    for t, p in zip(times, probs):
        if p >= activation or (speaking and p > deact):
            speech_acc += dt
            silence_acc = 0.0
            if not speaking and speech_acc >= min_speech:
                speaking, start = True, t
        else:
            silence_acc += dt
            speech_acc = 0.0
            if speaking and silence_acc >= min_silence:
                speaking = False
                out.append((start, t))
    if speaking:
        out.append((start, float(times[-1])))
    return out


def _mask(intervals: list[tuple[float, float]], times: np.ndarray) -> np.ndarray:
    m = np.zeros(len(times), dtype=bool)
    for a, b in intervals:
        m |= (times >= a) & (times < b)
    return m


@dataclass
class Scores:
    label: str
    threshold: float

    # --- real ground truth, clip level -------------------------------------
    neg_fire_rate: float = float("nan")
    """Fraction of no-speech clips in which the VAD reported speech. Lower is better."""
    neg_clips: int = 0
    neg_fired: int = 0
    pos_detect_rate: float = float("nan")
    """Fraction of speech clips in which the VAD reported speech. Higher is better."""
    pos_clips: int = 0
    pos_found: int = 0

    # --- real ground truth, frame level (guarded) --------------------------
    pad_fp_rate: float = float("nan")
    hit_rate: float = float("nan")
    """Frames of the speech-bearing body called speech. An upper bound, not accuracy."""
    f1: float = float("nan")

    # --- behaviour ----------------------------------------------------------
    spurious_turns: int = 0
    onset_ms: float = float("nan")
    onset_p90_ms: float = float("nan")
    release_ms: float = float("nan")
    """Excess hangover: END_OF_SPEECH minus body end minus the configured min_silence."""
    frames: int = 0
    per_item: dict = field(default_factory=dict, repr=False)


def score_run(
    runs: list,
    items: list,
    *,
    activation: float,
    guard: float | None = None,
    min_speech: float = 0.05,
    min_silence: float = 0.55,
) -> Scores:
    """Aggregate clip-level and frame-level scores for one model over the corpus."""
    seg_kw = {"min_speech": min_speech, "min_silence": min_silence}
    guard = min_silence + 0.1 if guard is None else guard

    tp = fp = tn = fn = 0
    spurious = 0
    neg_fired = neg_total = pos_found = pos_total = 0
    onsets: list[float] = []
    releases: list[float] = []

    for run, item in zip(runs, items):
        if len(run.times) == 0:
            continue
        intervals = segment(run.times, run.probs, activation=activation, **seg_kw)
        said = _mask(intervals, run.times)
        neg = item.certain_negative(run.times, guard=guard)
        expect = item.labels(run.times)

        tp += int(np.sum(said & expect))
        fn += int(np.sum(~said & expect))
        fp += int(np.sum(said & neg))
        tn += int(np.sum(~said & neg))

        body = item.body
        if item.kind == "negative":
            neg_total += 1
            if intervals:
                neg_fired += 1
                spurious += len(intervals)
        elif body is not None:
            pos_total += 1
            starts = [a for a, _ in intervals]
            in_body = [a for a in starts if body.start - 0.05 <= a <= body.end]
            spurious += sum(1 for a in starts if a < body.start - 0.05 or a > body.end)
            if in_body:
                pos_found += 1
                onsets.append((min(in_body) - body.start) * 1000.0)
                ends = [b for _, b in intervals if b >= body.start]
                if ends:
                    releases.append((max(ends) - body.end - min_silence) * 1000.0)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return Scores(
        label=runs[0].label if runs else "?",
        threshold=activation,
        neg_fire_rate=neg_fired / neg_total if neg_total else float("nan"),
        neg_clips=neg_total,
        neg_fired=neg_fired,
        pos_detect_rate=pos_found / pos_total if pos_total else float("nan"),
        pos_clips=pos_total,
        pos_found=pos_found,
        pad_fp_rate=fp / (fp + tn) if fp + tn else float("nan"),
        hit_rate=rec,
        f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        spurious_turns=spurious,
        onset_ms=float(np.median(onsets)) if onsets else float("nan"),
        onset_p90_ms=float(np.percentile(onsets, 90)) if onsets else float("nan"),
        release_ms=float(np.median(releases)) if releases else float("nan"),
        frames=tp + fp + tn + fn,
    )


def sweep(runs: list, items: list, thresholds, **kw) -> list[Scores]:
    """Score every threshold so each model can be read at its own best point."""
    return [score_run(runs, items, activation=float(t), **kw) for t in thresholds]


def balanced_score(s: Scores) -> float:
    """
    Operating-point objective: catch turns without firing on silence.

    Plain F1 is dominated by the speech-bearing body, which both detectors get
    mostly right, so it barely moves across thresholds and picks an operating
    point almost at random. This trades clip-level detection against clip-level
    false fires, which is what an agent actually experiences.
    """
    det = 0.0 if np.isnan(s.pos_detect_rate) else s.pos_detect_rate
    fire = 0.0 if np.isnan(s.neg_fire_rate) else s.neg_fire_rate
    return det - fire


def agreement(
    runs_a: list,
    runs_b: list,
    items: list,
    *,
    act_a: float,
    act_b: float,
    min_speech: float = 0.05,
    min_silence: float = 0.55,
) -> dict[str, float]:
    """
    How often the two detectors agree on "is the user speaking", frame by frame.

    The two disagreement directions are kept apart: `b_only` (PulseVAD speaking
    where silero is not) costs an agent truncated replies, `a_only` costs it
    deafness. They are not interchangeable, and a single "97 % agree" hides
    which one you bought.
    """
    seg_kw = {"min_speech": min_speech, "min_silence": min_silence}
    agree = total = both = only_a = only_b = 0
    for ra, rb, _item in zip(runs_a, runs_b, items):
        n = min(len(ra.times), len(rb.times))
        if n == 0:
            continue
        ta = ra.times[:n]
        ma = _mask(segment(ra.times, ra.probs, activation=act_a, **seg_kw), ta)
        mb = _mask(segment(rb.times, rb.probs, activation=act_b, **seg_kw), ta)
        agree += int(np.sum(ma == mb))
        total += n
        both += int(np.sum(ma & mb))
        only_a += int(np.sum(ma & ~mb))
        only_b += int(np.sum(~ma & mb))

    if not total:
        return dict.fromkeys(("agreement", "kappa", "a_only", "b_only"), float("nan"))
    po = agree / total
    pa1, pb1 = (both + only_a) / total, (both + only_b) / total
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return {
        "agreement": po,
        "kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0,
        "a_only": only_a / total,
        "b_only": only_b / total,
    }


def verify_replay(run, *, activation: float, **seg_kw) -> tuple[int, int]:
    """
    Cross-check `segment` against the live stream: (replayed turns, real turns).

    The sweep is only trustworthy while these agree, because every threshold
    other than the streamed one comes from the replay alone.
    """
    replay = segment(run.times, run.probs, activation=activation, **seg_kw)
    real = [t for k, t in run.events if k == "start_of_speech"]
    return len(replay), len(real)


def paired_timing(
    runs_a: list,
    runs_b: list,
    items: list,
    *,
    act_a: float,
    act_b: float,
    min_speech: float = 0.05,
    min_silence: float = 0.55,
) -> dict[str, float]:
    """
    Per-item onset/release difference between two detectors on identical audio.

    Comparing medians of absolute onset does not work on production turns. Those
    clips are cut by the upstream stack, not trimmed to the first phoneme, so a
    clip may open with half a second of line noise; both detectors then wait for
    the same lead-in and report the same absolute onset, which says nothing
    about either. Measured on 250 real turns both came out at exactly 644 ms —
    a property of the corpus, not of the models.

    Differencing **within an item** cancels the lead-in, because both detectors
    saw the same samples. `onset_delta_ms > 0` means `b` (PulseVAD) committed
    later than `a` (silero); that gap is what a barge-in feels.
    """
    onset_d: list[float] = []
    release_d: list[float] = []
    seg_kw = {"min_speech": min_speech, "min_silence": min_silence}

    for ra, rb, item in zip(runs_a, runs_b, items):
        body = item.body
        if body is None or len(ra.times) == 0 or len(rb.times) == 0:
            continue
        ia = segment(ra.times, ra.probs, activation=act_a, **seg_kw)
        ib = segment(rb.times, rb.probs, activation=act_b, **seg_kw)
        sa = [s for s, _ in ia if body.start - 0.05 <= s <= body.end]
        sb = [s for s, _ in ib if body.start - 0.05 <= s <= body.end]
        if sa and sb:
            onset_d.append((min(sb) - min(sa)) * 1000.0)
        ea = [e for _, e in ia if e >= body.start]
        eb = [e for _, e in ib if e >= body.start]
        if ea and eb:
            release_d.append((max(eb) - max(ea)) * 1000.0)

    def stats(v: list[float], prefix: str) -> dict[str, float]:
        if not v:
            return {
                f"{prefix}_median_ms": float("nan"),
                f"{prefix}_p90_ms": float("nan"),
                f"{prefix}_n": 0,
                f"{prefix}_b_later_pct": float("nan"),
            }
        arr = np.asarray(v)
        return {
            f"{prefix}_median_ms": float(np.median(arr)),
            f"{prefix}_p90_ms": float(np.percentile(arr, 90)),
            f"{prefix}_n": len(arr),
            f"{prefix}_b_later_pct": float(np.mean(arr > 0) * 100),
        }

    return {**stats(onset_d, "onset_delta"), **stats(release_d, "release_delta")}
