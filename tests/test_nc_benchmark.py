"""
Tests for the noise-cancellation benchmark harness.

These do not check that any model is *good* — that is what the benchmark itself
reports. They check the properties whose failure would make the benchmark lie
quietly: misalignment, leaked state, a frame contract that shifts with frame
size. Every one of these produces a plausible-looking table rather than an
error, which is exactly why they are worth asserting.

No network: everything here runs on synthesised audio and the two enhancers that
need no downloaded weights.
"""

import numpy as np
import pytest

pytest.importorskip("soxr", reason="benchmark extra not installed")

from stt_api.livekit_plugin.noise_cancellation.benchmark import enhancers  # noqa: E402
from stt_api.livekit_plugin.noise_cancellation.benchmark.audio import (  # noqa: E402
    align,
    estimate_delay,
    iter_frames,
)
from stt_api.livekit_plugin.noise_cancellation.benchmark.corpus import Item  # noqa: E402
from stt_api.livekit_plugin.noise_cancellation.benchmark.harness import (  # noqa: E402
    run_item,
    summarize,
)
from stt_api.livekit_plugin.noise_cancellation.benchmark.metrics import si_sdr  # noqa: E402

RATE = 16000
# Enhancers that need no downloaded weights, so the suite stays offline.
LOCAL = ["passthrough", "wiener"]


def speech_like(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """
    A harmonic stack with an amplitude envelope and pauses.

    Not real speech, but it has what these tests need: harmonic structure for the
    delay estimator to lock onto, and genuine silences so the noise-floor path is
    exercised.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f0 = 120.0
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    env = (np.sin(2 * np.pi * 2.5 * t) > -0.3).astype(np.float64)
    env = np.convolve(env, np.hanning(400) / np.hanning(400).sum(), mode="same")
    x = x * env
    x += 0.01 * rng.standard_normal(n)
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def noisy(clean: np.ndarray, snr_db: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    noise *= np.sqrt(np.mean(clean**2) / (10 ** (snr_db / 10)) / np.mean(noise**2))
    return (clean + noise).astype(np.float32)


def make_item(snr_db: float = 5.0) -> Item:
    clean = speech_like()
    return Item(id="synthetic", noisy=noisy(clean, snr_db), clean=clean, rate=RATE, source="test")


# --- alignment ------------------------------------------------------------


@pytest.mark.parametrize("true_delay", [0, 1, 37, 256, 512, 1103])
def test_delay_estimator_recovers_a_known_shift(true_delay):
    """The whole reference-based half of the benchmark rests on this."""
    x = speech_like()
    shifted = np.concatenate((np.zeros(true_delay, dtype=np.float32), x))
    assert estimate_delay(x, shifted, 4000) == true_delay


def test_delay_estimator_survives_heavy_spectral_change():
    """
    A denoiser changes the magnitude spectrum drastically by design. The estimate
    has to track phase, not spectral shape — this is why GCC-PHAT is in there.
    """
    x = speech_like()
    delay = 400
    shifted = np.concatenate((np.zeros(delay, dtype=np.float32), x))
    # Crude lowpass: a large, speech-like spectral tilt.
    k = np.hanning(64) / np.hanning(64).sum()
    filtered = np.convolve(shifted, k, mode="same").astype(np.float32)
    est = estimate_delay(x, filtered, 4000)
    assert abs(est - delay) <= 32, f"estimated {est}, expected ~{delay}"


def test_delay_estimator_never_returns_negative():
    """A causal filter can delay audio, not advance it. Admitting negative lags
    would let a bad estimate 'improve' a score by shifting the reference."""
    x = speech_like()
    advanced = x[300:]
    assert estimate_delay(x, advanced, 2000) >= 0


def test_align_trims_both_sides_to_one_length():
    a, b = align(np.arange(100.0), np.arange(120.0), 10)
    assert len(a) == len(b) == 100
    assert b[0] == 10


# --- metrics --------------------------------------------------------------


def test_si_sdr_is_scale_invariant():
    """Otherwise a model would be scored on output gain, which is trivially
    fixable downstream and not what anyone means by quality."""
    x = speech_like()
    est = 0.4 * x + 0.01 * np.random.default_rng(2).standard_normal(len(x)).astype(np.float32)
    assert si_sdr(x, est) == pytest.approx(si_sdr(x, 7.5 * est), abs=1e-6)


def test_si_sdr_ranks_less_noise_higher():
    x = speech_like()
    rng = np.random.default_rng(3)
    n = rng.standard_normal(len(x)).astype(np.float32)
    assert si_sdr(x, x + 0.01 * n) > si_sdr(x, x + 0.10 * n)


# --- frame contract -------------------------------------------------------


def test_iter_frames_pads_the_tail_rather_than_dropping_it():
    frames = list(iter_frames(np.ones(250, dtype=np.float32), 100))
    assert [len(f) for f in frames] == [100, 100, 100]
    assert frames[-1][50:].sum() == 0


@pytest.mark.parametrize("name", LOCAL)
@pytest.mark.parametrize("frame_ms", [10, 20, 50])
def test_every_frame_comes_back_the_same_length(name, frame_ms):
    enh = enhancers.build(name)
    enh.reset()
    n = RATE * frame_ms // 1000
    for f in iter_frames(speech_like(1.0), n):
        y = enh.process(f, RATE)
        assert len(y) == n
        assert np.isfinite(y).all()


@pytest.mark.parametrize("name", LOCAL)
def test_delay_does_not_move_with_frame_size(name):
    """
    A delay that tracks frame size means the buffering is wrong, and every
    quality number for that model is being measured at the wrong alignment.
    """
    item = make_item()
    delays = set()
    for frame_ms in (10, 20, 50):
        r = run_item(enhancers.build(name), item, frame_ms=frame_ms)
        assert r.error is None, r.error
        delays.add(r.delay_samples)
    assert len(delays) == 1, f"{name} delay moved with frame size: {sorted(delays)}"


# --- state isolation ------------------------------------------------------


@pytest.mark.parametrize("name", LOCAL)
def test_reset_fully_isolates_consecutive_items(name):
    """
    State carried across utterances would let a model arrive already adapted to
    the noise — a real advantage in a long call, a fabricated one on a benchmark
    of independent clips. A reused instance must match a fresh one exactly.
    """
    item = make_item()
    reused = enhancers.build(name)
    first = run_item(reused, item, frame_ms=20)
    reused_again = run_item(reused, item, frame_ms=20)
    fresh = run_item(enhancers.build(name), item, frame_ms=20)

    assert first.quality.pesq == pytest.approx(fresh.quality.pesq, abs=1e-9)
    assert reused_again.quality.pesq == pytest.approx(fresh.quality.pesq, abs=1e-9)
    assert reused_again.delay_samples == fresh.delay_samples


# --- harness --------------------------------------------------------------


def test_run_item_reports_errors_instead_of_aborting():
    """One bad item must not take down a run that is minutes into a corpus."""

    class Exploding(enhancers.Enhancer):
        name = "exploding"

        def reset(self):
            pass

        def process(self, frame, rate):
            raise ValueError("boom")

    r = run_item(Exploding(), make_item(), frame_ms=20)
    assert r.error is not None and "boom" in r.error
    assert r.quality is None


def test_summarize_drops_failures_and_nans_rather_than_zero_filling():
    item = make_item()
    ok = run_item(enhancers.build("passthrough"), item, frame_ms=20)
    bad = run_item(enhancers.build("passthrough"), item, frame_ms=20)
    bad.error = "ValueError: boom"

    s = summarize([ok, bad], frame_ms=20)
    assert s.items == 1 and s.failed == 1
    # A zero-filled failure would drag this toward 0 instead of matching the one
    # good item.
    assert s.quality["pesq"] == pytest.approx(ok.quality.pesq, abs=1e-9)


def test_passthrough_is_the_identity():
    """Every delta in the report is measured against this row."""
    item = make_item()
    r = run_item(enhancers.build("passthrough"), item, frame_ms=20)
    assert r.delay_samples == 0
    assert r.level_change_db == pytest.approx(0.0, abs=1e-9)


def test_budget_p99_is_relative_to_the_frame_duration():
    """`budget` is the number that says whether a filter keeps up; if it is not
    normalised by frame size it means nothing."""
    item = make_item()
    results = [run_item(enhancers.build("wiener"), item, frame_ms=20)]
    s = summarize(results, frame_ms=20)
    assert s.frame_ms["budget_p99"] == pytest.approx(s.frame_ms["p99"] / 20, rel=1e-9)
    assert 0 <= s.frame_ms["budget_p99"] < 1.0


# --- ASR repetition loops -------------------------------------------------


def test_degenerate_guard_catches_repetition_loops_only():
    """
    A single ASR repetition loop can outweigh a whole corpus, because pooled WER
    is unbounded under insertions. This guard is what keeps one Whisper failure
    from being reported as a property of the enhancer — it turned a model with a
    0 % median per-item WER into an apparent 70 % catastrophe.
    """
    from stt_api.livekit_plugin.noise_cancellation.benchmark.asr import is_degenerate

    ref = "six spoons of fresh snow peas and a snack for her brother bob"
    assert is_degenerate(ref, "the group is due to publish its report in the autumn " * 40)
    # A wrong transcript is not a loop: transcribing the *interfering speaker* is a
    # real failure the benchmark must keep counting.
    assert not is_degenerate("henman has been warned", "i never thought it would take this long")
    # Long but lexically varied output is a transcript, not a loop.
    assert not is_degenerate("a b c", " ".join(f"w{i}" for i in range(60)))
    assert not is_degenerate(ref, ref)


def test_accumulator_excludes_loops_and_reports_them():
    from stt_api.livekit_plugin.noise_cancellation.benchmark.asr import WerAccumulator

    acc = WerAccumulator()
    acc.add("good", "hello world", "hello world")
    acc.add("loop", "hello world", "spam spam spam spam " * 20)
    assert acc.n_degenerate == 1
    assert acc.wer == 0.0, "the loop must not pollute the pooled rate"
    assert acc.median_wer == 0.0
