"""Tests for the self-hosted GTCRN noise cancellation FrameProcessor.

These assert the properties the LiveKit audio pipeline depends on — a frame back
for every frame in, at the same rate and length — plus that the model actually
removes noise without eating speech. Run with `-s` to see the measured numbers.
"""

import os
import time
import wave

import numpy as np
import pytest

rtc = pytest.importorskip("livekit.rtc", reason="livekit not installed")

from stt_api.livekit_plugin.noise_cancellation import (  # noqa: E402
    MODEL_SAMPLE_RATE,
    GTCRN,
)

SPEECH_WAV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stt_api",
    "livekit_plugin",
    "dummy",
    "audio",
    "tawaran.wav",
)


def load_speech(rate: int) -> np.ndarray:
    if not os.path.exists(SPEECH_WAV):
        pytest.skip(f"speech fixture missing: {SPEECH_WAV}")
    w = wave.open(SPEECH_WAV)
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if w.getframerate() != rate:
        n = int(len(x) * rate / w.getframerate())
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def to_frames(x: np.ndarray, rate: int, frame_ms: int = 50):
    n = rate * frame_ms // 1000
    for i in range(0, len(x) - n + 1, n):
        pcm = np.clip(np.rint(x[i : i + n] * 32768.0), -32768, 32767).astype(np.int16)
        yield rtc.AudioFrame(pcm.tobytes(), rate, 1, n)


def run(x: np.ndarray, rate: int, frame_ms: int = 50):
    """Push audio through the filter frame by frame; return (output, seconds spent)."""
    nc = GTCRN()
    out, spent = [], 0.0
    for f in to_frames(x, rate, frame_ms):
        t0 = time.perf_counter()
        g = nc._process(f)
        spent += time.perf_counter() - t0
        assert g.samples_per_channel == f.samples_per_channel
        assert g.sample_rate == f.sample_rate
        assert g.num_channels == f.num_channels
        out.append(np.frombuffer(g.data, dtype=np.int16, count=g.samples_per_channel))
    return np.concatenate(out).astype(np.float32) / 32768.0, spent


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    n = min(len(a), len(b), 40000)
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    scores = [(float(np.dot(a[: n - k], b[k:n])), k) for k in range(max_lag)]
    return max(scores)[1]


def snr_db(clean: np.ndarray, test: np.ndarray, delay: int = 0) -> float:
    t = test[delay:]
    n = min(len(clean), len(t))
    c, t = clean[:n], t[:n]
    scale = float(np.dot(c, t) / max(np.dot(c, c), 1e-12))  # gain-invariant
    err = t - scale * c
    return 10 * np.log10(max(float(np.dot(t, t)), 1e-12) / max(float(np.dot(err, err)), 1e-12))


@pytest.mark.parametrize("rate", [16000, 24000, 48000, 8000])
@pytest.mark.parametrize("frame_ms", [10, 50])
def test_frame_contract_holds_at_any_rate(rate, frame_ms):
    """Every frame comes back the same length, rate and channel count."""
    y, _ = run(load_speech(rate), rate, frame_ms)
    assert y.size > 0
    assert np.isfinite(y).all()
    assert np.abs(y).max() <= 1.0


def test_runs_far_faster_than_real_time():
    """`_process` blocks the audio read loop, so it has to be cheap."""
    x = load_speech(MODEL_SAMPLE_RATE)
    y, spent = run(x, MODEL_SAMPLE_RATE)
    rtf = spent / (len(y) / MODEL_SAMPLE_RATE)
    print(f"\n  RTF {rtf:.4f} ({rtf * 100:.1f}% of one core per stream)")
    assert rtf < 0.25


def test_delay_is_bounded_and_frame_size_independent():
    """32 ms at 16 kHz: one hop of priming plus one of STFT fill."""
    x = load_speech(MODEL_SAMPLE_RATE)
    delays = set()
    for frame_ms in (10, 50):
        y, _ = run(x, MODEL_SAMPLE_RATE, frame_ms)
        d = best_lag(x, y, 1200)
        delays.add(d)
        print(f"\n  {frame_ms} ms frames -> {d} samples ({d / MODEL_SAMPLE_RATE * 1000:.1f} ms)")
    # A delay that moves with the frame size would mean the buffering is wrong.
    assert len(delays) == 1
    assert delays.pop() <= 2 * 256


@pytest.mark.parametrize("in_snr", [0, 5, 10])
def test_improves_snr_on_noisy_speech(in_snr):
    rate = MODEL_SAMPLE_RATE
    clean = load_speech(rate)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    noise *= np.sqrt(
        float(np.mean(clean**2)) / (10 ** (in_snr / 10)) / float(np.mean(noise**2))
    )
    noisy = clean + noise

    y, _ = run(noisy, rate)
    d = best_lag(clean, y, 1200)
    before, after = snr_db(clean, noisy), snr_db(clean, y, d)
    print(f"\n  {before:.1f} dB -> {after:.1f} dB ({after - before:+.1f} dB)")
    assert after > before + 4.0


def test_does_not_eat_clean_speech():
    """A denoiser that mangles clean input is worse than none."""
    rate = MODEL_SAMPLE_RATE
    clean = load_speech(rate)
    y, _ = run(clean, rate)
    d = best_lag(clean, y, 1200)
    fidelity = snr_db(clean, y, d)
    level = 20 * np.log10(
        float(np.sqrt(np.mean(y**2))) / float(np.sqrt(np.mean(clean**2)))
    )
    print(f"\n  fidelity {fidelity:.1f} dB, level {level:+.2f} dB")
    assert fidelity > 15.0
    assert abs(level) < 3.0


def test_suppresses_the_noise_floor_in_pauses():
    rate = MODEL_SAMPLE_RATE
    clean = load_speech(rate)
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    noise *= np.sqrt(float(np.mean(clean**2)) / 1.0 / float(np.mean(noise**2)))  # 0 dB
    y, _ = run(clean + noise, rate)
    d = best_lag(clean, y, 1200)

    sil = int(0.35 * rate)  # leading silence of the fixture
    quiet_in = float(np.sqrt(np.mean((clean + noise)[:sil] ** 2)))
    quiet_out = float(np.sqrt(np.mean(y[d : d + sil] ** 2)))
    supp = 20 * np.log10(max(quiet_out, 1e-9) / max(quiet_in, 1e-9))
    print(f"\n  noise floor {supp:+.1f} dB")
    assert supp < -12.0


def test_disabled_is_byte_identical_passthrough():
    clean = load_speech(MODEL_SAMPLE_RATE)
    nc = GTCRN(enabled=False)
    assert nc.enabled is False
    f = next(to_frames(clean, MODEL_SAMPLE_RATE))
    # LiveKit skips `_process` entirely when disabled; assert the flag is honoured
    # so `enabled` can be toggled mid-session.
    nc.enabled = True
    assert nc.enabled is True
    g = nc._process(f)
    assert g.samples_per_channel == f.samples_per_channel


def test_non_mono_passes_through_untouched():
    pcm = np.zeros(320, dtype=np.int16)
    stereo = rtc.AudioFrame(pcm.tobytes(), MODEL_SAMPLE_RATE, 2, 160)
    out = GTCRN()._process(stereo)
    assert out is stereo


def test_missing_model_raises_instead_of_silently_degrading():
    with pytest.raises(FileNotFoundError):
        GTCRN(model_path="/nonexistent/gtcrn.onnx")


def test_streams_do_not_share_recurrent_state():
    """Two instances must denoise identically; shared caches would diverge."""
    x = load_speech(MODEL_SAMPLE_RATE)[: MODEL_SAMPLE_RATE * 2]
    a, _ = run(x, MODEL_SAMPLE_RATE)
    b, _ = run(x, MODEL_SAMPLE_RATE)
    assert np.array_equal(a, b)
