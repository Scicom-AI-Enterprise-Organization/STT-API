"""
Tests for the Clearline plugins (`stt_api/livekit_plugin/clearline/`).

Each of these fails *silently* in production rather than raising, which is why
it is guarded here:

* **The operating point.** The default threshold (0.825 for the release at
  512-sample windows, 0.85 at 256, 0.75 for the distilled variant) and filter (0.35,
  0.5 for the distilled variant) belong to the pinned weights. Silero's 0.5 carried across would fire more and earlier,
  with nothing in the logs.
* **The filter's direction.** `smoothing_alpha` is LiveKit's ExpFilter alpha, the
  weight of the *previous* value: 1.0 freezes the filter (a VAD stuck on its first
  probability) and must be refused.
* **The stream.** Assert the effect, not the call: drive the real `VADStream` over
  real speech and check the events an `AgentSession` would get, at 16 and 48 kHz
  and with both windows.
* **The filter's frame contract.** A frame back for every frame in, at the same
  length, rate and channel count, in both modes; and it must actually denoise.

The weights come from the pinned, private Hugging Face revisions (`hub.py`).
Without an `HF_TOKEN` that can read them, or `CLEARLINE_VAD_ONNX_PATH` /
`CLEARLINE_NC_ONNX_PATH`, everything that needs them skips.
"""

from __future__ import annotations

import asyncio
import os
import wave

import numpy as np
import pytest

rtc = pytest.importorskip("livekit.rtc", reason="livekit not installed")
pytest.importorskip("onnxruntime", reason="livekit extra not installed")

from stt_api.livekit_plugin.clearline import hub  # noqa: E402

RATE = 16000
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
    x = (
        np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        / 32768.0
    )
    if w.getframerate() != rate:
        n = int(len(x) * rate / w.getframerate())
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(
            np.float32
        )
    return x


def _weights_or_skip(fetch):
    try:
        return fetch()
    except FileNotFoundError as e:
        pytest.skip(f"Clearline weights unavailable: {str(e)[:160]}")


@pytest.fixture(scope="module")
def vad_weights():
    return _weights_or_skip(
        lambda: {w: hub.vad_model_path("release", w) for w in (512, 256)}
    )


@pytest.fixture(scope="module")
def nc_weights():
    return _weights_or_skip(
        lambda: {m: hub.nc_model_path(m) for m in ("asr", "quality")}
    )


# --- the VAD ------------------------------------------------------------------


def _vad(**kw):
    from stt_api.livekit_plugin.clearline import ClearlineVAD

    return ClearlineVAD.load(**kw)


def _events(
    vad, x: np.ndarray, rate: int, frame_ms: int = 20
) -> list[tuple[str, float]]:
    """Push `x` through a real stream in `frame_ms` frames; (event, time) for every start and end."""
    from livekit import agents

    async def go():
        stream = vad.stream()
        n = rate * frame_ms // 1000
        pcm = np.clip(np.rint(x * 32767.0), -32768, 32767).astype(np.int16)
        for k in range(0, len(pcm) - n + 1, n):
            stream.push_frame(rtc.AudioFrame(pcm[k : k + n].tobytes(), rate, 1, n))
        stream.end_input()
        out = []
        async for ev in stream:
            if ev.type in (
                agents.vad.VADEventType.START_OF_SPEECH,
                agents.vad.VADEventType.END_OF_SPEECH,
            ):
                out.append((ev.type.value, ev.timestamp))
        await stream.aclose()
        return out

    return asyncio.run(go())


def test_unknown_variant_or_window_is_refused():
    with pytest.raises(ValueError, match="variant"):
        hub.vad_model_path("teacher", 512)
    with pytest.raises(ValueError, match="window"):
        hub.vad_model_path("release", 1024)
    with pytest.raises(ValueError, match="mode"):
        hub.nc_model_path("loud")


def test_frozen_filter_is_refused():
    # Checked before any download: LiveKit's ExpFilter with alpha=1 never moves.
    with pytest.raises(ValueError, match="freeze"):
        _vad(smoothing_alpha=1.0)


def test_missing_local_model_raises_instead_of_falling_back():
    with pytest.raises(FileNotFoundError):
        _vad(onnx_file_path="/nonexistent/clearline_vad.onnx")


def test_defaults_are_the_pinned_operating_point(vad_weights):
    vad = _vad()
    point = hub.OPERATING_POINT[("release", 512)]
    assert vad._opts.activation_threshold == point["activation_threshold"] == 0.825
    assert vad._opts.smoothing_alpha == point["smoothing_alpha"] == 0.35
    assert vad._opts.deactivation_threshold == pytest.approx(0.675)
    assert vad.model == "clearline-vad-release"


@pytest.mark.parametrize("variant", ["release", "distilled"])
@pytest.mark.parametrize("window", [512, 256])
def test_each_graph_gets_its_own_operating_point(vad_weights, variant, window):
    # The points differ by graph (0.825 for the release at 512 samples, 0.85 at
    # 256, 0.75 for the distilled variant): a single default would silently mis-set
    # three of the four.
    vad = _vad(variant=variant, window=window)
    point = hub.OPERATING_POINT[(variant, window)]
    assert vad._opts.activation_threshold == point["activation_threshold"]
    assert vad._opts.smoothing_alpha == point["smoothing_alpha"]


def test_every_graph_has_an_operating_point():
    assert set(hub.OPERATING_POINT) == set(hub.VAD_FILES)


@pytest.mark.parametrize("window, interval", [(512, 0.032), (256, 0.016)])
def test_capabilities_report_the_window(vad_weights, window, interval):
    assert _vad(window=window).capabilities.update_interval == pytest.approx(interval)


@pytest.mark.parametrize("window", [512, 256])
@pytest.mark.parametrize("rate", [16000, 48000])
def test_stream_reports_speech_where_speech_is(vad_weights, window, rate):
    speech = load_speech(rate)
    pad = np.zeros(int(1.5 * rate), np.float32)
    events = _events(_vad(window=window), np.concatenate([pad, speech, pad]), rate)
    starts = [t for k, t in events if k == "start_of_speech"]
    ends = [t for k, t in events if k == "end_of_speech"]
    body_start, body_end = 1.5, 1.5 + len(speech) / rate
    assert starts, "no START over real speech"
    assert (
        body_start - 0.1 <= starts[0] <= body_end
    ), f"speech reported at {starts[0]:.2f}s"
    assert (
        ends and ends[-1] > body_end - 0.5
    ), "the last turn never closed after the speech"
    assert len(ends) == len(starts), "a turn left open at the end of the stream"


def test_input_rate_does_not_move_the_turns(vad_weights):
    # Any input rate is resampled to 16 kHz for inference: 48 kHz must segment the same.
    # The comparison starts at the speech: the fixture opens with a 0.1 s noise burst
    # (-53 dBFS) after digital silence, which the release scores right at its
    # threshold, and LiveKit's resampler output varies slightly from run to run, so
    # that one borderline call is a coin toss at 48 kHz rather than a rate effect.
    lead = lambda x, r: x[int(0.5 * r) :]  # noqa: E731
    speech16, speech48 = (
        lead(load_speech(16000), 16000),
        lead(load_speech(48000), 48000),
    )
    pad = lambda r: np.zeros(int(1.5 * r), np.float32)  # noqa: E731
    a = _events(_vad(), np.concatenate([pad(16000), speech16, pad(16000)]), 16000)
    b = _events(_vad(), np.concatenate([pad(48000), speech48, pad(48000)]), 48000)
    assert [k for k, _ in a] == [k for k, _ in b]
    assert max(abs(ta - tb) for (_, ta), (_, tb) in zip(a, b)) <= 0.1


def test_silence_and_a_quiet_line_do_not_fire(vad_weights):
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [
            np.zeros(3 * RATE, np.float32),
            (rng.standard_normal(3 * RATE) * 10 ** (-50 / 20)).astype(np.float32),
        ]
    )
    assert not _events(_vad(), x, RATE)


def test_streams_do_not_share_state(vad_weights):
    # One session, a model state per stream: two streams over the same audio agree exactly.
    vad = _vad()
    x = np.concatenate(
        [np.zeros(RATE, np.float32), load_speech(RATE), np.zeros(RATE, np.float32)]
    )
    assert _events(vad, x, RATE) == _events(vad, x, RATE)


# --- the noise filter -----------------------------------------------------------


def _nc(**kw):
    from stt_api.livekit_plugin.clearline import ClearlineNC

    return ClearlineNC(**kw)


def run(nc, x: np.ndarray, rate: int, frame_ms: int = 20) -> np.ndarray:
    n = rate * frame_ms // 1000
    out = []
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    for k in range(0, len(pcm) - n + 1, n):
        f = rtc.AudioFrame(pcm[k : k + n].tobytes(), rate, 1, n)
        g = nc._process(f)
        assert g.samples_per_channel == n
        assert g.sample_rate == rate
        assert g.num_channels == 1
        out.append(np.frombuffer(g.data, dtype=np.int16, count=g.samples_per_channel))
    return np.concatenate(out).astype(np.float32) / 32768.0


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    n = min(len(a), len(b), 40000)
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    return max((float(np.dot(a[: n - k], b[k:n])), k) for k in range(max_lag))[1]


def snr_db(clean: np.ndarray, test: np.ndarray, delay: int = 0) -> float:
    t = test[delay:]
    n = min(len(clean), len(t))
    c, t = clean[:n], t[:n]
    scale = float(np.dot(c, t) / max(np.dot(c, c), 1e-12))
    err = t - scale * c
    return 10 * np.log10(
        max(float(np.dot(t, t)), 1e-12) / max(float(np.dot(err, err)), 1e-12)
    )


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        _nc(mode="loud")


def test_missing_local_model_raises_instead_of_passing_through():
    with pytest.raises(FileNotFoundError):
        _nc(model_path="/nonexistent/clearline_nc.onnx")


@pytest.mark.parametrize("mode", ["asr", "quality"])
@pytest.mark.parametrize("rate", [16000, 24000, 48000, 8000])
@pytest.mark.parametrize("frame_ms", [10, 50])
def test_frame_contract_holds_at_any_rate(nc_weights, mode, rate, frame_ms):
    y = run(_nc(mode=mode), load_speech(rate), rate, frame_ms)
    assert y.size > 0 and np.isfinite(y).all() and np.abs(y).max() <= 1.0


def test_delay_is_bounded_and_frame_size_independent(nc_weights):
    x = load_speech(RATE)
    delays = {
        best_lag(x, run(_nc(mode="quality"), x, RATE, ms), 1200) for ms in (10, 50)
    }
    assert len(delays) == 1, delays
    assert delays.pop() <= 2 * 256  # 32 ms: one hop of priming plus one of STFT fill


def _noisy(in_snr: float):
    clean = load_speech(RATE)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    noise *= np.sqrt(
        float(np.mean(clean**2)) / (10 ** (in_snr / 10)) / float(np.mean(noise**2))
    )
    return clean, clean + noise


@pytest.mark.parametrize("mode", ["asr", "quality"])
@pytest.mark.parametrize("in_snr", [0, 5, 10])
def test_models_improve_snr_on_noisy_speech(nc_weights, mode, in_snr):
    # each mode's model itself, nothing mixed back
    clean, noisy = _noisy(in_snr)
    y = run(_nc(mode=mode, mix_back=0.0), noisy, RATE)
    before, after = snr_db(clean, noisy), snr_db(clean, y, best_lag(clean, y, 1200))
    assert after > before + 4.0, f"{mode}: {before:.1f} dB -> {after:.1f} dB"


@pytest.mark.parametrize("in_snr", [0, 5])
def test_asr_mode_cleans_up(nc_weights, in_snr):
    # 70 % of the input mixed back: a gentle clean-up (+1.3 dB at 0 dB in on this
    # speech and white noise), never a removal
    clean, noisy = _noisy(in_snr)
    y = run(_nc(mode="asr"), noisy, RATE)
    before, after = snr_db(clean, noisy), snr_db(clean, y, best_lag(clean, y, 1200))
    assert before + 1.0 < after < before + 6.0, f"asr: {before:.1f} dB -> {after:.1f} dB"


def test_asr_mode_keeps_some_of_the_input(nc_weights):
    # mix_back: the ASR mode leaves 70 % of the input under the output, so a pure
    # noise input is attenuated, not erased; the quality mode removes more of it.
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(2 * RATE) * 0.05).astype(np.float32)
    level = {
        m: float(np.sqrt(np.mean(run(_nc(mode=m), x, RATE)[RATE // 2 :] ** 2)))
        for m in ("asr", "quality")
    }
    assert level["quality"] < level["asr"] < float(np.sqrt(np.mean(x**2)))


def test_non_mono_passes_through_untouched(nc_weights):
    pcm = np.zeros(320, dtype=np.int16)
    stereo = rtc.AudioFrame(pcm.tobytes(), RATE, 2, 160)
    assert _nc()._process(stereo) is stereo


def test_disabled_flag_is_honoured(nc_weights):
    # LiveKit skips `_process` while disabled; the flag must be settable mid-session.
    nc = _nc(enabled=False)
    assert nc.enabled is False
    nc.enabled = True
    assert nc.enabled is True


def test_streams_do_not_share_recurrent_state(nc_weights):
    x = load_speech(RATE)[: 2 * RATE]
    assert np.array_equal(run(_nc(), x, RATE), run(_nc(), x, RATE))
