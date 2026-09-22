"""
Tests for the PulseVAD plugin.

Four things are worth guarding, and every one of them fails *silently* in
production rather than raising:

* **The front end.** `frontend.log_mel` is a vectorised rewrite of upstream's
  `pulsevad.frontend_np`, and the claim in its docstring is bit-exactness, not
  approximation. A transcription of the upstream loop lives here and is asserted
  against — a drift of 1e-6 in a z-normalised feature would never show up as an
  error, only as slightly different probabilities.
* **The sigmoid.** The graph emits `[non_speech, speech]` logits, not a
  probability. Forgetting the sigmoid is the mirror image of the double-sigmoid
  bug in `../semantic_vad/`, which pinned every score to ~0.73 and looked like a
  model ignoring its input.
* **The ceiling.** p(speech) saturates at 0.711, so a silero-style
  `activation_threshold=0.8` produces a VAD that can never fire. `load()` must
  refuse it; if this test ever goes green by accident, the failure mode is an
  agent that never hears anyone and logs nothing.
* **The benchmark's offline replay.** `metrics.segment` reproduces the streaming
  state machine so a threshold sweep does not need re-inference. If it drifts
  from `vad.py`, every swept number is fiction.

The weights are vendored, so nothing here needs the network. LiveKit-dependent
tests skip when `livekit-agents` is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="livekit/benchmark extra not installed")

from stt_api.livekit_plugin.pulse_vad import (  # noqa: E402
    CEILINGS,
    WINDOW_SAMPLES,
    PulseVADModel,
    log_mel,
)

RATE = 16000


# --- the front end, against a transcription of upstream ---------------------


def _upstream_log_mel(waveform: np.ndarray) -> np.ndarray:
    """
    `pulsevad.frontend_np.extract_log_mel_np`, transcribed verbatim.

    Deliberately the slow loop-per-frame original: this is the reference the
    vectorised implementation must match exactly, so it must not share code with
    it.
    """
    from stt_api.livekit_plugin.pulse_vad.frontend import mel_filterbank

    n_fft, win_length, hop, eps, alpha = 512, 400, 160, 1e-5, 0.97
    pad_win = (n_fft - win_length) // 2
    hann = (
        0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_length) / win_length)
    ).astype(np.float32)
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    fb = mel_filterbank()

    x_pre = np.empty_like(x)
    x_pre[:, 0] = x[:, 0]
    x_pre[:, 1:] = x[:, 1:] - alpha * x[:, :-1]
    mean = x_pre.mean(axis=-1, keepdims=True)
    std = x_pre.std(axis=-1, keepdims=True, ddof=0)
    x_norm = (x_pre - mean) / (std + eps)
    x_padded = np.pad(x_norm, ((0, 0), (n_fft // 2, n_fft // 2)), mode="reflect")

    out = []
    for b in range(x.shape[0]):
        frames = []
        for i in range(21):
            start = i * hop
            chunk = np.zeros(n_fft, dtype=np.float32)
            chunk[pad_win : pad_win + win_length] = (
                x_padded[b, start + pad_win : start + pad_win + win_length] * hann
            )
            frames.append(np.abs(np.fft.rfft(chunk, n=n_fft)) ** 2)
        mel = (np.stack(frames, axis=0) @ fb).T
        log_m = np.log(mel + eps)
        m = log_m.mean(axis=-1, keepdims=True)
        s = log_m.std(axis=-1, keepdims=True, ddof=0)
        out.append((log_m - m) / (s + eps))
    return np.stack(out, axis=0).astype(np.float32)


@pytest.mark.parametrize("name", ["noise", "tone", "silence", "tiny", "clipped"])
def test_frontend_is_bit_exact_with_upstream(name):
    rng = np.random.default_rng(0)
    x = {
        "noise": (rng.standard_normal((4, WINDOW_SAMPLES)) * 0.1).astype(np.float32),
        "tone": np.sin(
            2 * np.pi * 440 * np.arange(WINDOW_SAMPLES) / RATE, dtype=np.float64
        ).astype(np.float32)[None],
        "silence": np.zeros((2, WINDOW_SAMPLES), np.float32),
        "tiny": (rng.standard_normal((2, WINDOW_SAMPLES)) * 1e-6).astype(np.float32),
        "clipped": np.clip(rng.standard_normal((3, WINDOW_SAMPLES)) * 4, -1, 1).astype(
            np.float32
        ),
    }[name]
    assert np.array_equal(log_mel(x), _upstream_log_mel(x))


def test_frontend_shape_and_validation():
    assert log_mel(np.zeros(WINDOW_SAMPLES, np.float32)).shape == (1, 64, 21)
    assert log_mel(np.zeros((5, WINDOW_SAMPLES), np.float32)).shape == (5, 64, 21)
    with pytest.raises(ValueError, match="3200"):
        log_mel(np.zeros(1000, np.float32))


# --- the model contract -----------------------------------------------------


def _speech_like(seconds: float = 0.2, seed: int = 0) -> np.ndarray:
    """A voiced-ish buzz: harmonic stack under a syllable-rate envelope."""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f0 = 120.0 + 15.0 * seed
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    return (0.3 * x * env / np.abs(x * env).max()).astype(np.float32)


def test_output_is_a_probability_not_a_raw_logit():
    """
    The graph's two outputs are logits; p = sigmoid(speech - non_speech).

    Skipping the sigmoid would leave values outside [0, 1] — the check that
    would have caught the semantic_vad double-sigmoid bug in reverse.
    """
    model = PulseVADModel()
    windows = np.stack([_speech_like(seed=i) for i in range(4)])
    p = model.probabilities(windows)
    assert p.shape == (4,)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_silence_scores_low_and_speech_scores_higher():
    model = PulseVADModel()
    silence = model(np.zeros(WINDOW_SAMPLES, np.float32))
    speech = model(_speech_like())
    assert silence < 0.1, f"digital silence scored {silence}"
    assert speech > silence


@pytest.mark.parametrize("variant,precision", sorted(CEILINGS))
def test_measured_ceiling_is_not_exceeded(variant, precision):
    """
    `CEILINGS` gates `activation_threshold`, so it must stay true.

    If a checkpoint is ever swapped for one with a wider output range this fails
    here rather than silently narrowing the usable threshold range.
    """
    model = PulseVADModel(variant, precision=precision)
    rng = np.random.default_rng(1)
    windows = np.concatenate(
        [
            np.stack([_speech_like(seed=i) for i in range(12)]),
            (rng.standard_normal((12, WINDOW_SAMPLES)) * 0.2).astype(np.float32),
            np.zeros((2, WINDOW_SAMPLES), np.float32),
        ]
    )
    assert model.probabilities(windows).max() <= CEILINGS[(variant, precision)] + 1e-6
    assert model.ceiling == CEILINGS[(variant, precision)]


def test_int8_and_fp32_agree_closely():
    """int8 is a quantisation of the same net, not a different model."""
    fp32, int8 = PulseVADModel(precision="fp32"), PulseVADModel(precision="int8")
    windows = np.stack([_speech_like(seed=i) for i in range(8)])
    delta = np.abs(fp32.probabilities(windows) - int8.probabilities(windows))
    assert delta.mean() < 0.05, f"mean |fp32 - int8| = {delta.mean():.4f}"


def test_unpublished_variant_is_refused_clearly():
    with pytest.raises(ValueError, match="no int8 build"):
        PulseVADModel("81k", precision="int8")


def test_wrong_window_length_is_refused():
    with pytest.raises(ValueError, match="3200"):
        PulseVADModel()(np.zeros(512, np.float32))


# --- the LiveKit integration ------------------------------------------------

livekit = pytest.importorskip("livekit.agents", reason="livekit extra not installed")


def _pulse_vad(**kw):
    from stt_api.livekit_plugin.pulse_vad import PulseVAD

    return PulseVAD.load(**kw)


def test_threshold_above_the_ceiling_is_refused():
    """
    The headline trap: silero saturates at 1.0 and PulseVAD at 0.711, so
    `activation_threshold=0.8` carried over from silero yields a VAD that can
    never report speech — and nothing errors at runtime.
    """
    with pytest.raises(ValueError, match="ceiling"):
        _pulse_vad(activation_threshold=0.8)
    with pytest.raises(ValueError, match="ceiling"):
        _pulse_vad(activation_threshold=0.711)
    _pulse_vad(activation_threshold=0.70)  # just under: fine


def test_teacher_allows_a_threshold_the_small_model_cannot():
    _pulse_vad(model="81k", activation_threshold=0.80)
    with pytest.raises(ValueError, match="ceiling"):
        _pulse_vad(model="81k", activation_threshold=0.90)


def test_capabilities_report_the_hop_not_the_window():
    """
    A 200 ms update interval would break LiveKit's duration defaults — a single
    window would satisfy `min_speech_duration=0.05`. The window slides by the
    hop, and the hop is what is advertised.
    """
    vad = _pulse_vad(hop_duration=0.032)
    assert vad.capabilities.update_interval == pytest.approx(0.032)
    assert _pulse_vad(hop_duration=0.064).capabilities.update_interval == pytest.approx(
        0.064
    )


def test_hop_longer_than_the_window_is_refused():
    with pytest.raises(ValueError, match="window"):
        _pulse_vad(hop_duration=0.5)


@pytest.mark.asyncio
async def test_stream_reports_speech_where_speech_is_and_not_where_it_is_not():
    """
    Assert the effect, not the call: drive the real `VADStream` and check the
    events an AgentSession would receive, not that inference happened.
    """
    from livekit import agents, rtc

    from stt_api.livekit_plugin.pulse_vad.benchmark.harness import run_vad

    pad = np.zeros(int(1.5 * RATE), np.float32)
    speech = np.concatenate([_speech_like(2.0, seed=s) for s in range(2)])
    audio = np.concatenate([pad, speech, pad])

    run = await run_vad(_pulse_vad(), audio, label="test")
    kinds = [k for k, _ in run.events]
    assert agents.vad.VADEventType.START_OF_SPEECH.value in kinds
    assert agents.vad.VADEventType.END_OF_SPEECH.value in kinds

    start = next(t for k, t in run.events if k == "start_of_speech")
    end = next(t for k, t in run.events if k == "end_of_speech")
    body_start, body_end = 1.5, 1.5 + len(speech) / RATE
    assert body_start <= start <= body_end, f"speech reported at {start}s"
    assert end > body_end, "turn closed before the speech ended"
    assert start < end
    assert rtc is not None


@pytest.mark.asyncio
async def test_offline_replay_matches_the_live_stream():
    """
    The benchmark sweeps thresholds by replaying `segment` over recorded
    probabilities instead of re-running inference. That is only valid while the
    replay reproduces the stream it is standing in for.
    """
    from stt_api.livekit_plugin.pulse_vad.benchmark.harness import run_vad
    from stt_api.livekit_plugin.pulse_vad.benchmark.metrics import segment

    audio = np.concatenate(
        [
            np.zeros(int(1.0 * RATE), np.float32),
            _speech_like(1.5),
            np.zeros(int(1.2 * RATE), np.float32),
            _speech_like(1.0, seed=3),
            np.zeros(int(1.0 * RATE), np.float32),
        ]
    )
    vad = _pulse_vad(activation_threshold=0.35)
    run = await run_vad(vad, audio, label="replay")

    replayed = segment(
        run.times, run.probs, activation=0.35, min_speech=0.05, min_silence=0.55
    )
    live = [t for k, t in run.events if k == "start_of_speech"]
    assert len(replayed) == len(live), f"replay {len(replayed)} turns, live {len(live)}"
    for (start, _), live_start in zip(replayed, live):
        assert start == pytest.approx(live_start, abs=0.05)


@pytest.mark.asyncio
async def test_flush_clears_the_context_window():
    """
    `VADStream.flush()` is a hard segment boundary, and the 200 ms context window
    has to reset with everything else.

    Without the reset the window still holds 168 ms of the *previous* speaker
    when the next turn's first hop arrives, so the first probability after a
    flush reports the old audio. The check is sharp: feed speech, flush, feed
    silence, and the very next inference must read as silence.
    """
    from livekit import rtc

    vad = _pulse_vad()
    stream = vad.stream()

    def push(pcm: np.ndarray) -> None:
        for i in range(0, len(pcm) - 320 + 1, 320):
            stream.push_frame(
                rtc.AudioFrame(
                    data=pcm[i : i + 320].tobytes(),
                    sample_rate=RATE,
                    num_channels=1,
                    samples_per_channel=320,
                )
            )

    push((_speech_like(1.0) * 32767).astype(np.int16))
    stream.flush()
    push(np.zeros(RATE, np.int16))
    stream.end_input()

    before, after = [], []
    flushed = False
    last_ts = -1.0
    async for ev in stream:
        if ev.type.value != "inference_done":
            continue
        # _reset_state() restarts the published clock, so a timestamp going
        # backwards is the flush boundary.
        if ev.timestamp < last_ts:
            flushed = True
        last_ts = ev.timestamp
        (after if flushed else before).append(ev.probability)
    await stream.aclose()

    assert flushed, "the flush boundary was never observed"
    assert before and after
    assert max(before) > 0.3, f"the speech half never registered (max {max(before):.3f})"
    assert after[0] < 0.1, (
        f"first probability after flush was {after[0]:.3f}; the previous turn's "
        "audio is still in the context window"
    )
