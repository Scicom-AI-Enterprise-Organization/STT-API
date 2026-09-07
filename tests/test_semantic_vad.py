"""
Tests for the semantic VAD plugin.

Two things are worth guarding here, and both are the silent kind of wrong:

* **The smart-turn preprocessing contract.** Its ONNX graph already applies a
  sigmoid, and short audio must be *left*-padded so the decision point stays at
  the end of the window. Getting either wrong produces confident, plausible
  numbers rather than an error — the double-sigmoid bug pinned every score to
  ~0.73 and made the model look like it was ignoring its input.
* **The LiveKit transport contract.** `_SemanticTransport` implements a private
  Protocol; if a LiveKit upgrade changes it, that should fail here rather than in
  a live call.

Model-dependent tests skip when the weights are not cached, so the suite stays
usable offline.
"""

import os

import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="benchmark/semantic-vad extra not installed")

from stt_api.livekit_plugin.semantic_vad import backends  # noqa: E402

RATE = 16000


def speech_like(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f0 = 110.0 + 20.0 * seed  # vary the pitch so inputs genuinely differ
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 10))
    env = np.convolve(
        (np.sin(2 * np.pi * 2.0 * t) > -0.2).astype(float),
        np.hanning(400) / np.hanning(400).sum(),
        mode="same",
    )
    return (0.3 * (x * env) / max(np.abs(x * env).max(), 1e-9)).astype(np.float32)


# --- preprocessing contract (no weights needed) ---------------------------


def test_window_is_left_padded_so_the_decision_point_stays_at_the_end():
    """
    The model is trained with the pause at the *end* of its 8 s window. Right-
    padding instead would put speech at the start followed by seconds of silence
    — a distribution it never saw, which it scores with confident nonsense
    rather than an error.
    """
    x = np.ones(RATE, dtype=np.float32)
    out = backends.SmartTurnV3._fit_window(x, 8 * RATE)
    assert len(out) == 8 * RATE
    assert out[-RATE:].sum() == pytest.approx(RATE)  # signal at the end
    assert out[: 7 * RATE].sum() == 0.0  # zeros at the front


def test_window_keeps_the_most_recent_audio_when_too_long():
    x = np.arange(10 * RATE, dtype=np.float32)
    out = backends.SmartTurnV3._fit_window(x, 8 * RATE)
    assert len(out) == 8 * RATE
    assert out[-1] == x[-1]


def test_window_passes_exact_length_through():
    x = np.ones(8 * RATE, dtype=np.float32)
    assert backends.SmartTurnV3._fit_window(x, 8 * RATE) is x


# --- model behaviour ------------------------------------------------------


@pytest.fixture(scope="module")
def smart_turn():
    try:
        return backends.SmartTurnV3()
    except Exception as e:  # noqa: BLE001 - no network / no cache
        pytest.skip(f"smart-turn weights unavailable: {e}")


def test_probability_is_a_probability_not_a_logit(smart_turn):
    """
    The exported graph already applies the sigmoid. Applying a second one is
    silent: it squashes every score toward 0.73 and destroys discrimination
    while still returning something in [0, 1].
    """
    ps = [smart_turn.predict(speech_like(s, seed=i)) for i, s in enumerate([1.0, 3.0, 6.0])]
    for p in ps:
        assert 0.0 <= p <= 1.0
    # A double sigmoid cannot produce values outside sigmoid's own range on
    # realistic logits, so the tell is that everything collapses together.
    assert max(ps) - min(ps) > 1e-6, "scores identical across inputs — features not reaching the model"


def test_empty_audio_is_a_hold(smart_turn):
    assert smart_turn.predict(np.zeros(0, dtype=np.float32)) == 0.0


def test_short_audio_does_not_raise(smart_turn):
    """A pause can be asked about long before 8 s of audio exists."""
    assert 0.0 <= smart_turn.predict(np.zeros(1600, dtype=np.float32)) <= 1.0


# --- LiveKit transport contract -------------------------------------------


def test_transport_satisfies_livekits_protocol():
    pytest.importorskip("livekit.agents", reason="livekit-agents not installed")
    from livekit.agents.inference.eot.base import _StreamingTurnDetectionTransport

    from stt_api.livekit_plugin.semantic_vad.detector import _SemanticTransport

    class Stub:
        window_seconds = 1.0

        def predict(self, pcm):
            return 0.5

    t = _SemanticTransport(backend=Stub(), sample_rate=RATE)
    assert isinstance(t, _StreamingTurnDetectionTransport)


def test_transport_resolves_and_a_failing_backend_holds():
    """
    A backend that raises must resolve as 0.0 (hold), so the agent waits for the
    endpointing timeout rather than interrupting the caller. This is deliberately
    the opposite of the text plugin, which returns 1.0 on a parse failure.
    """
    pytest.importorskip("livekit.agents", reason="livekit-agents not installed")
    import asyncio

    from livekit import rtc

    from stt_api.livekit_plugin.semantic_vad.detector import _SemanticTransport

    class Exploding:
        window_seconds = 1.0

        def predict(self, pcm):
            raise RuntimeError("boom")

    class FakeStream:
        got = None

        def _resolve_prediction(self, rid, prob, inference_duration):
            self.got = (rid, prob)

    async def run():
        t = _SemanticTransport(backend=Exploding(), sample_rate=RATE)
        stream = FakeStream()
        t.attach(stream)
        pcm = np.zeros(320, dtype=np.int16)
        t.push_frame(rtc.AudioFrame(pcm.tobytes(), RATE, 1, 320))
        t.run_inference("req-x")
        for _ in range(100):
            await asyncio.sleep(0.01)
            if stream.got:
                break
        t.detach()
        return stream.got

    got = asyncio.run(run())
    assert got == ("req-x", 0.0)


def test_flush_clears_the_buffer_between_turns():
    """Otherwise the previous speaker's audio bleeds into the next decision."""
    pytest.importorskip("livekit.agents", reason="livekit-agents not installed")
    from livekit import rtc

    from stt_api.livekit_plugin.semantic_vad.detector import _SemanticTransport

    class Stub:
        window_seconds = 1.0

        def predict(self, pcm):
            return 0.5

    t = _SemanticTransport(backend=Stub(), sample_rate=RATE)
    pcm = np.zeros(320, dtype=np.int16)
    t.push_frame(rtc.AudioFrame(pcm.tobytes(), RATE, 1, 320))
    assert len(t._buf) > 0
    t.flush()
    assert len(t._buf) == 0


# --- real AgentSession -----------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_LIVEKIT_INTEGRATION") != "1",
    reason="set RUN_LIVEKIT_INTEGRATION=1 (downloads silero weights, ~25s wall clock)",
)
def test_real_agent_session_lets_the_probability_move_the_turn_boundary():
    """
    The only test here that proves the plugin actually does something.

    Drives a real `AgentSession` — silero VAD, an STT, `SemanticVAD` as
    `turn_detection` — with audio paced at 1x wall clock, twice: once with a
    probability above the threshold and once below. A working integration commits
    the turn on `min_delay` in the first case and waits for `max_delay` in the
    second.

    Asserting the *effect* rather than the call matters. An earlier version of
    this test checked only that `run_inference` was reached and the turn
    committed, and it passed while the verdict was being ignored — and it also
    watched `user_input_transcribed`, which fires when the STT returns, not when
    the turn commits. Both readings looked like success.
    """
    import asyncio
    import time
    import wave

    from livekit import rtc
    from livekit.agents import Agent, AgentSession
    from livekit.agents.voice.io import AudioInput
    from livekit.plugins import silero

    from stt_api.livekit_plugin.dummy.stt import STT as DummySTT
    from stt_api.livekit_plugin.semantic_vad import SemanticVAD

    frame_ms = 50
    n = RATE * frame_ms // 1000
    MIN_DELAY, MAX_DELAY = 0.5, 3.0

    class Paced(AudioInput):
        def __init__(self, audio):
            super().__init__(label="paced")
            clip = np.concatenate(
                [np.zeros(int(0.3 * RATE), np.float32), audio, np.zeros(int(9 * RATE), np.float32)]
            )
            self.frames = [clip[i : i + n] for i in range(0, len(clip) - n + 1, n)]
            self.i, self.t0 = 0, None

        def stream_time(self):
            return 0.0 if self.t0 is None else time.monotonic() - self.t0

        async def __anext__(self):
            if self.t0 is None:
                self.t0 = time.monotonic()
            delay = self.i * (frame_ms / 1000) - (time.monotonic() - self.t0)
            if delay > 0:
                await asyncio.sleep(delay)
            chunk = self.frames[self.i] if self.i < len(self.frames) else np.zeros(n, np.float32)
            self.i += 1
            pcm = np.clip(np.rint(chunk * 32768), -32768, 32767).astype(np.int16)
            return rtc.AudioFrame(pcm.tobytes(), RATE, 1, n)

    class Const:
        """A fixed probability isolates the routing from any model's opinion."""

        window_seconds = 8.0

        def __init__(self, p):
            self.p = p

        def predict(self, pcm):
            return self.p

    wav = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "stt_api", "livekit_plugin", "dummy", "audio", "tawaran.wav",
    )
    if not os.path.exists(wav):
        pytest.skip("speech fixture missing")
    w = wave.open(wav)
    raw = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    if w.getframerate() != RATE:
        import soxr

        raw = soxr.resample(raw, w.getframerate(), RATE).astype(np.float32)
    speech_end = 0.3 + len(raw) / RATE

    async def commit_delay(prob, threshold):
        audio_in = Paced(raw)
        commits = []

        class Rec(Agent):
            async def on_user_turn_completed(self, turn_ctx, new_message):
                commits.append(audio_in.stream_time())

        session = AgentSession(
            vad=silero.VAD.load(),
            stt=DummySTT(),
            llm=None,
            tts=None,
            turn_handling={
                "turn_detection": SemanticVAD(
                    backend=Const(prob), unlikely_threshold=threshold
                ),
                "endpointing": {"min_delay": MIN_DELAY, "max_delay": MAX_DELAY},
                "interruption": {"mode": "vad"},
            },
            aec_warmup_duration=None,
            user_away_timeout=None,
        )
        session.input.audio = audio_in
        await session.start(Rec(instructions="test"))
        await asyncio.sleep(12.0)
        await session.aclose()
        assert commits, "the session never committed the user's turn"
        return commits[0] - speech_end

    fast = asyncio.run(commit_delay(0.9, 0.2))  # above threshold -> finished
    slow = asyncio.run(commit_delay(0.05, 0.5))  # below threshold -> keep listening

    assert fast == pytest.approx(MIN_DELAY, abs=0.35), f"expected the fast path, got {fast:.2f}s"
    assert slow == pytest.approx(MAX_DELAY, abs=0.5), f"expected the slow path, got {slow:.2f}s"
    assert slow - fast > 1.5, (
        f"the probability did not move the turn boundary ({fast:.2f}s vs {slow:.2f}s) — "
        "LiveKit is consulting the detector but ignoring its verdict"
    )


# --- ScicomEoT (Malaysian telephony family) -------------------------------


@pytest.fixture(scope="module")
def scicom_tiny():
    try:
        return backends.ScicomEoT("tiny")
    except Exception as e:  # noqa: BLE001 - no network / no cache
        pytest.skip(f"scicom weights unavailable: {e}")


def test_scicom_reads_its_own_window_config(scicom_tiny):
    """
    Window and normalisation come from the checkpoint's `eot_window.json`, not
    from a default.

    `do_normalize` differs between the two families — smart-turn wants it on, the
    Scicom models ship `false` — and the wrong value produces a plausible score
    rather than an error (measured on one tone: 0.39 vs 0.53). Trusting the
    checkpoint is what keeps that from being a silent 50/50 guess.
    """
    assert scicom_tiny.window_seconds == 8.0
    assert scicom_tiny.do_normalize is False


def test_scicom_rejects_an_unknown_size():
    with pytest.raises(ValueError, match="size must be one of"):
        backends.ScicomEoT("enormous")


def test_scicom_discriminates_complete_from_truncated(scicom_tiny):
    """
    Direction only. This is English synthetic audio, well outside the Malaysian
    telephony this model was trained on, so the *margin* means nothing — but a
    complete utterance still must not score below a truncated one, and if the
    preprocessing were wrong it would.
    """
    full = speech_like(3.0, seed=1)
    p_full = scicom_tiny.predict(full)
    p_cut = scicom_tiny.predict(full[: int(len(full) * 0.5)])
    assert 0.0 <= p_cut <= 1.0
    assert 0.0 <= p_full <= 1.0
    assert p_full >= p_cut


def test_scicom_satisfies_the_backend_protocol(scicom_tiny):
    assert isinstance(scicom_tiny, backends.Backend)
    assert scicom_tiny.predict(np.zeros(0, dtype=np.float32)) == 0.0
