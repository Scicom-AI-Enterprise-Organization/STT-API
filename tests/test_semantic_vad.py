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
    reason="set RUN_LIVEKIT_INTEGRATION=1 (downloads silero weights, ~10s wall clock)",
)
def test_real_agent_session_consults_the_detector_and_commits_a_turn():
    """
    The only test here that proves the plugin actually works.

    Everything else checks a contract in isolation; this drives a real
    `AgentSession` — silero VAD, an STT, `SemanticVAD` as `turn_detection` — with
    audio paced at 1x wall clock, because the endpointing timers are real
    `asyncio` sleeps and pushing a clip at once would make every pause look
    instantaneous.

    Asserts two things a unit test cannot: that LiveKit routes audio into our
    transport and calls `run_inference` at all, and that the session commits the
    user's turn off the probability we return.
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

    class Paced(AudioInput):
        def __init__(self, audio):
            super().__init__(label="paced")
            clip = np.concatenate(
                [np.zeros(int(0.3 * RATE), np.float32), audio, np.zeros(int(6 * RATE), np.float32)]
            )
            self.frames = [clip[i : i + n] for i in range(0, len(clip) - n + 1, n)]
            self.i, self.t0 = 0, None

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

    class Counting:
        def __init__(self, inner):
            self.inner, self.calls = inner, 0
            self.window_seconds = inner.window_seconds

        def predict(self, pcm):
            self.calls += 1
            return self.inner.predict(pcm)

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

    async def run():
        backend = Counting(backends.SmartTurnV3())
        audio_in = Paced(raw)
        finals = []
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=DummySTT(),
            llm=None,
            tts=None,
            turn_handling={
                "turn_detection": SemanticVAD(backend=backend),
                "endpointing": {"min_delay": 0.5, "max_delay": 3.0},
                "interruption": {"mode": "vad"},
            },
            aec_warmup_duration=None,
            user_away_timeout=None,
        )

        @session.on("user_input_transcribed")
        def _on_final(ev):
            if ev.is_final:
                finals.append(ev.transcript)

        session.input.audio = audio_in
        await session.start(Agent(instructions="test"))
        await asyncio.sleep(11.0)
        await session.aclose()
        return backend.calls, finals

    calls, finals = asyncio.run(run())
    assert calls > 0, "LiveKit never called the detector — the transport is not wired in"
    assert finals, "the session never committed the user's turn"
