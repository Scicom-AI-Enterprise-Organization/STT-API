"""
Tests for the Whisper STT plugin.

The defect this plugin exists to prevent is silent in the worst way: Whisper
answers silence with **a single space**, LiveKit only tests emptiness, and a
space is truthy — so the agent gets interrupted by silence with nothing logged.
These tests therefore assert the *effect* rather than the call, in the sense
`CLAUDE.md` means it: the thing checked is `bool(alternatives[0].text)`, which is
exactly the expression `audio_recognition.py:1215` evaluates before deciding to
fire `on_final_transcript` -> `_interrupt_by_audio_activity()`.

Two shapes of wrong are guarded specifically:

* Returning `alternatives=[]` instead of empty text. That stops the interrupt
  and introduces an `IndexError`, because `audio_recognition.py:1207` indexes
  `alternatives[0]` with no length check.
* Filtering on "has no letters" rather than on stripped length. Four turns in
  the production sample are real Tamil and Chinese speech containing no ASCII
  alphanumerics at all, and an `[A-Za-z0-9]` test would discard them.

Nothing here needs the network: the HTTP session is injected.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("livekit.agents", reason="livekit extra not installed")

from livekit import rtc  # noqa: E402
from livekit.agents import APIConnectOptions  # noqa: E402

from stt_api.livekit_plugin.whisper_stt import (  # noqa: E402
    DropBlankSTT,
    WhisperSTT,
    is_blank,
)

RATE = 16000
CONN = APIConnectOptions(max_retry=0, timeout=10)


def frame(pcm: np.ndarray, rate: int = RATE) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=pcm.tobytes(),
        sample_rate=rate,
        num_channels=1,
        samples_per_channel=len(pcm),
    )


def tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    return (np.sin(2 * np.pi * 220 * t) * amplitude * 32767).astype(np.int16)


def _stt(**kw) -> WhisperSTT:
    kw.setdefault("base_url", "https://example.invalid/stt")
    kw.setdefault("api_key", "test")
    kw.setdefault("model", "test-model")
    return WhisperSTT(**kw)


# --- what counts as blank ---------------------------------------------------


@pytest.mark.parametrize("text", ["", " ", "   ", "\n", "\t ", "　"])
def test_blank_texts_are_blank(text):
    assert is_blank(text)


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        ".",  # punctuation is content by this rule, matching the ".strip() length" spec
        " ஹலோ மாயா!",  # real Tamil speech, zero ASCII alphanumerics
        " 你好,我想现在付款",  # real Chinese speech, zero ASCII alphanumerics
    ],
)
def test_real_speech_is_not_blank(text):
    """An `[A-Za-z0-9]` filter would drop the last two. `.strip()` does not."""
    assert not is_blank(text)


def test_none_is_blank():
    assert is_blank(None)


# --- the blank event's shape is load-bearing --------------------------------


@pytest.mark.asyncio
async def test_blank_result_has_one_alternative_with_empty_text():
    """
    Not `alternatives=[]`: `audio_recognition.py:1207` does `alternatives[0]`
    without a length check, so an empty list swaps the interrupt for a crash.
    """
    stt = _stt()
    ev = await stt.recognize([frame(np.zeros(RATE, np.int16))], conn_options=CONN)
    assert len(ev.alternatives) == 1
    assert ev.alternatives[0].text == ""
    # the exact expression LiveKit evaluates before interrupting
    assert not ev.alternatives[0].text


# --- guard 1: never spend a request that cannot return words ----------------


@pytest.mark.asyncio
async def test_segment_below_the_duration_floor_is_skipped_without_a_request():
    stt = _stt(min_audio_duration=0.1)
    ev = await stt.recognize([frame(tone(0.05))], conn_options=CONN)
    assert not ev.alternatives[0].text
    assert stt.counters.skipped_short == 1
    assert stt.counters.requests == 0


@pytest.mark.asyncio
async def test_silence_is_skipped_without_a_request():
    """169 of 180 production non-speech turns are literally all-zero samples."""
    stt = _stt()
    ev = await stt.recognize(
        [frame(np.zeros(int(0.3 * RATE), np.int16))], conn_options=CONN
    )
    assert not ev.alternatives[0].text
    assert stt.counters.skipped_quiet == 1
    assert stt.counters.requests == 0


@pytest.mark.asyncio
async def test_the_rms_floor_can_be_disabled():
    stt = _stt(silence_floor_dbfs=None, http_session=_FakeSession(" "))
    await stt.recognize([frame(np.zeros(RATE, np.int16))], conn_options=CONN)
    assert stt.counters.skipped_quiet == 0
    assert stt.counters.requests == 1


@pytest.mark.asyncio
async def test_audible_speech_is_not_skipped():
    stt = _stt(http_session=_FakeSession("ya betul"))
    ev = await stt.recognize([frame(tone(1.0))], conn_options=CONN)
    assert stt.counters.requests == 1
    assert ev.alternatives[0].text == "ya betul"


# --- guard 2: a blank answer must not read as user activity -----------------


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status = payload, status

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    """Stands in for aiohttp; records what the plugin would have sent."""

    def __init__(self, text, status=200):
        self._text, self._status = text, status
        self.calls = 0

    def post(self, url, **kw):
        self.calls += 1
        return _FakeResponse({"text": self._text}, status=self._status)

    async def close(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("returned", [" ", "", "  \n\t"])
async def test_whitespace_response_cannot_interrupt(returned):
    """
    The headline case. `' '` is what this endpoint returns for silence, and it
    is truthy — so without this guard it passes LiveKit's `if not transcript`
    and interrupts the agent.
    """
    session = _FakeSession(returned)
    stt = _stt(http_session=session)
    ev = await stt.recognize([frame(tone(1.0))], conn_options=CONN)
    assert session.calls == 1
    assert not ev.alternatives[0].text, "a blank transcript would interrupt the agent"
    assert stt.counters.blank_responses == 1
    assert stt.counters.transcribed == 0


@pytest.mark.asyncio
async def test_real_transcript_passes_through_unchanged():
    stt = _stt(http_session=_FakeSession(" Boleh saya tahu baki akaun?"))
    ev = await stt.recognize([frame(tone(1.0))], conn_options=CONN)
    assert ev.alternatives[0].text == " Boleh saya tahu baki akaun?"
    assert stt.counters.transcribed == 1


@pytest.mark.asyncio
async def test_non_latin_transcript_survives():
    stt = _stt(http_session=_FakeSession(" ஹலோ மாயா!"))
    ev = await stt.recognize([frame(tone(1.0))], conn_options=CONN)
    assert ev.alternatives[0].text == " ஹலோ மாயா!"


# --- configuration ----------------------------------------------------------


def test_missing_endpoint_or_model_is_refused_clearly():
    with pytest.raises(ValueError, match="STT_URL"):
        WhisperSTT(base_url="", model="m", api_key="k", env_file="/nonexistent")
    with pytest.raises(ValueError, match="STT_MODEL"):
        WhisperSTT(base_url="https://x", model="", api_key="k", env_file="/nonexistent")


def test_capabilities_are_non_streaming():
    """
    Non-streaming, so `AgentSession` wraps it in `StreamAdapter` (agent.py:522)
    and a `vad=` is required alongside it.
    """
    caps = _stt().capabilities
    assert caps.streaming is False
    assert caps.interim_results is False


# --- the wrapper for other providers ----------------------------------------


class _StubSTT:
    """Minimal stand-in for another plugin, e.g. livekit-plugins-openai."""

    def __init__(self, text):
        from livekit.agents import stt as lkstt

        self.capabilities = lkstt.STTCapabilities(
            streaming=False, interim_results=False
        )
        self._text = text
        self.model = "stub"
        self.provider = "stub"

    async def recognize(self, buffer, **kw):
        from livekit.agents import stt as lkstt

        return lkstt.SpeechEvent(
            type=lkstt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[lkstt.SpeechData(language="en", text=self._text)],
        )

    async def aclose(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("returned,expected", [(" ", ""), ("", ""), ("hi", "hi")])
async def test_wrapper_filters_any_provider(returned, expected):
    stt = DropBlankSTT(_StubSTT(returned))
    ev = await stt.recognize([frame(tone(1.0))], conn_options=CONN)
    assert ev.alternatives[0].text == expected
    assert len(ev.alternatives) == 1
