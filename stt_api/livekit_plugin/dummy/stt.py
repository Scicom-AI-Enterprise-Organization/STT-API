from __future__ import annotations

import logging
import uuid

from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

logger = logging.getLogger("dummy-stt")

DUMMY_TRANSCRIPTS = [
    "Hello, how are you doing today?",
    "Can you tell me about the weather?",
    "What time is it right now?",
    "Thank you for your help.",
    "That sounds great, let's do it.",
]


class STT(stt.STT):
    """Dummy STT plugin that returns canned transcriptions without calling any external API."""

    def __init__(
        self,
        *,
        language: str = "en",
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._language = language
        self._transcript_index = 0

    @property
    def model(self) -> str:
        return "dummy"

    @property
    def provider(self) -> str:
        return "dummy"

    def _next_transcript(self) -> str:
        text = DUMMY_TRANSCRIPTS[self._transcript_index % len(DUMMY_TRANSCRIPTS)]
        self._transcript_index += 1
        return text

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        text = self._next_transcript()
        logger.debug("dummy STT recognized: %s", text)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[
                stt.SpeechData(
                    language=self._language,
                    text=text,
                    confidence=1.0,
                )
            ],
        )
