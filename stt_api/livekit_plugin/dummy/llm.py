from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from livekit.agents import llm
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

logger = logging.getLogger("dummy-llm")

DUMMY_RESPONSES = [
    "I'm doing well, thank you for asking! How can I help you today?",
    "The weather looks nice today. Is there anything else you'd like to know?",
    "That's a great question. Let me think about that for a moment.",
    "Sure, I'd be happy to help you with that.",
    "I appreciate you reaching out. What else can I assist you with?",
]


class LLM(llm.LLM):
    """Dummy LLM plugin that returns canned responses without calling any external API."""

    def __init__(self) -> None:
        super().__init__()
        self._response_index = 0

    @property
    def model(self) -> str:
        return "dummy"

    @property
    def provider(self) -> str:
        return "dummy"

    def _next_response(self) -> str:
        text = DUMMY_RESPONSES[self._response_index % len(DUMMY_RESPONSES)]
        self._response_index += 1
        return text

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: Optional[List[llm.Tool]] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict] = NOT_GIVEN,
    ) -> DummyLLMStream:
        return DummyLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            response_text=self._next_response(),
        )


class DummyLLMStream(llm.LLMStream):
    """Dummy LLM stream that emits a canned response in chunks."""

    def __init__(
        self,
        *,
        llm: LLM,
        chat_ctx: llm.ChatContext,
        tools: List[llm.Tool],
        conn_options: APIConnectOptions,
        response_text: str,
    ) -> None:
        super().__init__(llm=llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._response_text = response_text

    async def _run(self) -> None:
        request_id = str(uuid.uuid4())
        words = self._response_text.split(" ")
        for i, word in enumerate(words):
            content = word if i == 0 else f" {word}"
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=request_id,
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        content=content,
                    ),
                )
            )
        logger.debug("dummy LLM responded: %s", self._response_text)
