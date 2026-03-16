"""Dummy LLM plugin — returns a fixed short response. Does nothing."""

from __future__ import annotations

from typing import Any

from livekit.agents import llm, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.llm import Tool, ToolChoice
from livekit.agents.types import NOT_GIVEN, NotGivenOr


class _DummyStream(llm.LLMStream):
    def __init__(self, *, llm_instance: LLM, chat_ctx: llm.ChatContext) -> None:
        super().__init__(
            llm_instance, chat_ctx=chat_ctx, conn_options=DEFAULT_API_CONNECT_OPTIONS
        )

    async def _run(self) -> None:
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="dummy",
                delta=llm.ChoiceDelta(role="assistant", content="OK."),
            )
        )


class LLM(llm.LLM):
    @property
    def model(self) -> str:
        return "dummy"

    @property
    def provider(self) -> str:
        return "dummy"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> _DummyStream:
        return _DummyStream(llm_instance=self, chat_ctx=chat_ctx)
