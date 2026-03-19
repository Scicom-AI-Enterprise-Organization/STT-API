"""Dummy LiveKit plugins for STT, LLM, and TTS.

These plugins return canned responses and silent audio without calling any
external APIs, making them safe for load testing (e.g. lk perf agent-load-test)
without incurring costs from Groq, OpenAI, or other paid services.
"""

from .llm import LLM
from .stt import STT
from .tts import TTS

__all__ = ["STT", "LLM", "TTS"]
