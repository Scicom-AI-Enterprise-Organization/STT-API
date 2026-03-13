"""Custom Silero VAD plugin for LiveKit Agents with concurrency-limited inference."""

from .vad import VAD, VADStream

__all__ = ["VAD", "VADStream"]
