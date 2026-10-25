"""
Voice isolation for LiveKit Agents: keep the caller, remove every other voice and
the noise, before the VAD, the turn detector and the STT hear the call.

    from stt_api.livekit_plugin.voice_isolator import (
        LocalIsolator, RemoteIsolator, VoiceIsolator,
    )

    backend = LocalIsolator()                          # CPU, in the agent
    backend = RemoteIsolator("ws://gpu-host:8765")     # GPU release, on a server

    room_io.AudioInputOptions(
        sample_rate=16000,
        noise_cancellation=lambda params: VoiceIsolator(backend=backend),
    )

See README.md for which backend to pick and what it does to the agent's VAD.
"""

from .backends import Backend, LocalIsolator, RemoteIsolator, Stream
from .onnx_stream import OnnxStream, isolate

__all__ = [
    "Backend",
    "LocalIsolator",
    "OnnxStream",
    "RemoteIsolator",
    "Stream",
    "VoiceIsolator",
    "isolate",
]


def __getattr__(name: str):
    # Deferred so the backends import without livekit-agents present.
    if name in ("VoiceIsolator", "MODEL_SAMPLE_RATE"):
        from . import isolator

        return getattr(isolator, name)
    raise AttributeError(name)
