"""
Voice isolation for LiveKit Agents: keep the caller, remove every other voice and
the noise, before the VAD, the turn detector and the STT hear the call.

    from stt_api.livekit_plugin.voice_isolator import (
        LocalIsolator, RemoteIsolator, VoiceIsolator,
    )

    backend = LocalIsolator()                          # CPU, in the agent
    backend = RemoteIsolator("ws://gpu-host:8765")     # GPU release, on a server
    backend = LocalIsolator(voicebot=True)             # an agent that speaks first

    room_io.AudioInputOptions(
        sample_rate=16000,
        noise_cancellation=lambda params: VoiceIsolator(backend=backend),
    )

`RemoteIsolator(url, vad=True)` with `IsolatorVAD.load()` as the session's VAD: the
server scores the audio it sends, and the agent runs no VAD model (vad.py).

See README.md for which backend to pick and what it does to the agent's VAD.
"""

from .backends import Backend, LocalIsolator, RemoteIsolator, Stream
from .onnx_stream import OnnxStream, isolate

__all__ = [
    "Backend",
    "IsolatorVAD",
    "LocalIsolator",
    "OnnxStream",
    "RemoteIsolator",
    "Stream",
    "VoiceIsolator",
    "isolate",
]


def __getattr__(name: str):
    # Deferred so the backends import without livekit-agents present.
    if name in ("VoiceIsolator", "MODEL_SAMPLE_RATE", "VAD_KEY"):
        from . import isolator

        return getattr(isolator, name)
    if name == "IsolatorVAD":
        from . import vad

        return vad.IsolatorVAD
    raise AttributeError(name)
