"""
Semantic VAD for LiveKit Agents — end-of-turn decided from audio, not transcript.

The text turn detectors in `../turn_detector/` cannot answer until the STT has
produced words. On a production stack that transcript arrives a median 1.5 s
after the speaker stops, so no text detector can help before then. An
audio-native model answers from audio the agent already has.

    from stt_api.livekit_plugin.semantic_vad import SemanticVAD, SmartTurnV3

    session = AgentSession(
        stt=..., llm=..., tts=...,
        vad=ctx.proc.userdata["vad"],
        turn_detection=SemanticVAD(backend=SmartTurnV3()),
    )

See README.md for how this plugs into LiveKit's audio EoT interface without
implementing their protobuf websocket server, and for which backend to pick.
"""

from .backends import Backend, RemoteEoT, SmartTurnV3

__all__ = ["Backend", "RemoteEoT", "SemanticVAD", "SmartTurnV3"]


def __getattr__(name: str):
    # Deferred so the backends import without livekit-agents present.
    if name == "SemanticVAD":
        from .detector import SemanticVAD

        return SemanticVAD
    raise AttributeError(name)
