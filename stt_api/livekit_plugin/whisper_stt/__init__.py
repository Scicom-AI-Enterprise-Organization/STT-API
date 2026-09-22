"""
Whisper STT for LiveKit Agents — silence does not get to interrupt the agent.

    from stt_api.livekit_plugin.whisper_stt import WhisperSTT

    session = AgentSession(
        stt=WhisperSTT(),            # reads STT_URL / STT_API / STT_MODEL
        vad=silero.VAD.load(),       # required: this STT is not streaming
        llm=..., tts=...,
    )

Two guards, both measured against production audio (see README.md):

* **A blank transcript returns `text=""`, never a space.** Whisper answers
  silence with `' '`, which is truthy, so it passes LiveKit's `if not
  transcript` guard and interrupts the agent. 12 % of production turns look like
  this.
* **Silent segments never reach the network.** An RMS floor skips 94 % of the
  useless requests and loses no real speech.

`DropBlankSTT` applies the first guard to any other STT, for keeping an existing
provider — `livekit-plugins-openai` has the same defect.
"""

from .stt import Counters, is_blank, load_env

__all__ = ["Counters", "DropBlankSTT", "WhisperSTT", "is_blank", "load_env"]


def __getattr__(name: str):
    # Deferred so is_blank/load_env import without livekit-agents present.
    if name in ("WhisperSTT", "DropBlankSTT"):
        from . import stt

        return getattr(stt, name)
    raise AttributeError(name)
