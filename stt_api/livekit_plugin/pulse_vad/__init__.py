"""
PulseVAD for LiveKit Agents — frame-level voice activity detection, 2,118 params.

A drop-in for `silero.VAD` backed by PulseVAD
(https://github.com/AydinAdnan/PulseVAD, MIT): a depthwise-separable 1D CNN over
a 200 ms causal log-mel window, 12 KB of fp32 weights, 0.097 ms per window
end to end on one CPU thread.

    from stt_api.livekit_plugin.pulse_vad import PulseVAD

    def prewarm(proc):
        proc.userdata["vad"] = PulseVAD.load()          # 2.1k fp32

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        turn_detection=SemanticVAD(backend=ScicomEoT()),  # still a separate job
        stt=..., llm=..., tts=...,
    )

This answers **"is there speech right now"**, which is not what the other two
LiveKit plugins in this package answer — `../turn_detector/` and
`../semantic_vad/` both decide *has the speaker finished*. An `AgentSession`
wants a VAD **and** a turn detector; they are not alternatives.

Two things to read before tuning, both in README.md and both silent when wrong:

* p(speech) saturates at **0.711** (2.1k) / **0.895** (81k), not 1.0. A threshold
  at or above that never fires. `load()` raises rather than let you ship it.
* `precision="fp32"` is the default even though upstream defaults to quantised:
  int8 measured no faster off-microcontroller, in a larger file.
"""

from .frontend import N_FRAMES, N_MELS, SAMPLE_RATE, WINDOW_SAMPLES, log_mel
from .model import CEILINGS, PulseVADModel

__all__ = [
    "CEILINGS",
    "N_FRAMES",
    "N_MELS",
    "PulseVAD",
    "PulseVADModel",
    "PulseVADStream",
    "SAMPLE_RATE",
    "WINDOW_SAMPLES",
    "log_mel",
]


def __getattr__(name: str):
    # Deferred so the model and front end import without livekit-agents present.
    if name in ("PulseVAD", "PulseVADStream"):
        from . import vad

        return getattr(vad, name)
    raise AttributeError(name)
