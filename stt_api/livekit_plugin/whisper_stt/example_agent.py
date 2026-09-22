"""
A minimal, runnable agent wired the way the measurements say it should be.

    export STT_URL=... STT_API=... STT_MODEL=...
    python -m stt_api.livekit_plugin.whisper_stt.example_agent dev

Two settings here are not defaults, and both come from measurement rather than
taste. See README.md for the numbers; the short version:

1. `stt=WhisperSTT()` — Whisper answers silence with **a single space**, not an
   empty string, and a space is truthy. Unfiltered, that fires a
   `user_input_transcribed` event carrying `' '`, i.e. a blank chat bubble in
   the UI. 12 % of production turns look like this.

2. `min_interruption_words=1` — **the blank filter alone does not stop the
   agent being interrupted.** The VAD interrupts on its own before the
   transcript matters, because `VADEvent.speech_duration` keeps accumulating
   through silero's 0.55 s silence hangover: a 0.15 s false positive already
   reports 0.70 s, clearing LiveKit's 0.5 s bar. `min_words` is the one gate
   that sits on *both* paths, and it is inert at its default of 0.

Measured end to end on livekit-agents 1.3.11, agent mid-TTS, 0.35 s false
positive:

    setup                     min_words   TTS cut   bubble   transcript
    unfiltered STT                    0     yes        1       ' '
    WhisperSTT                        0     yes        0       --
    unfiltered STT                    1     no         1       ' '
    WhisperSTT                        1     no         0       --
    WhisperSTT, REAL speech           1     yes        1       'saya nak tanya'

The last row is the control that matters: `min_words=1` still lets a genuine
utterance barge in. Raising it above 1 would start costing one-word
interruptions like "stop".
"""

from __future__ import annotations

import logging

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai, silero

from stt_api.livekit_plugin.whisper_stt import WhisperSTT

logger = logging.getLogger("whisper-stt-example")


def prewarm(proc: JobProcess) -> None:
    """Load silero once per worker process, not once per call."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    stt = WhisperSTT()
    # Interruption-related options live on AgentSession, and the spelling is
    # version-dependent:
    #   livekit-agents 1.3.x-1.7.x   min_interruption_words=1
    #   livekit-agents 1.8+          turn_handling={"interruption": {"min_words": 1}}
    # `turn_handling` does not exist before 1.8 and raises TypeError there, so
    # this picks at runtime rather than pinning the file to one version.
    import livekit.agents as agents_pkg

    major, minor = (int(x) for x in agents_pkg.__version__.split(".")[:2])
    turn_opts: dict = (
        {"turn_handling": {"interruption": {"min_words": 1}}}
        if (major, minor) >= (1, 8)
        else {"min_interruption_words": 1}
    )

    session = AgentSession(
        stt=stt,
        # Required: WhisperSTT is non-streaming, so LiveKit wraps it in a
        # StreamAdapter and drives one request per VAD speech segment.
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        **turn_opts,
    )

    @session.on("user_input_transcribed")
    def _on_transcript(ev) -> None:
        # With WhisperSTT in place this never fires for silence. Without it,
        # this is where a blank `' '` bubble would come from.
        if ev.is_final:
            logger.info("user said %r", ev.transcript)

    await session.start(Agent(instructions="You are a helpful assistant."))

    logger.info(
        "started with %s; blank-transcript guard active, min_words=1", stt.model
    )

    # stt.counters shows where segments went, which is the quickest way to see
    # the local gate working: skipped_quiet + skipped_short never hit the network.
    @ctx.add_shutdown_callback
    async def _log_counters() -> None:
        logger.info("STT counters: %s", stt.counters.as_dict())
        await stt.aclose()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
