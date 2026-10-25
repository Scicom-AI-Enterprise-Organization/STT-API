"""
The benchmark's agent: a LiveKit Agents worker as production runs it, with the filter
under test on its audio input.

    LIVEKIT_URL=ws://127.0.0.1:17950 LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... \\
    VI_BENCH_FILTER=local VI_BENCH_MODEL=/models/voice-isolator-cpu VI_BENCH_OUT=/tmp/bench/local \\
        python -m stt_api.livekit_plugin.voice_isolator.benchmark.agent start

VI_BENCH_FILTER: none | gtcrn | local | local-int8 | remote (VOICE_ISOLATOR_URL).

Every room gets an `AgentSession` with LiveKit's Silero VAD (its defaults) and the
dummy STT, LLM and TTS, so a turn runs the whole pipeline without paying for one. The
filter is wrapped in `Timed`, which LiveKit calls exactly where it would call the
filter, and which records each frame: when it arrived, how many samples came before
it, how long the filter took and its input level (to line the stream up with the
recording). The VAD's verdicts come from the session's own
`user_state_changed` events. One JSON per room in VI_BENCH_OUT on shutdown, with the
job process's CPU time.
"""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

import numpy as np
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.voice import room_io
from livekit.plugins import silero

from stt_api.livekit_plugin.dummy import LLM, STT, TTS

FILTER = os.environ.get("VI_BENCH_FILTER", "none")
OUT = Path(os.environ.get("VI_BENCH_OUT", "/tmp/vi_bench"))


class Timed(rtc.FrameProcessor[rtc.AudioFrame]):
    """The filter under test (or none), timed frame by frame where LiveKit calls it."""

    def __init__(self, inner: rtc.FrameProcessor[rtc.AudioFrame] | None) -> None:
        self.inner = inner
        self._enabled = True
        self.t0: float | None = None
        self.n = 0
        # (arrival s, samples before it, filter s, input level dBFS)
        self.frames: list[tuple[float, int, float, float]] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        out = self.inner._process(frame) if self.inner is not None else frame
        dt = time.perf_counter() - now
        x = np.frombuffer(frame.data, np.int16).astype(np.float32) / 32768.0
        db = float(10 * np.log10(np.mean(x * x) + 1e-12))
        self.frames.append((now - self.t0, self.n, dt, db))
        self.n += frame.samples_per_channel
        return out

    def _close(self) -> None:
        if self.inner is not None:
            self.inner._close()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()
    if FILTER in ("local", "local-int8"):
        from stt_api.livekit_plugin.voice_isolator import LocalIsolator

        proc.userdata["backend"] = LocalIsolator(
            os.environ.get("VI_BENCH_MODEL"), quantized=FILTER == "local-int8"
        )
    elif FILTER == "remote":
        from stt_api.livekit_plugin.voice_isolator import RemoteIsolator

        proc.userdata["backend"] = RemoteIsolator()


def make_filter(proc: JobProcess) -> rtc.FrameProcessor[rtc.AudioFrame] | None:
    if FILTER == "gtcrn":
        from stt_api.livekit_plugin.noise_cancellation import GTCRN

        return GTCRN()
    if FILTER in ("local", "local-int8", "remote"):
        from stt_api.livekit_plugin.voice_isolator import VoiceIsolator

        return VoiceIsolator(backend=proc.userdata["backend"])
    return None


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    cpu0, wall0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    timed: list[Timed] = []
    events: list[tuple[float, str]] = []

    def selector(params) -> Timed:
        t = Timed(make_filter(ctx.proc))
        timed.append(t)
        return t

    session = AgentSession(
        stt=STT(), llm=LLM(), tts=TTS(), vad=ctx.proc.userdata["vad"]
    )

    @session.on("user_state_changed")
    def _on_user_state(ev) -> None:
        events.append((time.perf_counter(), ev.new_state))

    async def write_log() -> None:
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        t = timed[0] if timed else None
        log = {
            "room": ctx.room.name,
            "filter": FILTER,
            "seconds": time.perf_counter() - wall0,
            "cpu_s": (cpu.ru_utime - cpu0.ru_utime) + (cpu.ru_stime - cpu0.ru_stime),
            "rss_mb": cpu.ru_maxrss / 1024,
            "frames": t.frames if t else [],
            # user states in the stream's own time: seconds since its first frame
            "vad": [(w - t.t0, s) for w, s in events] if t and t.t0 is not None else [],
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{ctx.room.name}.json").write_text(json.dumps(log))

    ctx.add_shutdown_callback(write_log)
    await session.start(
        agent=Agent(instructions="You are a benchmark."),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                sample_rate=16000, noise_cancellation=selector, pre_connect_audio=False
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # every call gets a warm process: a benchmark of N calls is not a
            # benchmark of process start-up
            num_idle_processes=int(os.environ.get("VI_BENCH_IDLE", "8")),
            # the benchmark decides how many calls there are; the default load
            # (the container's CPU, cgroup-wide) reads other tenants' work on a
            # shared machine and turns rooms away
            load_fnc=lambda *_: 0.0,
            load_threshold=0.95,
            port=int(
                os.environ.get("VI_BENCH_HTTP_PORT", "0")
            ),  # health check: any free port
        )
    )
