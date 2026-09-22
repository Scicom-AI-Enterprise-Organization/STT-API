"""
Drive a VAD the way LiveKit drives it, and record what the agent would see.

Both candidates implement `livekit.agents.vad.VAD`, so there is exactly one code
path here and neither model gets a private fast lane. That matters more than it
sounds: this repo has already been bitten by turn-detector tests that asserted
the model was *consulted* rather than that the verdict had an *effect*
(`CLAUDE.md`, "assert the effect, not the call"). Reading probabilities straight
out of an ONNX session would repeat the mistake one layer down — it would skip
the resampling, the frame chunking, the hangover logic and the prefix padding,
which is where a VAD swap actually changes an agent's behaviour.

So audio goes in as 20 ms `rtc.AudioFrame`s, the same shape an SFU delivers, and
what comes back out is the event stream: `START_OF_SPEECH`, `END_OF_SPEECH`, and
the per-inference probabilities underneath them.

One asymmetry is deliberate and must be read into every number downstream: a
probability stamped `t` describes silero's trailing **32 ms** and PulseVAD's
trailing **200 ms**. Both are "what is known at t", which is the fair comparison
for an agent, but it is also precisely why PulseVAD's onset latency is worse and
why no threshold can fix that.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000

__all__ = ["Run", "run_vad", "load_silero", "load_pulse"]


@dataclass
class Run:
    label: str
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    probs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    events: list[tuple[str, float]] = field(default_factory=list)
    inference_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wall_seconds: float = 0.0
    audio_seconds: float = 0.0

    @property
    def rtf(self) -> float:
        """Wall time per second of audio. Contended-machine caveat in README."""
        return (
            self.wall_seconds / self.audio_seconds
            if self.audio_seconds
            else float("nan")
        )


async def run_vad(vad, audio: np.ndarray, *, label: str, frame_ms: int = 20) -> Run:
    """Push `audio` (mono float32, 16 kHz) through `vad` and collect the stream."""
    from livekit import agents, rtc

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    chunk = int(SAMPLE_RATE * frame_ms / 1000)
    stream = vad.stream()

    async def feed() -> None:
        for i in range(0, len(pcm) - chunk + 1, chunk):
            stream.push_frame(
                rtc.AudioFrame(
                    data=pcm[i : i + chunk].tobytes(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=chunk,
                )
            )
            # Yield so the consumer drains; without this the whole file is queued
            # before a single inference runs and `wall_seconds` measures the
            # backlog draining rather than the streaming cost.
            await asyncio.sleep(0)
        stream.end_input()

    times: list[float] = []
    probs: list[float] = []
    infer: list[float] = []
    events: list[tuple[str, float]] = []

    task = asyncio.create_task(feed())
    started = time.perf_counter()
    async for ev in stream:
        if ev.type == agents.vad.VADEventType.INFERENCE_DONE:
            times.append(ev.timestamp)
            probs.append(ev.probability)
            infer.append(ev.inference_duration * 1000.0)
        else:
            events.append((ev.type.value, ev.timestamp))
    wall = time.perf_counter() - started
    await task
    await stream.aclose()

    return Run(
        label=label,
        times=np.asarray(times, dtype=np.float64),
        probs=np.asarray(probs, dtype=np.float64),
        events=events,
        inference_ms=np.asarray(infer, dtype=np.float64),
        wall_seconds=wall,
        audio_seconds=len(audio) / SAMPLE_RATE,
    )


def load_silero(**kw):
    """The incumbent. Defaults left exactly as shipped — that is the baseline."""
    try:
        from livekit.plugins import silero
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "the silero baseline needs livekit-plugins-silero: "
            'pip install ".[benchmark]"'
        ) from e
    return silero.VAD.load(**kw)


def load_pulse(**kw):
    from .. import PulseVAD

    return PulseVAD.load(**kw)
