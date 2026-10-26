"""
Callers for the benchmark: N rooms at once, one participant each publishing a
recording as its microphone, in real time.

    python -m stt_api.livekit_plugin.voice_isolator.benchmark.callers \\
        --url ws://127.0.0.1:17950 --key ... --secret ... --calls 32 --files calls.txt --run local-32

`--files` lists one directory per line holding `mix.wav` (16 kHz mono): the call as it
arrives, the caller and whoever else is on the line. `--channel caller.wav` publishes
that file instead: the caller alone, the reference the other runs are compared with.
Each caller joins its room (`<run>-<i>`, so a worker with automatic dispatch sends an
agent), waits for the agent, publishes, then pushes 20 ms frames (a SIP line's packets)
paced on the wall clock. The calls are spread over `--procs` processes, a few each: one
Python process pushing dozens of calls falls behind real time, and a caller that falls
behind is a benchmark of the caller.

Joining is not part of the measurement. On a host with several network interfaces a
WebRTC participant can take seconds of CPU to connect, so dozens joining at once take
a minute. Every caller therefore connects and waits for its agent first, and the calls
start together at a set time (`--start-in` seconds after launch), spread over
`--spread` seconds so that their first seconds (the enrollment) do not all land on the
same instant: with 60 s recordings they all overlap for at least 50 s. A caller
publishes its microphone only then, so the agent's stream begins with the call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from livekit import api, rtc

SR = 16000
FRAME = SR // 50  # 20 ms


async def call(
    url: str,
    key: str,
    secret: str,
    room_name: str,
    wav: Path,
    go: float,
    timeout: float,
) -> dict:
    token = (
        api.AccessToken(key, secret)
        .with_identity(f"caller-{room_name}")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    room = rtc.Room()
    agent = asyncio.Event()

    @room.on("participant_connected")
    def _on_join(p: rtc.RemoteParticipant) -> None:
        if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            agent.set()

    await room.connect(url, token)
    if any(
        p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        for p in room.remote_participants.values()
    ):
        agent.set()
    await asyncio.wait_for(agent.wait(), timeout)
    # read as float and scale: libsndfile reads a float file as int16 by truncating,
    # which turns every sample of a recording into 0 or +-1
    x, sr = sf.read(wav, dtype="float32")
    assert sr == SR and x.ndim == 1, f"{wav}: 16 kHz mono"
    x = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    # The microphone goes up at the call's start, not at the join: the agent's audio
    # stream begins with the track, and a filter that enrolls the caller live counts
    # from the stream's first frame (a stream that opens with minutes of silence
    # would enroll on silence).
    start_late = time.time() - go
    if start_late < 0:
        await asyncio.sleep(-start_late)
    source = rtc.AudioSource(SR, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    for attempt in range(3):  # a publication can time out while many calls join
        try:
            await room.local_participant.publish_track(track, opts)
            break
        except rtc.participant.PublishTrackError:
            if attempt == 2:
                raise
    await asyncio.sleep(1.0)  # the agent subscribes
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    late = 0.0
    for k in range(0, len(x) - FRAME + 1, FRAME):
        await source.capture_frame(
            rtc.AudioFrame(x[k : k + FRAME].tobytes(), SR, 1, FRAME)
        )
        due = t0 + (k + FRAME) / SR
        wait = due - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        else:
            late = max(late, -wait)
    await asyncio.sleep(
        3.0
    )  # the last turn through the VAD's hangover and the pipeline
    await room.disconnect()
    return {
        "room": room_name,
        "file": str(wav),
        "seconds": len(x) / SR,
        "late_s": late,
        "start_late_s": max(0.0, start_late),
    }


async def group(a: argparse.Namespace, jobs: list[tuple[int, Path]]) -> list:
    tasks = []
    for i, d in jobs:
        go = a.go + i * a.spread / max(1, a.calls)
        tasks.append(
            asyncio.create_task(
                call(
                    a.url,
                    a.key,
                    a.secret,
                    f"{a.run}-{i:03d}",
                    d / a.channel,
                    go,
                    a.agent_timeout,
                )
            )
        )
        await asyncio.sleep(a.stagger * a.procs)
    res = await asyncio.gather(*tasks, return_exceptions=True)
    return [r if isinstance(r, dict) else repr(r) for r in res]


def run_group(
    a: argparse.Namespace, jobs: list[tuple[int, Path]], delay: float
) -> list:
    time.sleep(delay)  # the groups' joins interleave: one join every `stagger` overall
    return asyncio.run(group(a, jobs))


def main() -> None:
    ap = argparse.ArgumentParser()
    # no defaults: a benchmark must never land on whatever server answers on a port
    ap.add_argument("--url", default=os.environ.get("LIVEKIT_URL"))
    ap.add_argument("--key", default=os.environ.get("LIVEKIT_API_KEY"))
    ap.add_argument("--secret", default=os.environ.get("LIVEKIT_API_SECRET"))
    ap.add_argument("--files", required=True)
    ap.add_argument("--calls", type=int, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--channel", default="mix.wav")
    ap.add_argument("--out", default="/tmp/vi_bench")
    ap.add_argument("--stagger", type=float, default=0.1, help="seconds between joins")
    ap.add_argument(
        "--procs", type=int, default=0, help="caller processes (0: one per 4 calls)"
    )
    ap.add_argument(
        "--start-in",
        type=float,
        default=None,
        help="seconds from launch to the first call's audio (default 30 + 2.5 per call)",
    )
    ap.add_argument(
        "--spread", type=float, default=10.0, help="seconds over which calls start"
    )
    ap.add_argument(
        "--agent-timeout",
        type=float,
        default=600.0,
        help="seconds to wait for the agent",
    )
    a = ap.parse_args()
    if not (a.url and a.key and a.secret):
        ap.error(
            "--url, --key and --secret (or LIVEKIT_URL / _API_KEY / _API_SECRET) are required"
        )
    dirs = [Path(d) for d in Path(a.files).read_text().split()][: a.calls]
    a.procs = a.procs or max(1, math.ceil(len(dirs) / 4))
    a.calls = len(dirs)
    t = time.time()
    a.go = t + (a.start_in if a.start_in is not None else 30.0 + 2.5 * len(dirs))
    jobs = list(enumerate(dirs))
    # spawned, not forked: each process starts its own LiveKit runtime
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(a.procs, mp_context=ctx) as pool:
        futs = [
            pool.submit(run_group, a, jobs[g :: a.procs], g * a.stagger)
            for g in range(a.procs)
        ]
        res = [r for f in futs for r in f.result()]
    ok = [r for r in res if isinstance(r, dict)]
    bad = [r for r in res if not isinstance(r, dict)]
    out = Path(a.out) / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "callers.json").write_text(
        json.dumps({"calls": ok, "errors": bad, "wall_s": time.time() - t}, indent=1)
    )
    print(f"{a.run}: {len(ok)} calls, {len(bad)} errors", flush=True)


if __name__ == "__main__":
    main()
