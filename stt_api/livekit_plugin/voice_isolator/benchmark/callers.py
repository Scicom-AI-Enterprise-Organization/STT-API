"""
Callers for the benchmark: N rooms at once, one participant each publishing a
recording as its microphone, in real time.

    python -m stt_api.livekit_plugin.voice_isolator.benchmark.callers \\
        --url ws://127.0.0.1:17950 --key ... --secret ... --calls 32 --files calls.txt --run local-32

`--files` lists one directory per line holding `mix.wav` (16 kHz mono): the call as it
arrives, the caller and whoever else is on the line. `--channel caller.wav` publishes
that file instead: the caller alone, the reference the other runs are compared with.
Each caller joins its room (`<run>-<i>`, so a worker with automatic dispatch sends an
agent), waits for the agent, publishes, then pushes 10 ms frames paced on the wall
clock, the way a phone line delivers them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from livekit import api, rtc

SR = 16000


async def call(url: str, key: str, secret: str, room_name: str, wav: Path) -> dict:
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
    await asyncio.wait_for(agent.wait(), 60)
    source = rtc.AudioSource(SR, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, opts)
    await asyncio.sleep(1.0)  # the agent subscribes
    # read as float and scale: libsndfile reads a float file as int16 by truncating,
    # which turns every sample of a recording into 0 or +-1
    x, sr = sf.read(wav, dtype="float32")
    assert sr == SR and x.ndim == 1, f"{wav}: 16 kHz mono"
    x = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    late = 0.0
    for k in range(0, len(x) - 159, 160):
        await source.capture_frame(rtc.AudioFrame(x[k : k + 160].tobytes(), SR, 1, 160))
        due = t0 + (k + 160) / SR
        wait = due - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        else:
            late = max(late, -wait)
    await asyncio.sleep(
        3.0
    )  # the last turn through the VAD's hangover and the pipeline
    await room.disconnect()
    return {"room": room_name, "file": str(wav), "seconds": len(x) / SR, "late_s": late}


async def main() -> None:
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
    ap.add_argument("--stagger", type=float, default=0.05, help="seconds between joins")
    a = ap.parse_args()
    if not (a.url and a.key and a.secret):
        ap.error(
            "--url, --key and --secret (or LIVEKIT_URL / _API_KEY / _API_SECRET) are required"
        )
    dirs = [Path(d) for d in Path(a.files).read_text().split()][: a.calls]
    tasks = []
    for i, d in enumerate(dirs):
        tasks.append(
            asyncio.create_task(
                call(a.url, a.key, a.secret, f"{a.run}-{i:03d}", d / a.channel)
            )
        )
        await asyncio.sleep(a.stagger)
    t = time.time()
    res = await asyncio.gather(*tasks, return_exceptions=True)
    ok = [r for r in res if isinstance(r, dict)]
    bad = [repr(r) for r in res if not isinstance(r, dict)]
    out = Path(a.out) / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "callers.json").write_text(
        json.dumps({"calls": ok, "errors": bad, "wall_s": time.time() - t}, indent=1)
    )
    print(f"{a.run}: {len(ok)} calls, {len(bad)} errors", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
