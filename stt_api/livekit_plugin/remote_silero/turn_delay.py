"""
How much later than local silero a remote VAD reports a turn's start and end.

One call in real time, its frames pushed to three VADs at once: livekit-plugins-silero
on this CPU, `RemoteSileroVAD`, and `RemoteSileroVAD(compensate_delay=True)`. Each
START_OF_SPEECH and END_OF_SPEECH is timed on the wall clock and paired with local
silero's event of the same kind for the same turn.

    python -m stt_api.livekit_plugin.remote_silero.turn_delay \\
        --url wss://host/proxy/silero-vad/ --key-env SGPU_KEY --audio call.mp3

Prints one JSON line: per VAD, the median and the worst of (its event - local
silero's), starts and ends apart, and the round trip.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import numpy as np

from .benchmark import load_audio


async def run(args) -> dict:
    from livekit import agents, rtc
    from livekit.plugins import silero

    from . import RemoteSileroVAD

    pcm = load_audio(args.audio, args.rate)[: int(args.seconds * args.rate)]
    n = args.rate * args.frame_ms // 1000
    headers = None
    if args.key_env:
        headers = {"Authorization": f"Bearer {os.environ[args.key_env]}"}
    vads = {
        "silero": silero.VAD.load(),
        "remote": RemoteSileroVAD.load(args.url, headers=headers, max_lag=args.max_lag),
        "remote_compensated": RemoteSileroVAD.load(
            args.url, headers=headers, max_lag=args.max_lag, compensate_delay=True
        ),
    }
    streams = {k: v.stream() for k, v in vads.items()}
    events: dict[str, list[tuple[str, float, float]]] = {k: [] for k in vads}
    t0 = time.perf_counter()

    async def drain(name):
        async for ev in streams[name]:
            if ev.type != agents.vad.VADEventType.INFERENCE_DONE:
                events[name].append(
                    (ev.type.value, ev.timestamp, time.perf_counter() - t0)
                )

    tasks = [asyncio.create_task(drain(k)) for k in vads]
    for k in range(len(pcm) // n):
        f = rtc.AudioFrame(pcm[k * n : (k + 1) * n].tobytes(), args.rate, 1, n)
        for s in streams.values():
            s.push_frame(f)
        await asyncio.sleep(
            max(0.0, t0 + (k + 1) * args.frame_ms / 1000 - time.perf_counter())
        )
    for s in streams.values():
        s.end_input()
    await asyncio.gather(*tasks)
    for s in streams.values():
        await s.aclose()

    ref = events["silero"]
    out: dict = {"turns": sum(k == "start_of_speech" for k, _, _ in ref)}
    for name in ("remote", "remote_compensated"):
        row: dict = {}
        for kind, label in (("start_of_speech", "start"), ("end_of_speech", "end")):
            # the same turn: local silero's event of this kind nearest in the stream
            late = []
            for k, ts, tw in events[name]:
                near = [(abs(ts - a), tw - b) for q, a, b in ref if q == kind == k]
                if near and min(near)[0] < 0.3:
                    late.append(min(near)[1])
            row[f"{label}s"] = len(late)
            row[f"{label}_after_silero_ms_p50"] = (
                round(1e3 * float(np.median(late)), 1) if late else None
            )
            row[f"{label}_after_silero_ms_max"] = (
                round(1e3 * float(np.max(late)), 1) if late else None
            )
        st = vads[name].stats()
        row.update(
            {
                k: st[k]
                for k in ("round_trip_ms_p50", "round_trip_ms_p99", "local_share")
            }
        )
        out[name] = row
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=None, help="default SILERO_VAD_URL")
    ap.add_argument("--key-env", default=None)
    ap.add_argument("--audio", required=True, help="any file ffmpeg reads")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--frame-ms", type=int, default=20)
    ap.add_argument("--max-lag", type=float, default=0.2)
    args = ap.parse_args(argv)
    print(json.dumps(asyncio.run(run(args))), flush=True)


if __name__ == "__main__":
    main()
