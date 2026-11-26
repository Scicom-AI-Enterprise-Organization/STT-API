"""
What a VAD costs the agent's process: N streams fed real-time audio in one process,
and the process's CPU time (all threads) over a fixed stretch.

    taskset -c 10 python -m stt_api.livekit_plugin.remote_silero.benchmark \\
        --arm silero --streams 50 --seconds 30 --audio speech.wav

Arms: `none` (the harness alone: frames made and pushed, no VAD, the baseline to
subtract), `silero` (livekit-plugins-silero, as agents run it), `remote`
(RemoteSileroVAD, `--url`). Every arm gets the same frames on the same clock. Pin
the process to one core and keep the box quiet (CLAUDE.md, "Cost numbers need a
quiet machine").

Prints one JSON line: CPU percent of one core, per stream after the baseline is
taken off when `--baseline` is given, windows scored, and inference_duration p50/p99
(for `remote` that is the round trip to the server).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import subprocess
import time

import numpy as np


def load_audio(path: str, rate: int) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "quiet",
            "-i",
            path,
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.int16)


def cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


async def run(args) -> dict:
    from livekit import agents, rtc

    pcm = load_audio(args.audio, args.rate)
    n = args.rate * args.frame_ms // 1000
    frames_total = len(pcm) // n
    if args.arm == "silero":
        from livekit.plugins import silero

        vad = silero.VAD.load()
    elif args.arm == "remote":
        from stt_api.livekit_plugin.remote_silero import RemoteSileroVAD

        vad = RemoteSileroVAD.load(args.url)
    else:
        vad = None

    streams = [vad.stream() if vad else None for _ in range(args.streams)]
    durations: list[float] = []
    counts = {"windows": 0, "starts": 0}
    measuring = False

    async def drain(s):
        async for ev in s:
            if ev.type == agents.vad.VADEventType.INFERENCE_DONE:
                if measuring:
                    counts["windows"] += 1
                    durations.append(ev.inference_duration)
            elif ev.type == agents.vad.VADEventType.START_OF_SPEECH and measuring:
                counts["starts"] += 1

    tasks = [asyncio.create_task(drain(s)) for s in streams if s is not None]
    # each stream starts at its own offset into the audio, so calls are not in lockstep
    offsets = [(i * 997) % frames_total for i in range(args.streams)]
    period = args.frame_ms / 1000
    t0 = time.perf_counter()
    k = 0
    c0 = w0 = None
    while True:
        elapsed = time.perf_counter() - t0
        if c0 is None and elapsed >= args.warmup:
            measuring = True
            c0, w0 = cpu_s(), time.perf_counter()
        if elapsed >= args.warmup + args.seconds:
            break
        for i, s in enumerate(streams):
            j = (offsets[i] + k) % frames_total
            f = rtc.AudioFrame(pcm[j * n : (j + 1) * n].tobytes(), args.rate, 1, n)
            if s is not None:
                s.push_frame(f)
        k += 1
        # the next tick on the wall clock, not after this one's work: real time
        await asyncio.sleep(max(0.0, t0 + k * period - time.perf_counter()))
    c1, w1 = cpu_s(), time.perf_counter()
    measuring = False
    for s in streams:
        if s is not None:
            s.end_input()
    await asyncio.gather(*tasks)
    for s in streams:
        if s is not None:
            await s.aclose()
    d = np.sort(np.array(durations)) if durations else np.array([np.nan])
    pct = 100.0 * (c1 - c0) / (w1 - w0)
    out = {
        "arm": args.arm,
        "streams": args.streams,
        "rate": args.rate,
        "frame_ms": args.frame_ms,
        "seconds": round(w1 - w0, 2),
        "cpu_pct": round(pct, 3),
        "windows_per_s": round(counts["windows"] / (w1 - w0), 1),
        "starts": counts["starts"],
        "inference_ms_p50": round(1e3 * float(np.percentile(d, 50)), 3),
        "inference_ms_p99": round(1e3 * float(np.percentile(d, 99)), 3),
    }
    if vad is not None and hasattr(vad, "windows_local"):
        out["windows_local"] = vad.windows_local
    if args.baseline is not None:
        out["cpu_pct_per_stream_net"] = round((pct - args.baseline) / args.streams, 4)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", choices=("none", "silero", "remote"), required=True)
    ap.add_argument("--streams", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--frame-ms", type=int, default=20)
    ap.add_argument("--audio", required=True, help="any file ffmpeg reads; looped")
    ap.add_argument("--url", default=None)
    ap.add_argument(
        "--baseline", type=float, default=None, help="the `none` arm's cpu_pct"
    )
    args = ap.parse_args(argv)
    print(json.dumps(asyncio.run(run(args))), flush=True)


if __name__ == "__main__":
    main()
