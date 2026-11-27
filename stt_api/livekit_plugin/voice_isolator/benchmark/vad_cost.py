"""
What the VAD costs a call that already runs `RemoteIsolator`, and whether the
isolator's own scores cut the same turns.

One process is one call, as an agent's job process is: every frame goes through
`VoiceIsolator._process` on the wall clock and what comes back is pushed to the VAD.
The process's CPU time over a fixed stretch, by arm:

    none      the isolator alone: the baseline to take off
    silero    + livekit-plugins-silero, as agents run it
    remote    + RemoteSileroVAD (`--vad-url`): its own connection and upload
    isolator  RemoteIsolator(vad=True) + IsolatorVAD: the scores come with the audio

    taskset -c 10 python -m stt_api.livekit_plugin.voice_isolator.benchmark.vad_cost \\
        --arm isolator --url ws://gpu-host:8765 --audio call.wav --seconds 60

`--twins` pushes every frame to two streams of the VAD, as an AgentSession with a
non-streaming STT does. `--check` also runs upstream silero on the same frames in the
same process and reports how the turns compare (it costs CPU: not with a cost run).

Prints one JSON line. Run the arms one at a time on a quiet box, each pinned to its
own core (CLAUDE.md, "Cost numbers need a quiet machine").
"""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import time

import numpy as np

from ...remote_silero.benchmark import load_audio, synthetic


def cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


async def run(args) -> dict:
    from livekit import agents, rtc

    from .. import RemoteIsolator, VoiceIsolator

    pcm = (
        load_audio(args.audio, args.rate) if args.audio else synthetic(args.rate, 30.0)
    )
    n = args.rate * args.frame_ms // 1000
    total = len(pcm) // n
    vi = VoiceIsolator(backend=RemoteIsolator(args.url, vad=args.arm == "isolator"))
    if args.arm == "silero":
        from livekit.plugins import silero

        vad = silero.VAD.load()
    elif args.arm == "remote":
        from ...remote_silero import RemoteSileroVAD

        vad = RemoteSileroVAD.load(args.vad_url)
    elif args.arm == "isolator":
        from .. import IsolatorVAD

        vad = IsolatorVAD.load()
    else:
        vad = None
    ref = None
    if args.check:
        from livekit.plugins import silero

        ref = silero.VAD.load().stream()

    streams = [vad.stream() for _ in range(2 if args.twins else 1)] if vad else []
    turns: list[tuple[str, float, float]] = []  # kind, stream time, wall time
    ref_turns: list[tuple[str, float, float]] = []
    delays: list[float] = []
    t0 = time.perf_counter()

    async def drain(s, into, first: bool):
        async for ev in s:
            if ev.type == agents.vad.VADEventType.INFERENCE_DONE:
                if first and into is turns:
                    delays.append(ev.inference_duration)
            elif first:
                into.append((ev.type.value, ev.timestamp, time.perf_counter() - t0))

    tasks = [
        asyncio.create_task(drain(s, turns, i == 0)) for i, s in enumerate(streams)
    ]
    if ref is not None:
        tasks.append(asyncio.create_task(drain(ref, ref_turns, True)))
    period = args.frame_ms / 1000
    k = 0
    c0 = w0 = None
    busy = 0.0  # seconds inside _process: the isolator's wait for its reply included
    while True:
        elapsed = time.perf_counter() - t0
        if c0 is None and elapsed >= args.warmup:
            c0, w0 = cpu_s(), time.perf_counter()
        if elapsed >= args.warmup + args.seconds:
            break
        j = k % total
        f = rtc.AudioFrame(pcm[j * n : (j + 1) * n].tobytes(), args.rate, 1, n)
        t = time.perf_counter()
        g = vi._process(f)
        busy += time.perf_counter() - t
        for s in streams:
            s.push_frame(g)
        if ref is not None:
            ref.push_frame(
                rtc.AudioFrame(bytes(g.data), args.rate, 1, g.samples_per_channel)
            )
        k += 1
        await asyncio.sleep(max(0.0, t0 + k * period - time.perf_counter()))
    c1, w1 = cpu_s(), time.perf_counter()
    for s in streams + ([ref] if ref is not None else []):
        s.end_input()
    await asyncio.gather(*tasks)
    for s in streams + ([ref] if ref is not None else []):
        await s.aclose()
    out = {
        "arm": args.arm,
        "twins": bool(args.twins),
        "rate": args.rate,
        "frame_ms": args.frame_ms,
        "seconds": round(w1 - w0, 2),
        "cpu_pct": round(100.0 * (c1 - c0) / (w1 - w0), 3),
        "in_process_pct": round(100.0 * busy / (time.perf_counter() - t0), 1),
        "isolator_delay_ms": round(vi.delay_ms, 1),
        "isolator_unfiltered_ms": round(vi.unfiltered_ms, 1),
        "turns": len(turns),
    }
    vi._close()
    if delays:
        d = np.sort(np.array(delays))
        out["score_delay_ms_p50"] = round(1e3 * float(np.percentile(d, 50)), 3)
        out["score_delay_ms_p99"] = round(1e3 * float(np.percentile(d, 99)), 3)
    if vad is not None and hasattr(vad, "stats"):
        out.update({k: v for k, v in vad.stats().items() if k not in out})
    if args.baseline is not None:
        out["cpu_pct_net"] = round(out["cpu_pct"] - args.baseline, 3)
    if ref is not None and vad is not None:
        # pair each of the VAD's turns with silero's nearest of the same kind
        diffs, wall, unmatched = [], [], 0
        for kind, ts, tw in turns:
            near = [(abs(ts - a), tw - b) for q, a, b in ref_turns if q == kind]
            if not near or min(near)[0] > 0.3:
                unmatched += 1
                continue
            diffs.append(min(near)[0])
            wall.append(min(near)[1])
        out.update(
            {
                "ref_turns": len(ref_turns),
                "unmatched": unmatched,
                "turn_shift_ms_mean": round(1e3 * float(np.mean(diffs)), 1)
                if diffs
                else None,
                "turn_shift_ms_max": round(1e3 * float(np.max(diffs)), 1)
                if diffs
                else None,
                # on the wall clock: how long after silero's event this VAD's came
                "event_after_silero_ms_p50": round(
                    1e3 * float(np.percentile(wall, 50)), 1
                )
                if wall
                else None,
                "event_after_silero_ms_max": round(1e3 * float(np.max(wall)), 1)
                if wall
                else None,
            }
        )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--arm", choices=("none", "silero", "remote", "isolator"), required=True
    )
    ap.add_argument("--url", default=None, help="default VOICE_ISOLATOR_URL")
    ap.add_argument("--vad-url", default=None, help="remote: default SILERO_VAD_URL")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--frame-ms", type=int, default=50, help="LiveKit's default")
    ap.add_argument("--audio", default=None, help="any file ffmpeg reads; looped")
    ap.add_argument("--twins", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--baseline", type=float, default=None, help="the `none` arm's cpu_pct"
    )
    args = ap.parse_args(argv)
    print(json.dumps(asyncio.run(run(args))), flush=True)


if __name__ == "__main__":
    main()
