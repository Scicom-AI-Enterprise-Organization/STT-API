"""
What a VAD costs the agent's process: N calls fed real-time audio in one process,
and the process's CPU time (all threads) over a fixed stretch.

    taskset -c 10 python -m stt_api.livekit_plugin.remote_silero.benchmark \\
        --arm silero --streams 50 --seconds 30 --audio speech.wav

Arms: `none` (the harness alone: frames made and pushed, no VAD, the baseline to
subtract), `silero` (livekit-plugins-silero, as agents run it), `remote`
(RemoteSileroVAD, `--url`). Every arm gets the same frames on the same clock. Pin
the process to one core and keep the box quiet (CLAUDE.md, "Cost numbers need a
quiet machine").

`--twins same` gives every call two VAD streams pushed the same frame objects, as an
AgentSession with a non-streaming STT does (its own and its StreamAdapter's);
`--twins copy` gives the second stream a copy of each frame, which is what two
streams cost when nothing is shared. `--own-vad` gives every call its own VAD
object, as an agent whose job processes hold one call each has.

Without `--audio` the calls hear a synthetic signal (bursts of noise), so it also runs
where there is no ffmpeg: as a probe of the path from an agent's pod to the server,

    python -m stt_api.livekit_plugin.remote_silero.benchmark --arm remote \\
        --url "$SILERO_VAD_URL" --key-env SGPU_KEY --streams 1 --seconds 60

Prints one JSON line: CPU percent of one core, per call after the baseline is taken
off when `--baseline` is given, windows scored, inference_duration p50/p99 (for
`remote` that is the round trip to the server) and, for `remote`, the VAD's
`stats()`: `local_share` is the share of windows this process scored itself because
the server was late or gone.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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


def synthetic(rate: int, seconds: float = 30.0) -> np.ndarray:
    """Bursts of noise between silences: something for a VAD to chew on, no file."""
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * rate)) / rate
    on = (np.sin(2 * np.pi * t / 3.0) > 0.2).astype(np.float32)
    x = rng.standard_normal(len(t)).astype(np.float32) * 0.1 * on
    return (x * 32767).astype(np.int16)


def cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


async def run(args) -> dict:
    from livekit import agents, rtc

    pcm = (
        load_audio(args.audio, args.rate) if args.audio else synthetic(args.rate, 30.0)
    )
    n = args.rate * args.frame_ms // 1000
    frames_total = len(pcm) // n

    def load():
        if args.arm == "silero":
            from livekit.plugins import silero

            return silero.VAD.load()
        if args.arm == "remote":
            from stt_api.livekit_plugin.remote_silero import RemoteSileroVAD

            headers = None
            if args.key_env:
                headers = {"Authorization": f"Bearer {os.environ[args.key_env]}"}
            return RemoteSileroVAD.load(
                args.url,
                headers=headers,
                max_lag=args.max_lag,
                compensate_delay=args.compensate_delay,
            )
        return None

    # one VAD for every call, or one each: an agent's job process holds one call
    vads = [load() for _ in range(args.streams if args.own_vad else 1)]
    vad = vads[0]
    of = (lambda i: vads[i]) if args.own_vad else (lambda i: vad)
    streams = [of(i).stream() if vad else None for i in range(args.streams)]
    twins = [
        of(i).stream() if vad and args.twins != "no" else None
        for i in range(args.streams)
    ]
    durations: list[float] = []
    counts = {"windows": 0, "starts": 0}
    measuring = False
    at_start: dict = {}

    def stats() -> dict:
        """The VADs' counters added up; the round trip of the first."""
        tot = dict(vad.stats())
        for v in vads[1:]:
            st = v.stats()
            for k in ("windows", "windows_remote", "windows_local", "windows_late"):
                tot[k] += st[k]
            for k in ("connects", "outages", "streams", "connections"):
                tot[k] += st[k]
        n = tot["windows"]
        tot["local_share"] = round(tot["windows_local"] / n, 5) if n else None
        tot["late_share"] = round(tot["windows_late"] / n, 5) if n else None
        return tot

    async def drain(s, counted: bool):
        async for ev in s:
            if not counted:
                continue
            if ev.type == agents.vad.VADEventType.INFERENCE_DONE:
                if measuring:
                    counts["windows"] += 1
                    durations.append(ev.inference_duration)
            elif ev.type == agents.vad.VADEventType.START_OF_SPEECH and measuring:
                counts["starts"] += 1

    tasks = [asyncio.create_task(drain(s, True)) for s in streams if s is not None]
    tasks += [asyncio.create_task(drain(s, False)) for s in twins if s is not None]
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
            if hasattr(vad, "stats"):
                at_start = stats()
        if elapsed >= args.warmup + args.seconds:
            break
        for i, s in enumerate(streams):
            j = (offsets[i] + k) % frames_total
            f = rtc.AudioFrame(pcm[j * n : (j + 1) * n].tobytes(), args.rate, 1, n)
            if s is not None:
                s.push_frame(f)
            if twins[i] is not None:
                twins[i].push_frame(
                    f
                    if args.twins == "same"
                    else rtc.AudioFrame(bytes(f.data), args.rate, 1, n)
                )
        k += 1
        # the next tick on the wall clock, not after this one's work: real time
        await asyncio.sleep(max(0.0, t0 + k * period - time.perf_counter()))
    c1, w1 = cpu_s(), time.perf_counter()
    measuring = False
    at_end = stats() if hasattr(vad, "stats") else None
    for s in streams + twins:
        if s is not None:
            s.end_input()
    await asyncio.gather(*tasks)
    for s in streams + twins:
        if s is not None:
            await s.aclose()
    d = np.sort(np.array(durations)) if durations else np.array([np.nan])
    pct = 100.0 * (c1 - c0) / (w1 - w0)
    out = {
        "arm": args.arm,
        "streams": args.streams,
        "twins": args.twins,
        "own_vad": args.own_vad,
        "rate": args.rate,
        "frame_ms": args.frame_ms,
        "seconds": round(w1 - w0, 2),
        "cpu_pct": round(pct, 3),
        "windows_per_s": round(counts["windows"] / (w1 - w0), 1),
        "starts": counts["starts"],
        "inference_ms_p50": round(1e3 * float(np.percentile(d, 50)), 3),
        "inference_ms_p99": round(1e3 * float(np.percentile(d, 99)), 3),
    }
    if vad is not None and hasattr(vad, "stats"):
        out.update({k: v for k, v in at_end.items() if k not in out})
        # the measured stretch alone: the start (every call connecting at once) is
        # not what a running agent sees
        n = at_end["windows"] - at_start["windows"]
        for k in ("windows_local", "windows_late"):
            out[k.replace("windows_", "") + "_share_measured"] = (
                round((at_end[k] - at_start[k]) / n, 5) if n else None
            )
        out["outages_measured"] = at_end["outages"] - at_start["outages"]
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
    ap.add_argument(
        "--audio",
        default=None,
        help="any file ffmpeg reads, looped; default a synthetic signal",
    )
    ap.add_argument("--url", default=None, help="default SILERO_VAD_URL")
    ap.add_argument(
        "--key-env",
        default=None,
        help="the environment variable that holds the Bearer key to send "
        "(SILERO_VAD_API_KEY is sent without this)",
    )
    ap.add_argument(
        "--twins",
        choices=("no", "same", "copy"),
        default="no",
        help="a second VAD stream per call: pushed the same frames, or copies",
    )
    ap.add_argument(
        "--own-vad",
        action="store_true",
        help="a VAD object per call, as when every call has its own job process",
    )
    ap.add_argument("--max-lag", type=float, default=0.2)
    ap.add_argument("--compensate-delay", action="store_true")
    ap.add_argument(
        "--baseline", type=float, default=None, help="the `none` arm's cpu_pct"
    )
    args = ap.parse_args(argv)
    print(json.dumps(asyncio.run(run(args))), flush=True)


if __name__ == "__main__":
    main()
