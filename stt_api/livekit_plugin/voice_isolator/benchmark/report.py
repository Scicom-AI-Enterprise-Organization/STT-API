"""
Numbers from a benchmark: the filter's cost inside the agent and what the agent's VAD
did, against the run that published the caller alone.

    python -m stt_api.livekit_plugin.voice_isolator.benchmark.report \\
        --root /tmp/vi_bench --reference ref-32 --runs none-32 gtcrn-32 local-32 remote-32

Per run, from every room's agent log (`agent.py`):

- **frame**: what `_process` took per frame, median / p99 / max (ms); **filter**:
  its share of one core per call (time in `_process` / call length); **job**: the job
  process's whole CPU per call (the VAD, the dummy pipeline, WebRTC and the filter).
- **drift**: how far the agent fell behind the stream over a call (the spread of
  arrival time minus audio position, p99 over calls); a filter that cannot keep up
  shows here first.

And, against the reference run (the same recordings with only the caller on the line,
no filter), each room lined up with its recording by its level contour (`offset`):

- **false starts**: the VAD opens a turn while the caller is silent (no caller turn
  starts within 0.3 s): the agent stops talking for nobody. Per minute.
- **pauses heard as speech**: share of the caller's silence the VAD calls speech.
- **turn end p90**: how much later than on the caller alone the VAD closes the caller's
  turn, 90th percentile; **held**: caller turns it never closes before the caller talks
  again (the agent never gets its turn).
- **caller missed**: share of the caller's speech the VAD does not hear.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
GRID = 0.032


def offset(frames: np.ndarray, wav: Path, max_s: float = 10.0) -> int:
    """Stream samples before the recording's sample 0.

    The level contour of the frames the agent received (dBFS per frame, after Opus and
    WebRTC's start-up) against the recording's, at every 10 ms shift: the shift with
    the smallest difference over the whole call. A first-sound threshold is not
    enough: the first frames through WebRTC come in quieter, by a different number of
    frames in every room."""
    pos, lev = frames[:, 1], np.clip(frames[:, 3], -70.0, 0.0)
    n = int(np.median(np.diff(pos)))
    x = sf.read(wav, dtype="float32")[0].astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    starts = np.arange(0, len(x) - n, 160)
    rec = np.clip(10 * np.log10((cs[starts + n] - cs[starts]) / n + 1e-12), -70.0, 0.0)
    best, best_err = 0, np.inf
    for off in range(0, int(max_s * SR), 160):
        r = pos - off
        ok = (r >= 0) & (r < starts[-1])
        if ok.sum() < 50:
            continue
        err = float(np.mean((lev[ok] - rec[(r[ok] // 160).astype(int)]) ** 2))
        if err < best_err:
            best, best_err = off, err
    return best


def turns(log: dict, wav: Path) -> list[tuple[float, float]]:
    """The VAD's speaking intervals of one room, in seconds of its recording."""
    fr = np.asarray(log["frames"], dtype=np.float64)
    if not len(fr):
        return []
    arr, pos = fr[:, 0], fr[:, 1]
    off = offset(fr, wav)

    def file_time(w: float) -> float:
        j = int(np.clip(np.searchsorted(arr, w, side="right") - 1, 0, len(arr) - 1))
        return (pos[j] + (w - arr[j]) * SR - off) / SR

    out, start = [], None
    for w, state in log["vad"]:
        t = file_time(w)
        if state == "speaking" and start is None:
            start = t
        elif state != "speaking" and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, (pos[-1] - off) / SR))
    return out


def flags(iv: list[tuple[float, float]], seconds: float) -> np.ndarray:
    t = (np.arange(int(seconds / GRID)) + 0.5) * GRID
    f = np.zeros(len(t), bool)
    for a, b in iv:
        f |= (t >= a) & (t < b)
    return f


def against(test: list, ref: list, seconds: float) -> dict:
    ft, fr = flags(test, seconds), flags(ref, seconds)
    r_on = np.array([a for a, _ in ref])
    false_starts = sum(
        1
        for a, _ in test
        if not fr[min(len(fr) - 1, max(0, int(a / GRID)))]
        and not np.any(np.abs(r_on - a) < 0.3)
    )
    t_ends = np.array([b for _, b in test])
    delays, held = [], 0
    for k, (_, e) in enumerate(ref):
        nxt = ref[k + 1][0] if k + 1 < len(ref) else seconds
        later = t_ends[t_ends >= e - 0.55]
        if len(later) and later[0] <= max(nxt, e) + 1e-6:
            delays.append(float(later[0] - e))
        elif k + 1 < len(ref):
            held += 1
    return {
        "speech_s": float(fr.sum() * GRID),
        "silence_s": float((~fr).sum() * GRID),
        "missed_s": float((fr & ~ft).sum() * GRID),
        "false_s": float((ft & ~fr).sum() * GRID),
        "false_starts": false_starts,
        "turns": len(ref),
        "held": held,
        "delays": delays,
    }


def load(root: Path, run: str) -> tuple[dict, dict]:
    d = root / run
    callers = json.loads((d / "callers.json").read_text())
    files = {
        c["room"].split("-")[-1]: (Path(c["file"]), c["seconds"])
        for c in callers["calls"]
    }
    logs = {}
    for p in d.glob(f"{run}-*.json"):
        logs[p.stem.split("-")[-1]] = json.loads(p.read_text())
    return logs, files


def summarize(root: Path, ref_run: str, run: str) -> dict:
    logs, files = load(root, run)
    ref_logs, ref_files = load(root, ref_run)
    dts, filt, job, drift = [], [], [], []
    vad = []
    for i, log in logs.items():
        fr = np.asarray(log["frames"], dtype=np.float64)
        if not len(fr) or i not in files:
            continue
        seconds = files[i][1]
        dts.append(fr[:, 2] * 1000)
        filt.append(fr[:, 2].sum() / seconds)
        job.append(log["cpu_s"] / log["seconds"])
        lag = fr[:, 0] - fr[:, 1] / SR
        drift.append(float(lag.max() - lag.min()))
        if i in ref_logs:
            test_iv = turns(log, files[i][0])
            ref_iv = turns(ref_logs[i], ref_files[i][0])
            vad.append(against(test_iv, ref_iv, seconds))
    dt = np.concatenate(dts) if dts else np.zeros(1)
    out = {
        "run": run,
        "calls": len(filt),
        "frame_ms_p50": float(np.percentile(dt, 50)),
        "frame_ms_p99": float(np.percentile(dt, 99)),
        "frame_ms_max": float(dt.max()),
        "filter_core_per_call": float(np.mean(filt)) if filt else None,
        "job_core_per_call": float(np.mean(job)) if job else None,
        "drift_s_p99": float(np.percentile(drift, 99)) if drift else None,
    }
    if vad:
        minutes = sum(v["speech_s"] + v["silence_s"] for v in vad) / 60
        dl = np.concatenate([v["delays"] for v in vad if v["delays"]] or [np.zeros(0)])
        out.update(
            false_starts_per_min=sum(v["false_starts"] for v in vad) / minutes,
            pauses_heard_pct=100
            * sum(v["false_s"] for v in vad)
            / sum(v["silence_s"] for v in vad),
            caller_missed_pct=100
            * sum(v["missed_s"] for v in vad)
            / sum(v["speech_s"] for v in vad),
            turns=sum(v["turns"] for v in vad),
            held_pct=100
            * sum(v["held"] for v in vad)
            / max(1, sum(v["turns"] for v in vad)),
            turn_end_p90_s=float(np.percentile(dl, 90)) if len(dl) else None,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/vi_bench")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    rows = [summarize(root, a.reference, r) for r in a.runs]
    (root / "report.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(
            json.dumps(
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
            )
        )


if __name__ == "__main__":
    main()
