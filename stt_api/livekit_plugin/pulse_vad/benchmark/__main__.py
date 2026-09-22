"""CLI: run both VADs over a corpus and print the comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import numpy as np

from .corpus import build_items, local_clips, parse_s3_uri, s3_clips
from .harness import load_pulse, load_silero, run_vad
from .metrics import (
    wilson,
    agreement,
    balanced_score,
    paired_timing,
    sweep,
    verify_replay,
)


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m stt_api.livekit_plugin.pulse_vad.benchmark",
        description="Compare PulseVAD against silero through the LiveKit VAD interface.",
    )
    src = p.add_argument_group("audio source")
    src.add_argument(
        "--audio", nargs="+", metavar="PATH", help="local files or directories"
    )
    src.add_argument("--s3-uri", metavar="S3://...", help="an s3:// object or prefix")
    src.add_argument("--s3-bucket")
    src.add_argument("--s3-prefix")
    src.add_argument("--env-file", help="defaults to the repo root .env")
    src.add_argument("--limit", type=int, help="cap the number of clips")

    cfg = p.add_argument_group("corpus")
    cfg.add_argument(
        "--conditions",
        nargs="+",
        default=["clean"],
        choices=["clean", "noisy", "telephony"],
    )
    cfg.add_argument("--snr-db", type=float, default=10.0)
    cfg.add_argument("--pad-seconds", type=float, default=1.5)
    cfg.add_argument(
        "--pad-kinds", nargs="+", default=["room"], choices=["silence", "white", "room"]
    )

    mdl = p.add_argument_group("model")
    mdl.add_argument("--model", default="2.1k", choices=["2.1k", "81k"])
    mdl.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    mdl.add_argument("--hop-ms", type=float, default=32.0)
    mdl.add_argument("--min-speech", type=float, default=0.05)
    mdl.add_argument("--min-silence", type=float, default=0.55)
    p.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    return p.parse_args(argv)


def _clips(a: argparse.Namespace):
    if a.s3_uri:
        bucket, key = parse_s3_uri(a.s3_uri)
        return s3_clips(bucket, key, limit=a.limit, env_file=a.env_file)
    if a.s3_bucket or a.s3_prefix:
        return s3_clips(a.s3_bucket, a.s3_prefix, limit=a.limit, env_file=a.env_file)
    if a.audio:
        return local_clips(a.audio, limit=a.limit)
    raise SystemExit("give --audio, --s3-uri, or --s3-bucket/--s3-prefix")


async def _run_all(vad, items, label):
    return [await run_vad(vad, it.audio, label=label) for it in items]


def _table(rows: list[list[str]], head: list[str]) -> str:
    w = [max(len(head[i]), *(len(r[i]) for r in rows)) for i in range(len(head))]
    out = [
        "  ".join(h.ljust(w[i]) for i, h in enumerate(head)),
        "  ".join("-" * x for x in w),
    ]
    out += ["  ".join(r[i].ljust(w[i]) for i in range(len(head))) for r in rows]
    return "\n".join(out)


def _isolated_cost(pulse, silero, n: int = 400) -> dict[str, float]:
    """
    Per-window cost with the asyncio machinery taken out.

    `VADEvent.inference_duration` brackets the whole loop iteration — frame
    combining, the int16->float32 convert, and a `run_in_executor` thread hop
    that costs tens of microseconds on its own. At this scale that overhead is
    comparable to the model, so the streamed figure flatters neither model
    honestly. This times just the callable each stream invokes.
    """
    out: dict[str, float] = {}
    rng = np.random.default_rng(0)

    win = (rng.standard_normal(3200) * 0.1).astype(np.float32)
    pulse._model(win)
    t = time.perf_counter()
    for _ in range(n):
        pulse._model(win)
    out["pulsevad_ms"] = (time.perf_counter() - t) / n * 1000

    try:
        from livekit.plugins.silero import onnx_model

        m = onnx_model.OnnxModel(onnx_session=silero._onnx_session, sample_rate=16000)
        buf = (rng.standard_normal(m.window_size_samples) * 0.1).astype(np.float32)
        m(buf)
        t = time.perf_counter()
        for _ in range(n):
            m(buf)
        # silero's window is 32 ms and PulseVAD's hop is 32 ms, so one call each
        # per 32 ms of audio: these are directly comparable.
        out["silero_ms"] = (time.perf_counter() - t) / n * 1000
    except Exception as e:  # noqa: BLE001 - private API, benchmark-only
        out["silero_ms"] = float("nan")
        out["silero_error"] = str(e)[:80]
    return out


def main(argv: list[str] | None = None) -> int:
    a = _args(argv)
    seg = {"min_speech": a.min_speech, "min_silence": a.min_silence}

    clips = _clips(a)
    labelled = sum(1 for c in clips if c.has_speech is not None)
    n_neg = sum(1 for c in clips if c.has_speech is False)
    items = build_items(
        clips,
        pad_seconds=a.pad_seconds,
        pad_kinds=tuple(a.pad_kinds),
        conditions=tuple(a.conditions),
        snr_db=a.snr_db,
    )
    n_pos_items = sum(1 for i in items if i.kind == "positive")
    n_neg_items = len(items) - n_pos_items
    total_audio = sum(i.duration for i in items)

    print(
        f"corpus : {len(clips)} clips ({labelled} transcript-labelled, {n_neg} no-speech)"
    )
    print(
        f"         -> {len(items)} items ({n_pos_items} positive, {n_neg_items} negative), "
        f"{total_audio:.0f}s"
    )
    print(f"         conditions={a.conditions} pads={a.pad_seconds}s {a.pad_kinds}\n")
    if n_neg_items == 0:
        print("  NOTE: no no-speech clips in this corpus, so the clip-level false-fire")
        print(
            "        rate cannot be measured. Pull from the labelled S3 prefix for it.\n"
        )

    silero = load_silero(
        min_speech_duration=a.min_speech, min_silence_duration=a.min_silence
    )
    pulse = load_pulse(
        model=a.model,
        precision=a.precision,
        hop_duration=a.hop_ms / 1000.0,
        min_speech_duration=a.min_speech,
        min_silence_duration=a.min_silence,
    )
    print(f"silero   : {silero.model}")
    print(f"pulsevad : {pulse.model}  ceiling={pulse.ceiling}\n")

    runs_s = asyncio.run(_run_all(silero, items, "silero"))
    runs_p = asyncio.run(_run_all(pulse, items, "pulsevad"))

    mismatch = [verify_replay(r, activation=0.35, **seg) for r in runs_p]
    off = sum(1 for got, real in mismatch if got != real)
    print(
        f"replay check: {len(mismatch) - off}/{len(mismatch)} items reproduce the live "
        f"stream's turn count"
        + ("" if off == 0 else "   <-- SUSPECT, sweep unreliable")
    )

    grid = np.round(np.arange(0.10, 0.96, 0.05), 2)
    sw_s = sweep(runs_s, items, grid, **seg)
    sw_p = sweep(runs_p, items, grid[grid < pulse.ceiling], **seg)
    best_s = max(sw_s, key=balanced_score)
    best_p = max(sw_p, key=balanced_score)

    print("\n=== clip-level, against the production transcript (real ground truth) ===")
    if n_neg_items:

        def ci(k: int, n: int) -> str:
            lo, hi = wilson(k, n)
            return f"{lo * 100:.1f}-{hi * 100:.1f}"

        print(
            _table(
                [
                    [
                        s.label,
                        f"{s.threshold:.2f}",
                        f"{s.pos_detect_rate * 100:.2f}",
                        ci(s.pos_found, s.pos_clips),
                        f"{s.neg_fire_rate * 100:.2f}",
                        ci(s.neg_fired, s.neg_clips),
                        f"{s.neg_fired}/{s.neg_clips}",
                    ]
                    for s in (best_s, best_p)
                ],
                [
                    "model",
                    "thr",
                    "turns found %",
                    "95% CI",
                    "false fire %",
                    "95% CI",
                    "fired",
                ],
            )
        )
        print(f"\n  n = {best_s.pos_clips} speech clips, {best_s.neg_clips} no-speech.")
        print("  Overlapping intervals mean the gap is not resolvable at this n -")
        print(
            "  one no-speech clip is worth " f"{100 / max(best_s.neg_clips, 1):.2f} pp."
        )
    else:
        print("  (skipped - corpus has no no-speech clips)")

    print("\n=== frame-level and behaviour ===")
    print(
        _table(
            [
                [
                    s.label,
                    f"{s.threshold:.2f}",
                    f"{s.pad_fp_rate*100:.2f}",
                    f"{s.hit_rate*100:.2f}",
                    f"{s.f1:.4f}",
                    str(s.spurious_turns),
                    f"{s.onset_ms:.0f}",
                    f"{s.onset_p90_ms:.0f}",
                    f"{s.release_ms:+.0f}",
                ]
                for s in (best_s, best_p)
            ],
            [
                "model",
                "thr",
                "pad fp %",
                "body hit %",
                "F1",
                "spur",
                "onset ms",
                "p90",
                "release ms",
            ],
        )
    )

    print("\n=== operating-point sweep (detect rate - false-fire rate) ===")
    for name, sw in (("silero", sw_s), ("pulsevad", sw_p)):
        print(
            f"  {name:9s} "
            + " ".join(f"{s.threshold:.2f}:{balanced_score(s):+.3f}" for s in sw)
        )

    pt = paired_timing(
        runs_s, runs_p, items, act_a=best_s.threshold, act_b=best_p.threshold, **seg
    )
    print("\n=== timing, paired per item (cancels each clip's own lead-in) ===")
    print(
        f"  onset   pulsevad - silero : {pt['onset_delta_median_ms']:+.0f} ms median, "
        f"{pt['onset_delta_p90_ms']:+.0f} ms p90   "
        f"(later on {pt['onset_delta_b_later_pct']:.0f} % of {pt['onset_delta_n']} turns)"
    )
    print(
        f"  release pulsevad - silero : {pt['release_delta_median_ms']:+.0f} ms median, "
        f"{pt['release_delta_p90_ms']:+.0f} ms p90"
    )

    ag = agreement(
        runs_s, runs_p, items, act_a=best_s.threshold, act_b=best_p.threshold, **seg
    )
    print("\n=== agreement with the incumbent (silero is not ground truth) ===")
    print(f"  frames agreeing   {ag['agreement']*100:.2f} %")
    print(f"  Cohen's kappa     {ag['kappa']:.4f}")
    print(f"  silero-only       {ag['a_only']*100:.2f} %  (pulsevad deaf here)")
    print(
        f"  pulsevad-only     {ag['b_only']*100:.2f} %  (pulsevad would interrupt here)"
    )

    cost = _isolated_cost(pulse, silero)
    inf_s = np.concatenate([r.inference_ms for r in runs_s if len(r.inference_ms)])
    inf_p = np.concatenate([r.inference_ms for r in runs_p if len(r.inference_ms)])
    print("\n=== cost per 32 ms of audio (RUN PINNED AND ALONE - see README) ===")
    print(
        _table(
            [
                [
                    "silero",
                    f"{cost['silero_ms']:.3f}",
                    f"{np.median(inf_s):.3f}",
                    f"{np.percentile(inf_s,99):.3f}",
                ],
                [
                    "pulsevad",
                    f"{cost['pulsevad_ms']:.3f}",
                    f"{np.median(inf_p):.3f}",
                    f"{np.percentile(inf_p,99):.3f}",
                ],
            ],
            ["model", "model only ms", "in-stream median", "in-stream p99"],
        )
    )
    print("  'model only' excludes the asyncio thread hop and frame plumbing that")
    print("  'in-stream' includes; at this scale that overhead rivals the model.")

    print("\n=== verdict ===")
    if n_neg_items:
        print(
            f"  turns found      : {best_p.pos_detect_rate*100:.2f} % vs silero "
            f"{best_s.pos_detect_rate*100:.2f} %  "
            f"({(best_p.pos_detect_rate-best_s.pos_detect_rate)*100:+.2f} pp)"
        )
        print(
            f"  false fires      : {best_p.neg_fire_rate*100:.2f} % vs silero "
            f"{best_s.neg_fire_rate*100:.2f} %  "
            f"({(best_p.neg_fire_rate-best_s.neg_fire_rate)*100:+.2f} pp)"
        )
    print(
        f"  onset vs silero  : {pt['onset_delta_median_ms']:+.0f} ms median "
        f"(paired; absolute onset on production clips is lead-in, not latency)"
    )
    print(
        f"  model cost       : {cost['pulsevad_ms']:.3f} ms vs silero "
        f"{cost['silero_ms']:.3f} ms per 32 ms window"
    )
    print("  weights on disk  : 12 KB (fp32) vs silero ~1.8 MB")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(
                {
                    "corpus": {
                        "clips": len(clips),
                        "items": len(items),
                        "seconds": total_audio,
                        "positive_items": n_pos_items,
                        "negative_items": n_neg_items,
                        "conditions": a.conditions,
                    },
                    "best": {
                        s.label: {k: v for k, v in vars(s).items() if k != "per_item"}
                        for s in (best_s, best_p)
                    },
                    "sweep": {
                        "silero": [
                            {k: v for k, v in vars(s).items() if k != "per_item"}
                            for s in sw_s
                        ],
                        "pulsevad": [
                            {k: v for k, v in vars(s).items() if k != "per_item"}
                            for s in sw_p
                        ],
                    },
                    "agreement": ag,
                    "paired_timing": pt,
                    "cost_ms": cost,
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
