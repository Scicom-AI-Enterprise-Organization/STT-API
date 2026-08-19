"""Command line front end: `python -m stt_api.evaluation`.

    python -m stt_api.evaluation --input pairs.csv                 # ref,hyp columns
    python -m stt_api.evaluation --input pairs.jsonl --ref-field reference --hyp-field text
    python -m stt_api.evaluation --ref "dua puluh tiga ringgit" --hyp "RM23"
    python -m stt_api.evaluation --self-test                       # offline unit checks
    python -m stt_api.evaluation --input pairs.csv --mode both \\
        --base-url https://api.example.com/v1 --api-key sk-... --model <id>

Everything it does is available as a library call; see README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .canonical import load_canonical
from .llm import LLMClient
from .loaders import load_pairs
from .normalizer import Normalizer
from .report import format_report
from .score import Pair, score_pairs
from .selftest import run as run_selftest

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m stt_api.evaluation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_argument_group("input (one of)")
    src.add_argument("--input", help="CSV / TSV / JSON / JSONL with a reference and a "
                                     "hypothesis column (see --ref-field/--hyp-field)")
    src.add_argument("--ref", help="single reference string (the ground truth)")
    src.add_argument("--hyp", help="single hypothesis string (the ASR output)")
    ap.add_argument("--ref-field", default="ref", help="reference column (default: ref)")
    ap.add_argument("--hyp-field", default="hyp", help="hypothesis column (default: hyp)")
    ap.add_argument("--category-field", default="category",
                    help="grouping column for the breakdown (default: category)")
    ap.add_argument("--mode", choices=("deterministic", "llm", "both", "pair"),
                    default="deterministic",
                    help="default 'deterministic': free, offline, and measured to capture "
                         "~90%% of the recoverable gap. 'both' adds the LLM pass")
    ap.add_argument("--shuffle-hyp", action="store_true",
                    help="LEAKAGE CONTROL for --mode pair: give each reference a DIFFERENT "
                         "clip's hypothesis as its partner. Convention gain is corpus-wide "
                         "so it survives; anything that only appears with the true partner "
                         "is the model converging the two texts, i.e. fake")
    ap.add_argument("--drop-fillers", action="store_true",
                    help="also delete fillers from both sides (changes what is measured)")
    ap.add_argument("--cache", default="llm_norm_cache.json",
                    help="text->normalization cache; shared across models on purpose")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--canonical", metavar="HF_DATASET",
                    help="apply this dataset's per-row `canonical` variant map, as the "
                         "official scorer does, so 'as scored' reproduces its headline "
                         "(e.g. Revolab/ASR-Benchmark-Public)")
    ap.add_argument("--canonical-config", default=None)
    ap.add_argument("--canonical-split", default="train")
    ap.add_argument("--top", type=int, default=10, help="examples to print (0 = none)")
    ap.add_argument("--out", help="write the full report to this JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="run the built-in unit checks (no network, no data needed)")
    llm = ap.add_argument_group("LLM endpoint (only for --mode llm/both/pair)")
    llm.add_argument("--env-file", help="read OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME "
                                        "from this .env (process env wins over it)")
    llm.add_argument("--base-url", help="OpenAI-compatible base URL ending in /v1 "
                                        "(env: OPENAI_BASE_URL)")
    llm.add_argument("--api-key", help="(env: OPENAI_API_KEY)")
    llm.add_argument("--model", help="model id (env: MODEL_NAME)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.self_test:
        failures = run_selftest()
        for f in failures:
            print(f"FAIL {f}")
        print("self-test:", "OK" if not failures else f"{len(failures)} failure(s)")
        return 1 if failures else 0

    if args.ref is not None or args.hyp is not None:
        if args.ref is None or args.hyp is None:
            ap.error("--ref and --hyp must be given together")
        pairs = [Pair(ref=args.ref, hyp=args.hyp, id="cli")]
    elif args.input:
        try:
            pairs = load_pairs(args.input, args.ref_field, args.hyp_field,
                               args.category_field, args.limit)
        except (FileNotFoundError, KeyError, ValueError) as e:
            sys.exit(str(e))
    else:
        ap.error("give --input, or --ref/--hyp (or --self-test)")
    if not pairs:
        sys.exit("no usable ref/hyp pairs found")

    client = LLMClient.from_env(env_file=args.env_file, base_url=args.base_url,
                                api_key=args.api_key, model=args.model)
    if args.mode != "deterministic" and not client.configured:
        sys.exit(f"--mode {args.mode} needs an endpoint: pass --base-url/--api-key/--model, "
                 f"--env-file, or set OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME")

    canon: dict[str, dict[str, str]] = {}
    if args.canonical:
        canon = load_canonical(args.canonical, args.canonical_config, args.canonical_split)
        hit = sum(1 for p in pairs if p.id in canon)
        print(f"canonical map: {len(canon)} rows loaded, {hit}/{len(pairs)} matched by id")
        if hit == 0:
            sys.exit("canonical map matched no ids — wrong dataset/split, or ids have drifted")

    if args.mode == "pair" and args.shuffle_hyp:
        print("LEAKAGE CONTROL: references paired with the WRONG clip's hypothesis")

    norm = Normalizer(args.mode, client=client,
                      cache=Path(args.cache) if args.cache else None,
                      drop_fillers=args.drop_fillers)
    report = score_pairs(pairs, normalizer=norm, workers=args.workers,
                         shuffle_hyp=args.shuffle_hyp, canonical=canon)
    norm.save()
    print(format_report(report, args.top, warn_no_canonical=False))
    if args.out:
        Path(args.out).write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=1) + "\n")
        print(f"\nWrote {args.out}")
    return 0
