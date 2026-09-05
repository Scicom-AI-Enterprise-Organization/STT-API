"""Command line for the noise-cancellation shootout."""

from __future__ import annotations

import argparse
import sys
import time

from . import asr as asr_mod
from . import corpus as corpus_mod
from . import enhancers as enh_mod
from .harness import run_item, summarize
from .report import format_report, to_json

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m stt_api.livekit_plugin.noise_cancellation.benchmark",
        description="Compare noise-cancellation models under LiveKit's real streaming constraints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # quick look, 40 utterances, the cheap models\n"
            "  python -m ...benchmark --limit 40\n\n"
            "  # full standard benchmark, comparable to published PESQ/STOI\n"
            "  python -m ...benchmark --corpus voicebank --models all\n\n"
            "  # does it actually help the STT?\n"
            "  python -m ...benchmark --limit 100 --asr-url http://127.0.0.1:8000\n\n"
            "  # real phone-call-ish audio, no clean reference, DNSMOS only\n"
            "  python -m ...benchmark --corpus dns-real --limit 50\n"
        ),
    )
    p.add_argument(
        "--models",
        default=",".join(enh_mod.DEFAULT),
        help="comma-separated, or 'all'. available: " + ", ".join(sorted(enh_mod.REGISTRY)),
    )
    p.add_argument("--corpus", default="voicebank", choices=sorted(corpus_mod.SOURCES))
    p.add_argument("--limit", type=int, default=50, help="utterances (0 = all)")
    p.add_argument("--seed", type=int, default=0, help="subset selection seed")
    p.add_argument(
        "--frame-ms",
        type=int,
        default=20,
        help="frame size the SFU delivers; LiveKit's default is 50",
    )
    p.add_argument("--rate", type=int, default=16000, help="pipeline sample rate")
    p.add_argument(
        "--degrade",
        default=None,
        choices=["telephony", "crosstalk"],
        help="extra degradation on the input: 'telephony' = 8 kHz + G.711 mu-law, "
        "the condition inbound SIP audio arrives in; 'crosstalk' = a competing "
        "speaker, the condition that breaks VAD and end-of-turn detection",
    )
    p.add_argument(
        "--crosstalk-sir",
        type=float,
        default=5.0,
        help="dB of target-to-interferer ratio for --degrade crosstalk",
    )
    p.add_argument("--no-dnsmos", action="store_true", help="skip DNSMOS (faster)")
    p.add_argument(
        "--asr-url",
        default=None,
        help="enable the WER axis against this stt-api instance, e.g. http://127.0.0.1:8000",
    )
    p.add_argument("--asr-language", default=None, help="language hint for the STT")
    p.add_argument(
        "--asr-whisper",
        nargs="?",
        const="openai/whisper-large-v3",
        default=None,
        help="enable the WER axis with a LOCAL whisper instead of --asr-url. "
        "Required to say anything trustworthy about the generative restorers: "
        "DNSMOS cannot detect hallucinated speech, WER can",
    )
    p.add_argument("--json", dest="json_path", default=None, help="also write results as JSON")
    p.add_argument("--list", action="store_true", help="show which models can run here, then exit")
    p.add_argument("--quiet", action="store_true", help="suppress per-model progress")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        for name, why in enh_mod.available().items():
            print(f"  {name:14} {'ok' if why is None else why}")
        return 0

    names = (
        sorted(enh_mod.REGISTRY) if args.models == "all" else [n.strip() for n in args.models.split(",") if n.strip()]
    )
    # passthrough is the baseline every delta is measured against, so it is always
    # run even if not asked for; without it the table has numbers but no meaning.
    if "passthrough" not in names:
        names.insert(0, "passthrough")

    limit = None if args.limit in (0, None) else args.limit
    items = list(
        corpus_mod.load(
            args.corpus,
            limit,
            rate=args.rate,
            seed=args.seed,
            degrade=args.degrade,
            crosstalk_sir=args.crosstalk_sir,
        )
    )
    if not items:
        print(f"no items loaded from {args.corpus}", file=sys.stderr)
        return 1
    has_ref = any(i.clean is not None for i in items)

    dnsmos = None
    if not args.no_dnsmos:
        from .dnsmos import DNSMOS

        dnsmos = DNSMOS()

    client = None
    if args.asr_whisper:
        client = asr_mod.WhisperClient(
            model=args.asr_whisper, language=args.asr_language or "en"
        )
    elif args.asr_url:
        client = asr_mod.AsrClient(base_url=args.asr_url, language=args.asr_language)
    if client is not None:
        try:
            client.check()
        except Exception as e:  # noqa: BLE001
            print(f"WER axis disabled: {type(e).__name__}: {e}", file=sys.stderr)
            client = None
        if client and not has_ref:
            print(
                "WER axis disabled: this corpus has no clean signal to build a reference from",
                file=sys.stderr,
            )
            client = None

    # One clean-audio transcript per item, reused by every model. Doing it once
    # keeps the reference identical across the table and keeps a long run's STT
    # cost linear in items rather than in items x models.
    refs: dict[str, str] = {}
    if client:
        if not args.quiet:
            print(f"transcribing {len(items)} clean references ...", file=sys.stderr)
        for it in items:
            try:
                refs[it.id] = client.transcribe(it.clean, it.rate)
            except Exception as e:  # noqa: BLE001
                print(f"  ref {it.id}: {e}", file=sys.stderr)

    summaries = []
    for name in names:
        try:
            enh = enh_mod.build(name)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"skipping {name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        t0 = time.perf_counter()
        results, wer = [], None
        acc = asr_mod.WerAccumulator() if client else None
        for it in items:
            r = run_item(
                enh,
                it,
                frame_ms=args.frame_ms,
                dnsmos=dnsmos,
                keep_audio=acc is not None,
            )
            if acc is not None and r.error is None and r.enhanced is not None and it.id in refs:
                try:
                    acc.add(it.id, refs[it.id], client.transcribe(r.enhanced, it.rate))
                except Exception as e:  # noqa: BLE001
                    print(f"  {name}/{it.id}: {e}", file=sys.stderr)
            r.enhanced = None  # do not hold every utterance in memory
            results.append(r)
        if acc is not None and acc.ref_words:
            wer = acc.wer

        s = summarize(
            results,
            streaming=enh.streaming,
            generative=getattr(enh, "generative", False),
            frame_ms=args.frame_ms,
        )
        s.wer = wer
        if acc is not None:
            s.wer_median = acc.median_wer
            s.asr_loops = acc.n_degenerate
        summaries.append(s)
        if not args.quiet:
            print(
                f"  {name:14} {len(results):4d} items in {time.perf_counter() - t0:6.1f}s",
                file=sys.stderr,
            )

    text = format_report(
        summaries,
        frame_ms=args.frame_ms,
        corpus=args.corpus + (f"+{args.degrade}" if args.degrade else ""),
        items=len(items),
    )
    print(text)

    if args.json_path:
        with open(args.json_path, "w") as f:
            f.write(to_json(summaries, frame_ms=args.frame_ms, corpus=args.corpus, items=len(items)))
        print(f"\nwrote {args.json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
