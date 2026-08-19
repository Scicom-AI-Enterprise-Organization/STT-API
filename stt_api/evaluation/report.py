"""Text rendering for a `ScoreReport`. Separate from scoring so a caller can take
the numbers and print nothing, or print their own.
"""

from __future__ import annotations

from .score import ScoreReport

__all__ = ["format_report"]


def format_report(report: ScoreReport, top: int = 10, warn_no_canonical: bool = False) -> str:
    """Both readings side by side, a per-category breakdown, and examples."""
    out: list[str] = []
    raw, new = report.as_scored, report.normalized

    out.append(f"\nsamples {len(report.rows)}   mode={report.mode}"
               f"{'  drop-fillers' if report.drop_fillers else ''}")
    if warn_no_canonical:
        out.append("⚠ this dataset ships a `canonical` variant map and it was NOT applied:\n"
                   "  'as scored' reads higher than the official figure, so 'recovered' is an\n"
                   "  upper bound rather than the marginal gain.")
    out.append(f"{'':14s} {'WER%':>8} {'CER%':>8}")
    out.append(f"{'as scored':14s} {raw.wer*100:8.2f} {raw.cer*100:8.2f}")
    out.append(f"{'normalized':14s} {new.wer*100:8.2f} {new.cer*100:8.2f}")
    share = (raw.wer - new.wer) / max(raw.wer, 1e-9) * 100
    out.append(f"{'recovered':14s} {(raw.wer-new.wer)*100:+8.2f} {(raw.cer-new.cer)*100:+8.2f}"
               f"   ({share:.0f}% of the WER was convention)")

    cats = report.per_category()
    if cats:
        out.append(f"\n{'category':18s} {'as scored':>10} {'normalized':>11} {'Δ':>8}")
        for c, (a, b) in cats.items():
            out.append(f"{c:18s} {a.wer*100:10.2f} {b.wer*100:11.2f} {(b.wer-a.wer)*100:+8.2f}")

    if top:
        gains = sorted((r for r in report.rows if r.recovered > 1e-9),
                       key=lambda r: -r.recovered)
        if gains:
            out.append(f"\ntop {min(top, len(gains))} convention-only recoveries "
                       f"({len(gains)} rows changed):")
            for r in gains[:top]:
                out.append(f"  -{r.recovered*100:5.1f} pp  {r.pair.id}")
                out.append(f"      ref  {r.pair.ref[:95]}")
                out.append(f"      ref* {r.ref_norm[:95]}")
                out.append(f"      hyp  {r.pair.hyp[:95]}")
                out.append(f"      hyp* {r.hyp_norm[:95]}")

    if report.rejected:
        out.append(f"\n⚠ {len(report.rejected)} LLM edit(s) REJECTED by validation "
                   f"(original kept — these did not affect the score):")
        for rej in report.rejected[:top or 5]:
            out.append(f"  {rej.violations[0]}")
            out.append(f"      in  {rej.original[:95]}")
            out.append(f"      out {rej.candidate[:95]}")
    if report.errors:
        out.append(f"\n⚠ {report.errors} LLM call(s) failed; those texts were left unnormalized.")
    return "\n".join(out)
