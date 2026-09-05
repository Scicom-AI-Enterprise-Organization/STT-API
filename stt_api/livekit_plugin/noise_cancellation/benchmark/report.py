"""
Rendering. Three tables, because they answer three different questions and
collapsing them into one ranking would be dishonest — the best-sounding model
here is not the cheapest, and neither is necessarily the best for WER.

Everything is shown as an absolute value plus a delta against `passthrough`.
The absolute numbers are what compare against published results; the deltas are
what tell you whether a model earns its place, and they are the reason
`passthrough` is always run. A row that cannot beat doing nothing is worse than
no noise cancellation, and that has to be visible at a glance rather than
inferred by the reader.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict

from .harness import Summary

__all__ = ["format_report", "to_json"]

_BASE = "passthrough"


def _f(v: float | None, spec: str = ".3f", dash: str = "  -") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return dash
    return format(v, spec)


_NA = "n/a"


def _wave_metric(s: Summary, key: str, spec: str) -> str:
    """
    Waveform-comparison metrics, blanked for generative restorers.

    PESQ, ESTOI and SI-SDR all compare the output waveform against the clean
    reference sample by sample. A vocoder emits a *different* waveform that sounds
    like the same speech, so these score it near the bottom however good it sounds
    — SI-SDR in particular goes sharply negative for output a listener would
    prefer. Printing that number next to an acoustic denoiser's invites exactly
    the wrong conclusion, so this class gets `n/a` and is judged on DNSMOS and WER
    instead. The raw values are still written to `--json` for anyone who wants
    to see how badly the mismatch bites.
    """
    if s.generative:
        return _NA
    return _f(s.quality.get(key), spec)


def _delta(v: float | None, base: float | None, spec: str = "+.2f") -> str:
    if v is None or base is None:
        return ""
    if not (math.isfinite(v) and math.isfinite(base)):
        return ""
    return format(v - base, spec)


def _table(rows: list[list[str]], headers: list[str], align: str) -> list[str]:
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        out = []
        for i, c in enumerate(cells):
            out.append(c.ljust(widths[i]) if align[i] == "l" else c.rjust(widths[i]))
        return "  ".join(out).rstrip()

    return [line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in rows)]


def format_report(summaries: list[Summary], *, frame_ms: int, corpus: str, items: int) -> str:
    if not summaries:
        return "no results"
    order = {s.enhancer: s for s in summaries}
    base = order.get(_BASE)
    out: list[str] = []

    total_audio = max((s.audio_seconds for s in summaries), default=0.0)
    out.append(
        f"corpus {corpus} · {items} items · {total_audio / 60:.1f} min audio · "
        f"{frame_ms} ms frames"
    )
    out.append("")

    # ---- quality -----------------------------------------------------------
    out.append("QUALITY  (vs clean reference; delta is against passthrough)")
    rows = []
    for s in summaries:
        q, d = s.quality, s.dnsmos
        gen = s.generative
        rows.append(
            [
                s.enhancer + ("" if s.streaming else " *"),
                _wave_metric(s, "pesq", ".3f"),
                "" if gen else _delta(q.get("pesq"), base.quality.get("pesq") if base else None),
                _wave_metric(s, "estoi", ".3f"),
                "" if gen else _delta(q.get("estoi"), base.quality.get("estoi") if base else None, "+.3f"),
                _wave_metric(s, "si_sdr", ".1f"),
                "" if gen else _delta(q.get("si_sdr"), base.quality.get("si_sdr") if base else None, "+.1f"),
                _f(d.get("sig"), ".2f"),
                _f(d.get("bak"), ".2f"),
                _f(d.get("ovrl"), ".2f"),
                _delta(d.get("ovrl"), base.dnsmos.get("ovrl") if base else None),
            ]
        )
    out += _table(
        rows,
        ["model", "PESQ", "Δ", "ESTOI", "Δ", "SI-SDR", "Δ", "SIG", "BAK", "OVRL", "Δ"],
        "lrrrrrrrrrr",
    )
    out.append("")

    # ---- realtime cost -----------------------------------------------------
    out.append("COST  (per frame, one core; p99 is what breaks calls, not the mean)")
    rows = []
    for s in summaries:
        f = s.frame_ms
        rows.append(
            [
                s.enhancer + ("" if s.streaming else " *"),
                _f(s.rtf, ".4f"),
                _f(f.get("p50"), ".2f"),
                _f(f.get("p95"), ".2f"),
                _f(f.get("p99"), ".2f"),
                _f(f.get("max"), ".2f"),
                _f(f.get("budget_p99"), ".1%") if f else "  -",
                _f(s.delay_ms, ".1f"),
                _f(s.noise_floor_db, "+.1f"),
                _f(s.level_change_db, "+.2f"),
            ]
        )
    out += _table(
        rows,
        ["model", "RTF", "p50ms", "p95ms", "p99ms", "maxms", "budget", "delay", "floor", "level"],
        "lrrrrrrrrr",
    )
    out.append("")

    # ---- turn-taking -------------------------------------------------------
    if any(s.crosstalk for s in summaries):
        out.append(
            "CROSSTALK  (competing speaker; selectivity is the honest column — "
            "suppression alone can just be turning everything down)"
        )
        rows = []
        for s in summaries:
            c = s.crosstalk
            rows.append(
                [
                    s.enhancer + ("" if s.streaming else " *"),
                    _f(c.get("suppression_db"), "+.1f"),
                    _f(c.get("target_db"), "+.1f"),
                    _f(c.get("selectivity_db"), "+.1f"),
                    _f(c.get("vad_false_trigger_before"), ".0%"),
                    _f(c.get("vad_false_trigger"), ".0%"),
                ]
            )
        out += _table(
            rows,
            ["model", "supp dB", "tgt dB", "SELECT", "VAD before", "VAD after"],
            "lrrrrr",
        )
        out.append("")

    # ---- downstream --------------------------------------------------------
    if any(s.wer is not None for s in summaries):
        out.append("DOWNSTREAM  (WER against the STT's own transcript of the clean signal)")
        rows = []
        for s in summaries:
            rows.append(
                [
                    s.enhancer + ("" if s.streaming else " *"),
                    _f(s.wer, ".2%") if s.wer is not None else "  -",
                    _delta(s.wer, base.wer if base else None, "+.2%")
                    if s.wer is not None
                    else "",
                    _f(s.wer_median, ".2%") if s.wer_median is not None else "  -",
                    str(s.asr_loops) if s.wer is not None else "-",
                ]
            )
        out += _table(rows, ["model", "WER", "Δ", "median", "loops"], "lrrrr")
        out.append("")
        if any(s.asr_loops for s in summaries):
            out.append(
                "  loops = ASR repetition-loop hypotheses, excluded from the pooled WER "
                "and counted here. Pooled WER is unbounded under insertions, so one "
                "400-word loop against a 20-word reference can outweigh a whole corpus; "
                "a large gap between WER and median means the pooled figure is being "
                "driven by a few items, not by the model."
            )
            out.append("")

    if any(not s.streaming for s in summaries):
        out.append(
            "* offline: whole-file with unlimited lookahead. A quality ceiling, not a "
            "deployable option. RTF and the per-frame percentiles are blank because "
            "there are no frames; the delay shown is the model's own algorithmic "
            "lookahead and excludes the buffering a streaming version would add."
        )
    if any(s.generative for s in summaries):
        out.append(
            "n/a: generative restorers resynthesise the waveform, so PESQ/ESTOI/SI-SDR "
            "compare against a signal that no longer exists sample-for-sample and rank "
            "them near the bottom however good they sound. Judge this class on DNSMOS "
            "and WER. None of them is a LiveKit inline filter: they need a GPU, are not "
            "causal, and their chunk latency is hundreds of milliseconds."
        )
    failed = [s for s in summaries if s.failed]
    if failed:
        out.append("")
        for s in failed:
            out.append(f"! {s.enhancer}: {s.failed} item(s) failed — {'; '.join(s.errors)}")
    return "\n".join(out)


def to_json(summaries: list[Summary], *, frame_ms: int, corpus: str, items: int) -> str:
    return json.dumps(
        {
            "corpus": corpus,
            "items": items,
            "frame_ms": frame_ms,
            "results": [asdict(s) for s in summaries],
        },
        indent=2,
        default=float,
    )
