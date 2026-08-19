"""The high-level entry point: pairs in, both numbers out.

    from stt_api.evaluation import score_pairs

    report = score_pairs([("dua puluh tiga ringgit", "RM23")])
    report.as_scored.wer      # what an ordinary scorer charges you
    report.normalized.wer     # what is left once spelling conventions are folded

`as_scored` is the headline and `normalized` is a second reading of the SAME run,
never a replacement for it. Quoting a normalized WER against a figure from an
ordinary scorer compares two different metrics.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .deterministic import deterministic_normalize
from .llm import LLMClient
from .metrics import Metrics, corpus_metrics, score_one
from .normalizer import Normalizer, Rejection

__all__ = ["Pair", "ScoredPair", "ScoreReport", "score_pairs"]


@dataclass
class Pair:
    """One reference/hypothesis pair.

    `ref` is the ground truth (what was actually said, as a human wrote it) and
    `hyp` is the ASR output being graded. Getting them the wrong way round does
    not raise — WER is not symmetric, so the number is simply wrong. If you are
    reading a results file, the reference is the dataset's text column.

    `variant_map` is the row's declared `variant -> canonical` equivalences, if
    the dataset ships any (see `canonical.load_canonical`). `category` is any
    grouping key you want a breakdown by.
    """

    ref: str
    hyp: str
    id: str = ""
    category: str | None = None
    variant_map: Mapping[str, str] | None = None

    @classmethod
    def coerce(cls, item: Any, index: int = 0, ref_field: str = "ref",
               hyp_field: str = "hyp") -> "Pair":
        """Accept a `Pair`, a `(ref, hyp)` tuple, or a dict/row."""
        if isinstance(item, cls):
            return item
        if isinstance(item, Mapping):
            return cls(
                ref=str(item[ref_field]),
                hyp=str(item.get(hyp_field) or ""),
                id=str(item.get("id", index)),
                category=item.get("category"),
                variant_map=item.get("variant_map") or item.get("vmap"),
            )
        if isinstance(item, Sequence) and not isinstance(item, str):
            ref, hyp = item[0], item[1]
            return cls(ref=str(ref), hyp=str(hyp), id=str(index))
        raise TypeError(f"cannot read a ref/hyp pair from {type(item).__name__}")


@dataclass
class ScoredPair:
    """A pair with both normalizations and both scores attached."""

    pair: Pair
    ref_norm: str
    hyp_norm: str
    raw: Metrics
    normalized: Metrics

    @property
    def recovered(self) -> float:
        """WER points this row gave back to normalization (may be 0)."""
        return self.raw.wer - self.normalized.wer

    def as_dict(self) -> dict:
        return {
            "id": self.pair.id,
            "category": self.pair.category,
            "ref": self.pair.ref,
            "hyp": self.pair.hyp,
            "ref_norm": self.ref_norm,
            "hyp_norm": self.hyp_norm,
            "raw": self.raw.as_dict(),
            "normalized": self.normalized.as_dict(),
        }


@dataclass
class ScoreReport:
    """Corpus totals, per-category breakdown, and every row.

    ⚠ When deciding what to FIX, rank by error mass (`word_dist`), never by
    per-clip WER. Per-clip WER sorts SHORT clips to the top — one word, one
    mismatch, 100% — so it surfaces convention artefacts and misrepresents the
    category they came from. `Metrics` carries the raw counts for exactly this.
    """

    rows: list[ScoredPair] = field(default_factory=list)
    mode: str = "deterministic"
    drop_fillers: bool = False
    canonical_applied: bool = False
    rejected: list[Rejection] = field(default_factory=list)
    errors: int = 0

    @property
    def as_scored(self) -> Metrics:
        return corpus_metrics([(r.pair.ref, r.pair.hyp, r.pair.variant_map) for r in self.rows])

    @property
    def normalized(self) -> Metrics:
        return corpus_metrics([(r.ref_norm, r.hyp_norm, r.pair.variant_map) for r in self.rows])

    @property
    def recovered_wer(self) -> float:
        return self.as_scored.wer - self.normalized.wer

    def per_category(self) -> dict[str, tuple[Metrics, Metrics]]:
        cats: dict[str, tuple[Metrics, Metrics]] = {}
        for r in self.rows:
            if not r.pair.category:
                continue
            a, b = cats.get(r.pair.category, (Metrics(), Metrics()))
            cats[r.pair.category] = (a + r.raw, b + r.normalized)
        return dict(sorted(cats.items()))

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "drop_fillers": self.drop_fillers,
            "canonical_applied": self.canonical_applied,
            "samples": len(self.rows),
            "as_scored": self.as_scored.as_dict(),
            "normalized": self.normalized.as_dict(),
            "recovered_wer": self.recovered_wer,
            "per_category": {
                c: {"as_scored": a.as_dict(), "normalized": b.as_dict()}
                for c, (a, b) in self.per_category().items()
            },
            "rejected": len(self.rejected),
            "llm_errors": self.errors,
            "rows": [r.as_dict() for r in self.rows],
        }


def _normalize_independently(pairs: list[Pair], norm: Normalizer, workers: int) -> list[tuple[str, str]]:
    """One normalization per UNIQUE text, both sides drawn from the same table.

    Deduplicating is not only cheaper — it is what guarantees a reference shared
    by several arms normalizes identically in all of them.
    """
    texts = sorted({p.ref for p in pairs} | {p.hyp for p in pairs})
    if norm.mode in ("llm", "both") and workers > 1:
        pending = [t for t in texts if not norm.cached(t)]
        if pending:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(norm._llm, pending))
    table = {t: norm(t) for t in texts}
    return [(table[p.ref], table[p.hyp]) for p in pairs]


def _normalize_pairwise(pairs: list[Pair], norm: Normalizer, workers: int,
                        shuffle_hyp: bool) -> list[tuple[str, str]]:
    """The leakage study. See `LLMClient.normalize_pair` before using the output."""
    n = len(pairs)
    if not shuffle_hyp:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            done = list(ex.map(lambda p: norm._pair(p.ref, p.hyp), pairs))
        return [(norm.rules_only(a), norm.rules_only(b)) for a, b in done]
    # Control: BOTH sides must see a stranger, or the hypothesis still gets the
    # true reference and leakage walks in through the side that wasn't shuffled.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ref_side = list(ex.map(lambda k: norm._pair(pairs[k].ref, pairs[(k + 1) % n].hyp), range(n)))
        hyp_side = list(ex.map(lambda k: norm._pair(pairs[k].hyp, pairs[(k + 1) % n].ref), range(n)))
    return [(norm.rules_only(a), norm.rules_only(h))
            for (a, _), (h, _) in zip(ref_side, hyp_side)]


def score_pairs(
    pairs: Iterable[Any],
    mode: str = "deterministic",
    normalizer: Normalizer | None = None,
    client: LLMClient | None = None,
    cache: Any = None,
    drop_fillers: bool = False,
    fillers: Iterable[str] | None = None,
    workers: int = 8,
    shuffle_hyp: bool = False,
    canonical: Mapping[str, Mapping[str, str]] | None = None,
) -> ScoreReport:
    """Score ref/hyp pairs twice: as normally scored, and with conventions folded.

    `pairs` accepts `Pair` objects, `(ref, hyp)` tuples, or dicts with `ref`/`hyp`
    keys. Pass an existing `normalizer` to share one cache across several models —
    that is what keeps their shared references byte-identical.

    `canonical` is `{row_id: {variant: canonical}}`; rows are matched by `Pair.id`.
    """
    items = [Pair.coerce(p, i) for i, p in enumerate(pairs)]
    if not items:
        return ScoreReport(mode=mode, drop_fillers=drop_fillers)
    if canonical:
        for p in items:
            if p.variant_map is None:
                p.variant_map = canonical.get(p.id, {})

    norm = normalizer or Normalizer(mode, client=client, cache=cache,
                                    drop_fillers=drop_fillers, fillers=fillers)
    if norm.mode == "pair":
        normed = _normalize_pairwise(items, norm, workers, shuffle_hyp)
    else:
        normed = _normalize_independently(items, norm, workers)

    rows = [
        ScoredPair(
            pair=p,
            ref_norm=rn,
            hyp_norm=hn,
            raw=score_one(p.ref, p.hyp, p.variant_map),
            # The variant map is applied AFTER normalizing too, so this layer only
            # ever adds folding on top of the dataset's. Dropping it here made the
            # normalized score WORSE than the raw one on half the categories: a
            # declared map folds content variants (ini/ni/nih, kerana/karena) that
            # no general rule covers, and the respells compose with it rather than
            # replacing it.
            normalized=score_one(rn, hn, p.variant_map),
        )
        for p, (rn, hn) in zip(items, normed)
    ]
    return ScoreReport(
        rows=rows,
        mode=norm.mode,
        drop_fillers=norm.drop_fillers,
        canonical_applied=bool(canonical),
        rejected=norm.rejected,
        errors=norm.errors,
    )


def normalize_text(text: str, drop_fillers: bool = False) -> str:
    """Convenience: the deterministic layer on one string, no client needed."""
    return deterministic_normalize(text, drop_fillers)
