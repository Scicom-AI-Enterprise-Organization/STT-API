"""The one-call API: `score(hypothesis, reference)`.

    from stt_api.evaluation import score

    r = score("saya bayar 23 ringgit", "saya bayar dua puluh tiga ringgit")
    r.wer                       # [0.5]   as scored, per item
    r.normalized_wer            # [0.0]   the whole error was how a number was written
    r.normalized_hypothesis     # ['saya bayar 23 ringgit']
    r.normalized_reference      # ['saya bayar 23 ringgit']

Lists in, lists out — and a single string pair is just a list of one, so the
shape of the result never depends on the shape of the input.

    r = score(["okay lah, saya nak bayar", "nombor 012"],
              ["ok la saya nak bayar",     "nombor kosong satu dua"])
    r.wer                       # [0.4, 0.75]
    r.corpus_wer                # 0.5556 — pooled, which is NOT the mean of r.wer

⚠ ARGUMENT ORDER IS (hypothesis, reference). WER is not symmetric — swapping
them gives a different, wrong number without raising. If you are ever unsure,
pass them by keyword: `score(hypothesis=..., reference=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .llm import LLMClient
from .metrics import Metrics
from .normalizer import Normalizer
from .score import Pair, ScoreReport, score_pairs

__all__ = ["ScoreResult", "score"]


@dataclass
class ScoreResult:
    """Per-item lists plus the pooled corpus figures.

    ⚠ Per-item WER is a rate, so it sorts SHORT items to the top — one word, one
    mismatch, 100%. When you are deciding what to FIX, rank by `word_dist` in
    `metrics` (error mass), not by `wer`. When you are REPORTING, quote
    `corpus_wer`, which is total distance over total reference length, never the
    mean of `wer`.
    """

    wer: list[float] = field(default_factory=list)
    cer: list[float] = field(default_factory=list)
    normalized_wer: list[float] = field(default_factory=list)
    normalized_cer: list[float] = field(default_factory=list)
    normalized_hypothesis: list[str] = field(default_factory=list)
    normalized_reference: list[str] = field(default_factory=list)
    metrics: list[Metrics] = field(default_factory=list)
    normalized_metrics: list[Metrics] = field(default_factory=list)
    corpus: Metrics = field(default_factory=Metrics)
    normalized_corpus: Metrics = field(default_factory=Metrics)
    report: ScoreReport | None = None

    @property
    def corpus_wer(self) -> float:
        return self.corpus.wer

    @property
    def corpus_cer(self) -> float:
        return self.corpus.cer

    @property
    def normalized_corpus_wer(self) -> float:
        return self.normalized_corpus.wer

    @property
    def normalized_corpus_cer(self) -> float:
        return self.normalized_corpus.cer

    @property
    def recovered_wer(self) -> float:
        """Corpus WER points that were spelling convention, not recognition error."""
        return self.corpus.wer - self.normalized_corpus.wer

    def __len__(self) -> int:
        return len(self.wer)

    def as_dict(self) -> dict:
        return {
            "wer": self.wer,
            "cer": self.cer,
            "normalized_wer": self.normalized_wer,
            "normalized_cer": self.normalized_cer,
            "normalized_hypothesis": self.normalized_hypothesis,
            "normalized_reference": self.normalized_reference,
            "corpus": self.corpus.as_dict(),
            "normalized_corpus": self.normalized_corpus.as_dict(),
            "recovered_wer": self.recovered_wer,
        }


def _as_list(x: Any, name: str) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, Iterable):
        return [str(i) if i is not None else "" for i in x]
    raise TypeError(f"{name} must be a string or an iterable of strings, got {type(x).__name__}")


def score(
    hypothesis: str | Sequence[str],
    reference: str | Sequence[str],
    mode: str = "deterministic",
    normalizer: Normalizer | None = None,
    client: LLMClient | None = None,
    cache: Any = None,
    drop_fillers: bool = False,
    fillers: Iterable[str] | None = None,
    workers: int = 8,
    variant_maps: Sequence[Mapping[str, str]] | None = None,
    categories: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
) -> ScoreResult:
    """Score ASR output against ground truth, raw and convention-normalized.

    Args:
        hypothesis: the ASR output — one string, or a list of them.
        reference: the ground truth, same shape and order as `hypothesis`.
        mode: `deterministic` (default, offline and free), `llm`/`both` to add the
            LLM pass, `pair` for the leakage study only.
        normalizer: an existing `Normalizer`, to share one cache across several
            models. That is what keeps their shared references byte-identical.
        client: `LLMClient` for the LLM modes; `LLMClient.from_env()` reads
            OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME.
        drop_fillers: also delete fillers from both sides. Changes what is being
            measured — read the warning on `FILLERS` first.
        variant_maps: per-item `{variant: canonical}` equivalences, folded into
            both sides before measuring (see `load_canonical`).
        categories: per-item grouping key, for `result.report.per_category()`.

    Returns:
        `ScoreResult` — per-item `wer`/`cer` lists, the normalized text of both
        sides, and the pooled corpus figures. Always lists, even for one pair.
    """
    hyps = _as_list(hypothesis, "hypothesis")
    refs = _as_list(reference, "reference")
    if len(hyps) != len(refs):
        raise ValueError(
            f"hypothesis and reference must be the same length, got "
            f"{len(hyps)} and {len(refs)}"
        )
    if variant_maps is not None and len(variant_maps) != len(hyps):
        raise ValueError("variant_maps must be the same length as hypothesis")

    pairs = [
        Pair(
            ref=r,
            hyp=h,
            id=str(ids[i]) if ids is not None else str(i),
            category=categories[i] if categories is not None else None,
            variant_map=variant_maps[i] if variant_maps is not None else None,
        )
        for i, (h, r) in enumerate(zip(hyps, refs))
    ]
    report = score_pairs(pairs, mode=mode, normalizer=normalizer, client=client,
                         cache=cache, drop_fillers=drop_fillers, fillers=fillers,
                         workers=workers)
    return ScoreResult(
        wer=[row.raw.wer for row in report.rows],
        cer=[row.raw.cer for row in report.rows],
        normalized_wer=[row.normalized.wer for row in report.rows],
        normalized_cer=[row.normalized.cer for row in report.rows],
        normalized_hypothesis=[row.hyp_norm for row in report.rows],
        normalized_reference=[row.ref_norm for row in report.rows],
        metrics=[row.raw for row in report.rows],
        normalized_metrics=[row.normalized for row in report.rows],
        corpus=report.as_scored,
        normalized_corpus=report.normalized,
        report=report,
    )
