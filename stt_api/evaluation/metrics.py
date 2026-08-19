"""Tokenizer, edit distance and corpus-level WER/CER.

This is the scoring core, deliberately free of every other layer: no LLM, no
network, no optional dependency. Import it on its own when all you want is a
number.

It is a byte-for-byte copy of the tokenizer + edit distance used by the STT
benchmark harness (`evaluation/stt/evaluate_stt.py`). If you change one, change
both, or the two stop being comparable.

Two properties worth knowing before you quote anything it returns:

- **Corpus WER is pooled, not averaged.** Total edit distance over total
  reference length — long clips weigh more, and a one-word clip that is wholly
  wrong contributes one error, not a 100% rate.
- **Mandarin is a caveat, not a metric.** The word split is whitespace-based, so
  a Chinese utterance is roughly one token. Read CER for `zh`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "Metrics",
    "corpus_metrics",
    "fold_canonical",
    "levenshtein",
    "normalize",
    "score_one",
    "wer_cer",
]

_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)


def normalize(text: str) -> list[str]:
    """NFKC + lowercase + strip punctuation -> tokens. Digits stay digits."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT_RE.sub(" ", text).replace("_", " ")
    return [t for t in (tok.strip("'") for tok in text.split()) if t]


def fold_canonical(tokens: Sequence[str], variant_map: Mapping[str, str] | None) -> list[str]:
    """Fold each token through a dataset's `variant -> canonical` map.

    Applied to BOTH sides before any distance is computed, which is what makes a
    dataset's own declared equivalences (`ramlee`/`ramli`, `card`/`kad`) cost
    nothing. A hand-declared map is the only defensible way to do this: phonetic
    distance cannot tell `fallujah`/`faluyah` (same city, misspelt) from
    `bordentown`/`bordertown` (different cities).
    """
    if not variant_map:
        return list(tokens)
    return [variant_map.get(t, t) for t in tokens]


def levenshtein(ref: Sequence, hyp: Sequence) -> int:
    """Edit distance over any sequence — token lists for WER, characters for CER."""
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class Metrics:
    """Pooled counts, so slices can be summed without re-scoring.

    Rates are derived, never stored: adding two `Metrics` gives the correct
    corpus rate for the union, which averaging two rates would not.
    """

    word_dist: int = 0
    ref_words: int = 0
    char_dist: int = 0
    ref_chars: int = 0

    @property
    def wer(self) -> float:
        return self.word_dist / max(self.ref_words, 1)

    @property
    def cer(self) -> float:
        return self.char_dist / max(self.ref_chars, 1)

    def __add__(self, other: "Metrics") -> "Metrics":
        return Metrics(
            self.word_dist + other.word_dist,
            self.ref_words + other.ref_words,
            self.char_dist + other.char_dist,
            self.ref_chars + other.ref_chars,
        )

    def as_dict(self) -> dict:
        return {
            "wer": self.wer,
            "cer": self.cer,
            "word_dist": self.word_dist,
            "ref_words": self.ref_words,
            "char_dist": self.char_dist,
            "ref_chars": self.ref_chars,
        }


def score_one(ref: str, hyp: str, variant_map: Mapping[str, str] | None = None) -> Metrics:
    """Score a single ref/hyp pair into poolable counts."""
    rt = fold_canonical(normalize(ref), variant_map)
    ht = fold_canonical(normalize(hyp), variant_map)
    rs, hs = " ".join(rt), " ".join(ht)
    return Metrics(levenshtein(rt, ht), len(rt), levenshtein(list(rs), list(hs)), len(rs))


def corpus_metrics(items: Iterable[Sequence]) -> Metrics:
    """Pool `(ref, hyp)` or `(ref, hyp, variant_map)` triples into one `Metrics`."""
    total = Metrics()
    for item in items:
        ref, hyp = item[0], item[1]
        vmap = item[2] if len(item) > 2 else None
        total = total + score_one(ref, hyp, vmap)
    return total


def wer_cer(items: Iterable[Sequence]) -> tuple[float, float]:
    """`(wer, cer)` for a corpus — the harness's signature, kept for parity."""
    m = corpus_metrics(items)
    return m.wer, m.cer
