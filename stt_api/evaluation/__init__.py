"""WER/CER scoring with spelling conventions normalized away.

A chunk of measured WER is not recognition error. It is the reference and the
hypothesis disagreeing about how to *write* the same utterance — `23` vs `dua
puluh tiga`, `herm` vs `hmm` vs `um`, `okay` vs `ok`, `[laugh]` vs nothing.
Scoring verbatim is the right default for tracking one model over time, but it
charges a real error for every convention mismatch. This package measures that,
reproducibly.

    from stt_api.evaluation import score

    r = score("saya bayar 23 ringgit", "saya bayar dua puluh tiga ringgit")
    r.wer                     # [0.5]   as scored
    r.normalized_wer          # [0.0]   the error was how a number was written
    r.normalized_hypothesis   # ['saya bayar 23 ringgit']

`score(hypothesis, reference)` takes one pair or two equal-length lists and always
returns lists. `score_pairs()` underneath it takes ids, categories and per-row
variant maps, and returns a full `ScoreReport`.

Layers, each able to veto the next:

  1. deterministic  numbers (Malay + English) -> digits, whitelisted filler
                    respells, non-speech tags dropped, NFKC/case/punctuation.
                    Free, offline, reproducible; ~90% of the recoverable gap.
  2. llm            only the residue the rules cannot settle. Opt-in.
  3. validation     every LLM edit must be explainable as identity, a whitelisted
                    respell, a letter-preserving join/split, or a value-preserving
                    number rewrite. Anything else is REJECTED and the original
                    text kept.

⚠ The LLM never sees the reference and the hypothesis together — see `llm.py`.
⚠ Both numbers are always reported. The normalized WER is a second reading of the
  same run, never a replacement for the headline.

See `README.md` for usage and `CLAUDE.md` for the invariants that make the number
mean something.
"""

from .canonical import load_canonical
from .deterministic import (
    FILLERS,
    RESPELL,
    deterministic_normalize,
    numbers_to_digits,
    strip_nonspeech,
)
from .llm import LLMClient, load_dotenv
from .loaders import load_pairs, load_rows
from .metrics import (
    Metrics,
    corpus_metrics,
    fold_canonical,
    levenshtein,
    normalize,
    score_one,
    wer_cer,
)
from .normalizer import Normalizer, Rejection
from .report import format_report
from .score import Pair, ScoredPair, ScoreReport, score_pairs
from .simple import ScoreResult, score
from .validation import validate

__all__ = [
    "FILLERS",
    "LLMClient",
    "Metrics",
    "Normalizer",
    "Pair",
    "RESPELL",
    "Rejection",
    "ScoreReport",
    "ScoreResult",
    "ScoredPair",
    "corpus_metrics",
    "deterministic_normalize",
    "fold_canonical",
    "format_report",
    "levenshtein",
    "load_canonical",
    "load_dotenv",
    "load_pairs",
    "load_rows",
    "normalize",
    "numbers_to_digits",
    "score",
    "score_one",
    "score_pairs",
    "strip_nonspeech",
    "validate",
    "wer_cer",
]
