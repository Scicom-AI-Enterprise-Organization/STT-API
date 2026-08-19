"""The guard: every LLM edit must be explainable, or it is thrown away.

An LLM asked to "normalize spelling" will occasionally fix grammar, finish a
truncated sentence, or swap a word it thinks was misheard. Each of those changes
what is being measured — silently, and in the direction that flatters the score.

`validate()` walks the original and the candidate token by token and demands that
every token of the candidate be derivable from the original by one of four moves:

    identity            case/punctuation-insensitive
    whitelisted respell `lah` -> `la`, `okay` -> `ok`
    letter-preserving   a join (`peduli kan` -> `pedulikan`) or its inverse split
    number rewrite      value-preserving (`dua puluh tiga` -> `23`)

Anything else — an inserted word, a deleted one, a substituted content word — is
returned as a violation and the caller keeps the original text.

⚠ Rejection is the NORMAL case, not an error. 13% (723/5,700) of edits were
rejected in a full production run. A run that rejects nothing is more suspicious
than one that rejects a lot.
"""

from __future__ import annotations

from itertools import product

from .deterministic import NUMWORDS, RESPELL, SINGLE_DIGIT, compose, strip_nonspeech
from .metrics import normalize

__all__ = ["INVERSE", "validate"]


def _num_equiv(orig: list[str], new: str) -> bool:
    """True when `new` is a value-preserving rewrite of the `orig` tokens."""
    if new == compose(orig):
        return True
    if all(t in SINGLE_DIGIT for t in orig):
        if new == "".join(str(SINGLE_DIGIT[t]) for t in orig):
            return True
    return len(orig) == 1 and orig[0] in NUMWORDS and compose(orig) == new


# canonical -> the variants that fold onto it, for undoing a respell during a split
INVERSE: dict[str, list[str]] = {}
for _v, _c in RESPELL.items():
    INVERSE.setdefault(_c, []).append(_v)


def _split_of(span: list[str], word: str) -> bool:
    """Is `span` a letter-preserving split of `word`, allowing respelled pieces?

    `baguslah` -> `bagus la` reads as letter-losing (`bagusla`) until you notice
    the model also applied the lah->la respell. Each piece is therefore
    re-expanded to its variants before comparing. Bounded to 3 pieces so the
    product stays tiny.
    """
    opts = [[t] + INVERSE.get(t, []) for t in span]
    total = 1
    for o in opts:
        total *= len(o)
    if total > 64:
        return "".join(span) == word
    return any("".join(c) == word for c in product(*opts))


def validate(orig: str, norm: str, max_join: int = 8) -> list[str]:
    """Return the violations in `norm` relative to `orig`. Empty list = accept.

    Allowed: identity (case/punct-insensitive), a whitelisted respell, a join of
    adjacent tokens whose letters concatenate unchanged, or a value-preserving
    number rewrite. Everything else — a substituted content word, an inserted
    word, a deleted one — is reported, and the caller keeps the original text.
    """
    # Non-speech tags are stripped from the ORIGINAL first: deleting `[laugh]` is
    # a sanctioned edit (deterministic_normalize does it too), so comparing
    # against the un-stripped text would report the model's correct removal as a
    # deleted word.
    ow, nw = normalize(strip_nonspeech(orig)), normalize(strip_nonspeech(norm))
    violations: list[str] = []
    i = j = 0
    while j < len(nw):
        if i >= len(ow):
            violations.append(f"inserted {' '.join(nw[j:j+3])!r}")
            break
        o, n = ow[i], nw[j]
        if o == n or RESPELL.get(o) == n:
            i += 1
            j += 1
            continue
        matched = False
        for k in range(1, max_join + 1):      # join, or multi-word number
            span = ow[i:i + k]
            if not span:
                break
            if "".join(span) == n or _num_equiv(span, n):
                i += k
                j += 1
                matched = True
                break
        if matched:
            continue
        # ...and the mirror image: a letter-preserving SPLIT (`baguslah` ->
        # `bagus la`). This was 25 of 25 rejections on the first real run — an
        # affix the model detached, which is as safe as the join it is the
        # inverse of, and is exactly the ref/hyp disagreement the pass exists to
        # remove.
        for m in range(2, 4):
            span = nw[j:j + m]
            if len(span) < m:
                break
            if _split_of(span, o):
                i += 1
                j += m
                matched = True
                break
        if matched:
            continue
        violations.append(f"{o!r} -> {n!r} is not a whitelisted edit")
        i += 1
        j += 1
    if not violations and i < len(ow):
        violations.append(f"deleted {' '.join(ow[i:i+3])!r}")
    return violations
