"""The free normalization layer: rules only — no network, no key, no model.

Spelled-out numbers become digits, hesitation fillers collapse onto one spelling,
non-speech annotations disappear, and case/punctuation/NFKC are folded away. On
Malaysian call-centre audio this captures roughly 90% of the recoverable
convention gap between a reference and a hypothesis, which is why it is the
default mode and the LLM pass is opt-in.

Everything here is a pure function of one text. Nothing in this module ever sees
both sides of a pair — that is the invariant the whole measurement rests on.
"""

from __future__ import annotations

import re
from typing import Iterable

from .metrics import normalize

__all__ = [
    "FILLERS",
    "RESPELL",
    "deterministic_normalize",
    "numbers_to_digits",
    "strip_nonspeech",
]

# ------------------------------------------------------------------ respells --
# Hesitation fillers have no agreed spelling; every transcriber picks one. Folding
# them onto a single form is convention normalization, not content editing.
RESPELL: dict[str, str] = {}
for _variants, _canon in (
    (("lah", "laa", "laaa"), "la"),
    (("ye", "yah", "yer", "yeah"), "ya"),
    (("aa", "aaa", "ah"), "ah"),
    (("okay", "oke", "okey", "okie", "ok"), "ok"),
    (("emm", "em", "err", "erm", "hmm", "hmmm", "umm", "um", "mmm",
      "uh", "uhh", "uhm", "ermm"), "herm"),
    (("takde", "xde"), "tiada"),
    (("x",), "tak"),
    (("orait", "oraite", "alrite"), "alright"),
):
    for _v in _variants:
        RESPELL[_v] = _canon

# Fillers carry no content, so `drop_fillers=True` removes them from BOTH sides.
# Off by default: it changes what is being measured, it does not just tidy the
# spelling — and it is the only lever that can touch filler DELETIONS, which a
# respell cannot (a missing token has no spelling to fix).
#
# ⚠ This set is inherited verbatim from the benchmark harness and is known to be
# wrong in both directions: it omits the top hitters (`aa`, `uh`, `um`, `haa`,
# `hmm`, `[event]` tags) and it includes `ya` (*yes*), `kan` (a question tag) and
# `la`, which makes yes/no unscoreable in an IVR transcript. It is left as-is so
# recorded numbers stay reproducible. Pass your own set to `fillers=` rather than
# editing this one, and say which set you used when you publish a number.
FILLERS = frozenset({"herm", "ah", "la", "ya", "eh", "oh", "lo", "ma", "kan"})

# -------------------------------------------------------------------- numbers --
_EN_UNIT = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_EN_SCALE = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}

_MS_UNIT = {
    "kosong": 0, "sifar": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
    "enam": 6, "tujuh": 7, "lapan": 8, "delapan": 8, "sembilan": 9,
}
_MS_SCALE = {"puluh": 10, "ratus": 100, "ribu": 1_000, "juta": 1_000_000}
# `se-` forms are a single word meaning "one <scale>".
_MS_SE = {"sepuluh": 10, "sebelas": 11, "seratus": 100, "seribu": 1_000, "sejuta": 1_000_000}

NUMWORDS = (set(_EN_UNIT) | set(_EN_SCALE) | set(_MS_UNIT) | set(_MS_SCALE)
            | set(_MS_SE) | {"belas", "and"})
SINGLE_DIGIT = {w: v for w, v in list(_EN_UNIT.items()) + list(_MS_UNIT.items()) if v <= 9}


def compose(tokens: list[str]) -> str | None:
    """Fold a run of number words into one value. None when the run is not composable.

    Handles Malay's postfix scales (`dua puluh tiga` -> 23, `tiga ratus` -> 300)
    and English's prefix ones (`twenty three`, `three hundred`), plus `belas`
    teens and the `se-` forms.
    """
    total = 0        # accumulated across thousand/million boundaries
    cur = 0          # accumulated below the next thousand
    pending: int | None = None   # a unit word not yet consumed by a scale
    seen = False

    def scale(mult: int) -> bool:
        """Apply a scale word to whatever is pending. False = not composable."""
        nonlocal total, cur, pending
        if mult >= 1_000:
            base = cur + (pending or 0)
            total += (base or 1) * mult
            cur = 0
        else:
            cur += (pending if pending is not None else 1) * mult
        pending = None
        return True

    for tok in tokens:
        if tok == "and":                      # "one hundred and twenty"
            continue
        seen = True
        if tok in _MS_SE:                     # sepuluh/sebelas are values; se{ratus,ribu,juta} are scales
            v = _MS_SE[tok]
            if v >= 100:
                scale(v)
            elif pending is None:
                pending = v
            else:
                return None
        elif tok == "belas":                  # `tiga belas` -> 13
            if pending is None or not 1 <= pending <= 9:
                return None
            pending += 10
        elif tok in _MS_SCALE or tok in _EN_SCALE:
            scale(_MS_SCALE.get(tok) or _EN_SCALE[tok])
        elif tok in _MS_UNIT or tok in _EN_UNIT:
            v = _MS_UNIT[tok] if tok in _MS_UNIT else _EN_UNIT[tok]
            if pending is None:
                pending = v
            elif pending % 10 == 0 and 20 <= pending < 100 and v < 10:
                pending += v              # `twenty three`
            else:
                # `dua tiga` is a digit run, not a composition — and
                # numbers_to_digits has already peeled those off before we get here.
                return None
        else:
            return None
    return str(total + cur + (pending or 0)) if seen else None


# A multiplier word before a digit is dictation shorthand for a repeat: `triple 0`
# is 000, `double 4` is 44. These are everywhere in call-centre digit read-back (IC
# numbers, phone numbers, application numbers) and nothing here handled them, so
# `triple 0` on one side and `000` on the other scored as a pure mismatch on exactly
# the strings this pass exists for.
#
# `treble` is the British form of `triple`. Malay `dua kali` / `tiga kali` are
# deliberately NOT included: `kali` is also ordinary multiplication ("dua kali lima"
# = two times five), so treating it as a repeat would corrupt real arithmetic.
MULTIPLIER = {"double": 2, "triple": 3, "treble": 3, "quadruple": 4}


def _repeat_digit(mult: str, tok: str) -> str | None:
    """`('triple', '0') -> '000'`, `('double', 'four') -> '44'`; else None.

    The digit may already be a numeral (`triple 0`) or still a word (`triple zero`),
    because this runs both before and after other digit rewrites.
    """
    n = MULTIPLIER.get(mult)
    if n is None:
        return None
    if len(tok) == 1 and tok.isdigit():
        return tok * n
    if tok in SINGLE_DIGIT:
        return str(SINGLE_DIGIT[tok]) * n
    return None


def numbers_to_digits(tokens: list[str]) -> list[str]:
    """Rewrite spelled-out numbers as digits, leaving everything else untouched."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        # Multipliers first: `double`/`triple` are not NUMWORDS, so the run-scanner
        # below would step straight past them and leave the digit bare.
        #
        # Two shapes, deliberately handled differently:
        #  - multiplier + digit WORD (`double four`): expand into repeated words and
        #    reprocess, so an adjacent spelled run absorbs them and `double four one`
        #    becomes `441` rather than `44 1`.
        #  - multiplier + NUMERAL (`triple 0`): emit the repeat as its own token. Do
        #    NOT merge across groups -- `Triple 0. Triple 0.` is two dictated groups
        #    and must stay `000 000`, matching how both humans and the ASR write it.
        #    Greedy merging would give `000000` on one side and `000 000` on the
        #    other, i.e. a mismatch manufactured by the normalizer.
        if i + 1 < len(tokens) and tokens[i] in MULTIPLIER:
            n, nxt = MULTIPLIER[tokens[i]], tokens[i + 1]
            if nxt in SINGLE_DIGIT:
                tokens = tokens[:i] + [nxt] * n + tokens[i + 2:]
                continue                      # reprocess at the same index
            if len(nxt) == 1 and nxt.isdigit():
                out.append(nxt * n)
                i += 2
                continue
        if tokens[i] not in NUMWORDS or tokens[i] == "and":
            out.append(tokens[i])
            i += 1
            continue
        j = i
        while j < len(tokens) and tokens[j] in NUMWORDS:
            j += 1
        run = tokens[i:j]
        while run and run[-1] == "and":       # don't swallow a trailing conjunction
            run.pop()
            j -= 1
        # A run of bare single digits is read out, not composed: `kosong satu dua`
        # is the string 012 (an account or phone number), not zero-one-two or
        # twelve. Which convention wins matters less than applying the SAME one to
        # both sides.
        if len(run) >= 2 and all(t in SINGLE_DIGIT for t in run):
            out.append("".join(str(SINGLE_DIGIT[t]) for t in run))
        else:
            composed = compose(run)
            out.extend([composed] if composed is not None else run)
        i = j
    return out


# ---------------------------------------------------------------- non-speech --
# Non-speech annotations in a reference: [laugh], [inaudible], [noise], (clears
# throat). The shared tokenizer strips the BRACKETS but keeps the word, so the
# reference ends up holding a token `laugh` that no ASR can emit — a guaranteed
# error charged to the model. Measured on Revolab: 12/820 clips, and `laugh`
# scored 7 wrong out of 7.
_NONSPEECH_RE = re.compile(r"[\[\(\<][^\]\)\>]{0,40}[\]\)\>]")


def strip_nonspeech(text: str) -> str:
    return _NONSPEECH_RE.sub(" ", text)


def deterministic_normalize(
    text: str,
    drop_fillers: bool = False,
    fillers: Iterable[str] | None = None,
) -> str:
    """Strip non-speech tags, then case/punct/NFKC, respells, numbers.

    `fillers` overrides `FILLERS` when `drop_fillers` is on — read the warning on
    `FILLERS` before relying on the default.
    """
    drop = frozenset(fillers) if fillers is not None else FILLERS
    toks = [RESPELL.get(t, t) for t in normalize(strip_nonspeech(text))]
    toks = numbers_to_digits(toks)
    if drop_fillers:
        toks = [t for t in toks if t not in drop]
    return " ".join(toks)
