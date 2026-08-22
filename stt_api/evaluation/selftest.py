"""Offline checks on the deterministic layer and the validator.

No network, no key, no dataset — so anyone handed this package can verify it
behaves as documented before trusting a number from it. The same constants are
what `tests/test_evaluation.py` parametrizes over, so the two can never drift.
"""

from __future__ import annotations

from .deterministic import deterministic_normalize
from .validation import validate

__all__ = ["CORRUPTING_EDITS", "LEGAL_EDITS", "NORMALIZE_CASES", "run"]

NORMALIZE_CASES = [
    ("saya bayar dua puluh tiga ringgit", "saya bayar 23 ringgit"),
    ("i paid twenty three ringgit", "i paid 23 ringgit"),
    ("nombor saya kosong satu dua tiga empat", "nombor saya 01234"),
    ("tiga ratus lima puluh", "350"),
    ("seribu dua ratus", "1200"),
    ("tiga belas", "13"),
    ("one hundred and twenty", "120"),
    ("sebelas", "11"),
    ("okay lah, hmm saya tak tahu", "ok la herm saya tak tahu"),
    ("RM50 sahaja", "rm50 sahaja"),          # digits already written stay put
    ("dia ada lima anak", "dia ada 5 anak"),
    # non-speech annotations vanish entirely — brackets AND the word inside
    ("Belum. [laugh] Bodoh.", "belum bodoh"),
    ("[inaudible] nak buat", "nak buat"),
    ("saya (clears throat) nak bayar", "saya nak bayar"),
    # multiplier shorthand in digit read-back: pervasive in call-centre audio
    # (IC / phone / application numbers) and previously unhandled, so `triple 0`
    # against `000` scored as a pure mismatch on exactly the strings that matter.
    ("nombor aplikasi ialah Triple 0", "nombor aplikasi ialah 000"),
    ("double four one", "441"),
    ("triple zero", "000"),
    ("treble seven", "777"),
    ("quadruple nine", "9999"),
    ("double empat", "44"),
    # ...and it must NOT fire when the multiplier is an ordinary word
    ("i double checked the double room", "i double checked the double room"),
]

# The validator must PASS these — they are convention edits, which is the point.
LEGAL_EDITS = [
    ("dua puluh tiga ringgit", "23 ringgit"),
    ("Okay lah", "ok la"),
    ("peduli kan", "pedulikan"),
    # a detached affix, respelled on the way out — the single commonest real LLM
    # edit on this corpus
    ("Baguslah. Itu saja", "bagus la itu saja"),
    ("okaylah", "ok la"),
    # multiplier shorthand: the single commonest rejection on call-centre read-back
    ("Triple 0. Triple 0.", "000 000"),
    ("nombor saya double 4 triple 0", "nombor saya 44 000"),
    ("triple zero", "000"),
]

# ...and CATCH these, every one of which would corrupt the measurement.
CORRUPTING_EDITS = [
    ("saya nak bayar", "saya nak bayar sekarang"),   # inserted
    ("saya nak bayar bil", "saya nak bayar"),        # deleted
    ("dua puluh tiga", "24"),                        # wrong value
    ("saya suka nasi", "saya suka mee"),             # swapped content
    # a multiplier must not become a licence to emit any digit string
    ("triple 0", "111"),                             # wrong digit
    ("triple 0", "00"),                              # wrong repeat count
    ("double room", "00"),                           # not a digit at all
]


def run() -> list[str]:
    """Return a list of failure descriptions — empty means everything passed."""
    failures: list[str] = []
    for src, want in NORMALIZE_CASES:
        got = deterministic_normalize(src)
        if got != want:
            failures.append(f"{src!r}\n  want {want!r}\n  got  {got!r}")
    for o, n in LEGAL_EDITS:
        bad = validate(o, n)
        if bad:
            failures.append(f"validator rejected a legal edit: {o!r} -> {n!r} {bad}")
    for o, n in CORRUPTING_EDITS:
        if not validate(o, n):
            failures.append(f"validator ACCEPTED a corrupting edit: {o!r} -> {n!r}")
    return failures
