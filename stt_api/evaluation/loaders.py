"""Read ref/hyp pairs out of the file formats these evaluations tend to produce.

CSV, TSV, JSON and JSONL, plus the per-sample dumps the STT benchmark harness
writes. Nothing clever — it exists so every caller does not re-write it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .score import Pair

__all__ = ["load_pairs", "load_rows"]


def load_rows(path: str | Path) -> list[dict]:
    """Rows from a `.csv` / `.tsv` / `.json` / `.jsonl` file."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)
    suffix = src.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    if suffix in (".csv", ".tsv"):
        import csv
        with src.open(newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ","))
    data = json.loads(src.read_text())
    if isinstance(data, dict):
        # A results file with the rows under some key — take the first list of dicts.
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        raise ValueError(f"{src.name} is a JSON object with no list of rows in it")
    return data


def load_pairs(
    path: str | Path,
    ref_field: str = "ref",
    hyp_field: str = "hyp",
    category_field: str = "category",
    limit: int = 0,
) -> list[Pair]:
    """Rows -> `Pair`s, skipping any row with an empty reference.

    `ref_field` is the ground truth and `hyp_field` is the ASR output. A row whose
    hypothesis is an empty string is KEPT — a blank transcription is a real
    failure mode (it scores as all-deletions) and dropping those flatters the
    model. Only a missing/None hypothesis field is skipped.
    """
    rows = load_rows(path)
    if rows and ref_field not in rows[0]:
        raise KeyError(f"no column {ref_field!r} in {Path(path).name}; columns are "
                       f"{sorted(rows[0])}")
    pairs: list[Pair] = []
    for i, r in enumerate(rows):
        ref, hyp = r.get(ref_field), r.get(hyp_field)
        if not ref or hyp is None:
            continue
        pairs.append(Pair(ref=str(ref), hyp=str(hyp), id=str(r.get("id", i)),
                          category=r.get(category_field)))
    return pairs[:limit] if limit else pairs
