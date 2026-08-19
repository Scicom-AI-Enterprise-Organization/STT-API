"""Load a dataset's per-row `canonical` variant map from HuggingFace parquet.

Some benchmark sets ship a hand-declared `variant -> canonical` map per row
(Revolab/ASR-Benchmark-Public: 316 pairs over 198 keys, on 594 of 820 rows) and
the official scorer folds it into BOTH sides before measuring. It forgives
`ramlee`/`ramli`, `film`/`filem`, `card`/`kad`, plus ~80 filler spellings.

⚠ If the dataset has one, apply it, or your baseline is not the published number
— measured 10.39 without it against the true 9.33, and every "recovered" figure
then bills this library for folding the harness already did.

⚠ A hand-declared list is the only defensible mechanism. Phonetic distance cannot
separate `fallujah`/`faluyah` (same city, misspelt) from
`bordentown`/`bordertown` (different cities) at any threshold — what distinguishes
them is whether the referent is the same, which is semantics.

This is the only module here with third-party imports, and they are lazy: the
rest of the package is standard library only.
"""

from __future__ import annotations

import json
import os
import re

__all__ = ["load_canonical"]


def load_canonical(
    dataset: str,
    config: str | None = None,
    split: str = "train",
    token: str | None = None,
) -> dict[str, dict[str, str]]:
    """`{row_id: {variant: canonical}}` from the dataset's `canonical` column.

    Returns `{}` when the dataset has no such column. `config` is the parquet
    subdirectory (the HF config name), default `data`.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:  # pragma: no cover - depends on the install extra
        raise ImportError(
            f"load_canonical needs `pip install 'stt-api[evaluation]'` "
            f"(huggingface_hub + pyarrow): {e}. Every other part of "
            f"stt_api.evaluation is standard-library only."
        ) from e

    prefix = config or "data"
    token = token or os.environ.get("HF_TOKEN")
    shards = sorted(
        f for f in HfApi().list_repo_files(dataset, repo_type="dataset", token=token)
        if re.fullmatch(rf"{re.escape(prefix)}/{re.escape(split)}-\d+-of-\d+\.parquet", f))
    if not shards:
        raise FileNotFoundError(f"no parquet shards for '{prefix}/{split}-*' in {dataset}")
    out: dict[str, dict[str, str]] = {}
    for shard in shards:
        table = pq.read_table(hf_hub_download(dataset, shard, repo_type="dataset", token=token))
        if "canonical" not in table.column_names:
            return {}
        ids = table["id"] if "id" in table.column_names else None
        for i in range(table.num_rows):
            raw = table["canonical"][i].as_py()
            vmap: dict[str, str] = {}
            for canon, variants in json.loads(raw or "{}").items():
                for v in variants:
                    vmap[v.lower()] = canon.lower()
            rid = ids[i].as_py() if ids is not None else f"{split}-{len(out):05d}"
            out[rid] = vmap
    return out
