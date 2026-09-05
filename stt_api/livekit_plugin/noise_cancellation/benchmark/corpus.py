"""
Evaluation corpora, cached locally.

Two sources, chosen because they answer different questions:

* **`voicebank`** — the VoiceBank+DEMAND test set, 824 paired clean/noisy
  utterances. This is *the* standard speech-enhancement benchmark, which is the
  point: GTCRN, DTLN and DeepFilterNet all publish PESQ/STOI on exactly this set,
  so the numbers here can be checked against the papers instead of being taken on
  trust. If a model scores far from its published figure, the harness is wrong,
  not the model.

* **`dns`** — the DNS Challenge 2020 dev test set, in two flavours.
  `synthetic_no_reverb` has clean references; `real_recordings` has none and is
  scored by DNSMOS alone. Real recordings matter here because VoiceBank's noise is
  additively mixed at known SNRs, which is not what a phone call sounds like.

Both arrive as single parquet files with the audio inline, so a run costs one
download and then nothing.

A note on rate. Both sets are 16 kHz, matching the agent pipeline this plugin runs
in. The 48 kHz models (RNNoise, DeepFilterNet) therefore see upsampled,
band-limited input and cannot use the top octave they were trained on. That is not
a flaw in the benchmark — it is the condition they would actually face in a 16 kHz
agent — but it does mean their scores here are a floor, not their ceiling on
full-band audio.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

__all__ = ["Item", "SOURCES", "fetch_model", "load"]

_CACHE = Path(
    os.environ.get(
        "NC_BENCHMARK_CACHE", Path.home() / ".cache" / "stt-api" / "nc-benchmark"
    )
)


@dataclass(frozen=True)
class Item:
    """
    One evaluation utterance.

    `clean` is None for real recordings, which have no reference. Every
    reference-based metric skips those rows rather than inventing a reference —
    scoring against the noisy input instead, which is a tempting shortcut, would
    reward a denoiser for doing nothing.
    """

    id: str
    noisy: np.ndarray
    clean: np.ndarray | None
    rate: int
    source: str
    interferer: np.ndarray | None = None
    """The competing speaker, as mixed in, when `degrade="crosstalk"`.

    Kept separately from `noisy` because the interesting question is not how
    clean the output is overall but how much of *this* signal survived — and
    that can only be asked where the target is silent and this is not."""


SOURCES: dict[str, tuple[str, str, bool]] = {
    # name -> (hf dataset repo, parquet path, has clean reference)
    "voicebank": (
        "JacobLinCool/VoiceBank-DEMAND-16k",
        "data/test-00000-of-00001.parquet",
        True,
    ),
    "dns": (
        "nkdem/DNS-Challenge-2020-DevTest-16k",
        "data/synthetic_no_reverb-00000-of-00001.parquet",
        True,
    ),
    "dns-real": (
        "nkdem/DNS-Challenge-2020-DevTest-16k",
        "data/real_recordings-00000-of-00001.parquet",
        False,
    ),
}


def fetch_model(repo: str, filename: str, repo_type: str = "model") -> Path:
    """Download a weight file to the shared cache, or return the cached copy."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo, filename, repo_type=repo_type, cache_dir=str(_CACHE / "hf")
        )
    )


def _decode(blob: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(rate)


def _column(row: dict, *names: str) -> bytes | None:
    """Pull the first present audio column, tolerating schema drift between mirrors."""
    for n in names:
        v = row.get(n)
        if isinstance(v, dict) and v.get("bytes"):
            return v["bytes"]
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
    return None


def load(
    source: str,
    limit: int | None = None,
    *,
    rate: int = 16000,
    seed: int = 0,
    degrade: str | None = None,
    crosstalk_sir: float = 5.0,
) -> Iterator[Item]:
    """
    Yield evaluation items, downloading and caching the parquet on first use.

    `limit` takes a *deterministic random* subset rather than the first N rows.
    VoiceBank's test set is ordered by speaker, so the first N rows are two
    speakers at a handful of SNRs — a subset that moves the scores by more than
    the differences between the models being compared.

    `degrade="telephony"` additionally pushes the noisy signal through an 8 kHz
    µ-law phone path. The clean reference is left untouched, so the task becomes
    "recover wideband speech from a phone call" rather than "remove additive
    noise" — which is what inbound SIP audio actually needs, and the domain the
    call-centre restorers were trained for.

    `degrade="crosstalk"` mixes in a *competing speaker* from another item at
    `crosstalk_sir` dB. This is a different problem from every other condition
    here and the one a voice agent actually breaks on: background speech is
    speech, so a denoiser trained on speech-versus-noise has no reason to remove
    it, and whatever survives reaches the VAD, gets transcribed, and lands in the
    text the end-of-turn model reads. WER on the target speaker cannot see any of
    that — the target's own words may be perfectly clean while the turn
    boundaries are ruined.
    """
    if source not in SOURCES:
        raise SystemExit(
            f"unknown corpus {source!r}; available: {', '.join(sorted(SOURCES))}"
        )
    repo, path, has_clean = SOURCES[source]

    import pyarrow.parquet as pq

    table = pq.read_table(fetch_model(repo, path, repo_type="dataset"))
    n = table.num_rows
    order = np.arange(n)
    if limit is not None and limit < n:
        order = np.random.default_rng(seed).permutation(n)[:limit]
        order.sort()  # keep disk/row order for locality; selection is still random

    from .audio import StreamResampler

    # For crosstalk, the interferer is drawn from a *different* row, offset by a
    # fixed stride so the pairing is deterministic and never pairs an item with
    # itself. Same speaker interfering with itself would be a separation task no
    # model here claims to solve, and would flatter nobody.
    rows_for_interferer = list(order)
    stride = max(1, len(rows_for_interferer) // 2 + 1)

    for pos, i in enumerate(order):
        row = table.slice(int(i), 1).to_pylist()[0]
        noisy_blob = _column(row, "noisy", "audio", "noisy_audio")
        if noisy_blob is None:
            continue
        noisy, sr = _decode(noisy_blob)
        clean = None
        if has_clean:
            clean_blob = _column(row, "clean", "clean_audio")
            if clean_blob is not None:
                clean, _ = _decode(clean_blob)

        if sr != rate:
            noisy = StreamResampler(sr, rate).push(noisy)
            if clean is not None:
                clean = StreamResampler(sr, rate).push(clean)
        if clean is not None:
            m = min(len(clean), len(noisy))
            clean, noisy = clean[:m], noisy[:m]

        interferer = None
        if degrade == "telephony":
            from .audio import telephony_degrade

            # Only the input is degraded. The reference stays wideband, so a model
            # that merely passes the narrowband signal through cannot score well.
            noisy = telephony_degrade(noisy, rate)
        elif degrade == "crosstalk":
            j = rows_for_interferer[(pos + stride) % len(rows_for_interferer)]
            other = table.slice(int(j), 1).to_pylist()[0]
            blob = _column(other, "clean", "clean_audio") or _column(other, "noisy", "audio")
            if blob is not None:
                spk, sr2 = _decode(blob)
                if sr2 != rate:
                    spk = StreamResampler(sr2, rate).push(spk)
                interferer = _fit_and_scale(spk, noisy, crosstalk_sir)
                noisy = (noisy + interferer).astype(np.float32)
        elif degrade:
            raise SystemExit(
                f"unknown degradation {degrade!r}; available: telephony, crosstalk"
            )

        yield Item(
            id=str(row.get("id") or row.get("filename") or f"{source}-{i}"),
            noisy=noisy,
            clean=clean,
            rate=rate,
            source=source,
            interferer=interferer,
        )


def _fit_and_scale(spk: np.ndarray, target: np.ndarray, sir_db: float) -> np.ndarray:
    """Tile or trim the interferer to the target's length, then set its level."""
    if len(spk) < len(target):
        reps = int(np.ceil(len(target) / max(len(spk), 1)))
        spk = np.tile(spk, reps)
    spk = spk[: len(target)].astype(np.float32)
    p_t = float(np.mean(target.astype(np.float64) ** 2)) or 1e-12
    p_s = float(np.mean(spk.astype(np.float64) ** 2)) or 1e-12
    gain = np.sqrt(p_t / (10 ** (sir_db / 10.0)) / p_s)
    return (spk * gain).astype(np.float32)
