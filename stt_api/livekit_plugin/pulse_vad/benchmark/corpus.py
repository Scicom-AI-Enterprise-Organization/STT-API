"""
Audio in, labelled items out — local files or an S3 prefix.

"Can PulseVAD replace silero" is only answerable with **certain negatives**, and
a plain speech recording does not provide them: a clip labelled "speech" also
contains breaths, inter-word gaps and trailing silence, so a detector that fires
on 70 % of its frames may be exactly right or badly wrong and the recording
cannot tell you which.

Two sources of labels, and the first is much better than the second.

**Real, from production.** The `stt-dev/drift/` prefix holds one wav per user
turn with a sibling `.json` carrying the Whisper transcript of that turn. An
**empty transcript is a negative**: audio the production stack cut as a turn and
in which a large ASR then found nothing. Roughly 13 % of the corpus is that, and
it is exactly the material a VAD false-fires on — real room tone, line noise,
breaths, keyboard, hold music — rather than anything synthetic. The label is
evidence rather than proof (Whisper can drop very quiet or very short speech),
but it is the *same* label for both detectors, so the comparison is fair even
where the label is not perfect.

**Constructed, for timing.** Onset latency needs a known speech-start instant,
and production turns are already tightly cut. So positive clips are also
sandwiched between pads of known non-speech:

    |<-- pad -->|<------- clip -------->|<-- pad -->|
     certain                             certain
     negative    speech-bearing region   negative

Pads are not only digital silence: a VAD that merely has to reject `zeros` is
untested, and a 2,118-parameter model is likeliest to fail against silero's
1.8 M exactly where there is a real noise floor.
"""

from __future__ import annotations

import json as _json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")

__all__ = [
    "Clip",
    "Item",
    "Region",
    "build_items",
    "load_audio",
    "local_clips",
    "parse_s3_uri",
    "s3_clips",
]


@dataclass
class Clip:
    name: str
    audio: np.ndarray
    transcript: str | None = None
    has_speech: bool | None = None
    """True/False where a transcript settles it, None where nothing is known."""


@dataclass(frozen=True)
class Region:
    start: float
    end: float
    speech: bool
    """True where speech is *expected* — an upper bound, because the region still
    contains pauses. False is ground truth: firing there is wrong."""


@dataclass
class Item:
    name: str
    audio: np.ndarray
    condition: str
    kind: str = "positive"
    """`positive` (a transcribed turn) or `negative` (empty transcript)."""
    regions: list[Region] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    @property
    def body(self) -> Region | None:
        return next((r for r in self.regions if r.speech), None)

    def labels(self, times: np.ndarray) -> np.ndarray:
        out = np.zeros(len(times), dtype=bool)
        for r in self.regions:
            if r.speech:
                out |= (times >= r.start) & (times < r.end)
        return out

    def certain_negative(self, times: np.ndarray, guard: float = 0.0) -> np.ndarray:
        """
        Mask of timestamps where firing is definitely wrong.

        `guard` seconds after each speech region are excluded. That window is the
        VAD's *configured* hangover (`min_silence_duration`) draining, not a false
        positive — counting it as one makes both detectors look ~20 % wrong and
        measures the setting rather than the model.
        """
        out = np.zeros(len(times), dtype=bool)
        for r in self.regions:
            if not r.speech:
                out |= (times >= r.start) & (times < r.end)
        if guard > 0:
            for r in self.regions:
                if r.speech:
                    out &= ~((times >= r.end) & (times < r.end + guard))
        return out


def load_audio(path: str | Path, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 at `rate`, via the shared StreamResampler."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != rate:
        from ...noise_cancellation.benchmark.audio import StreamResampler

        audio = StreamResampler(sr, rate).push(audio.astype(np.float32))
    return np.ascontiguousarray(audio, dtype=np.float32)


def local_clips(paths: list[str], limit: int | None = None) -> list[Clip]:
    """Expand files and directories. No transcripts, so `has_speech` stays None."""
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(
                sorted(f for f in path.rglob("*") if f.suffix.lower() in AUDIO_SUFFIXES)
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"no such audio path: {p}")
    if limit:
        files = files[:limit]
    return [Clip(name=f.stem, audio=load_audio(f)) for f in files]


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/some/key.wav` -> `("bucket", "some/key.wav")`."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {uri!r}")
    return bucket, key


_S3_ALIASES: dict[str, tuple[str, ...]] = {
    "endpoint_url": ("S3_ENDPOINT_URL", "S3_ENDPOINT", "AWS_ENDPOINT_URL"),
    "region": ("S3_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
    "access_key": ("S3_ACCESS_KEY_ID", "S3_ACCESS_KEY", "AWS_ACCESS_KEY_ID"),
    "secret_key": ("S3_SECRET_ACCESS_KEY", "S3_SECRET_KEY", "AWS_SECRET_ACCESS_KEY"),
    "session_token": ("S3_SESSION_TOKEN", "AWS_SESSION_TOKEN"),
    "bucket": ("S3_BUCKET",),
    "prefix": ("S3_PREFIX",),
    "addressing_style": ("S3_ADDRESSING_STYLE",),
}
"""Several spellings per setting, deliberately. `S3_SECRET_ACCESS_KEY` mirrors
boto3's own names and `S3_SECRET_KEY` is the shorter form people actually type;
a benchmark that ignores a filled-in credential over an `_ACCESS` infix is a bad
way to spend an afternoon."""


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _s3_config(env_file: str | Path | None = None) -> dict[str, str]:
    """
    Resolve S3 settings from a `.env` and the environment, under any spelling.

    Precedence is `.env` < process environment, and reading the file never
    mutates `os.environ` — the rule `stt_api.evaluation` follows, for the same
    reason: a library that edits the process environment on import breaks
    whatever is hosting it.
    """
    path = Path(env_file) if env_file else Path(__file__).resolve().parents[4] / ".env"
    raw = _read_env_file(path)
    known = {name for names in _S3_ALIASES.values() for name in names}
    raw.update({k: v for k, v in os.environ.items() if v and k in known})
    cfg: dict[str, str] = {}
    for canonical, names in _S3_ALIASES.items():
        for name in names:
            if raw.get(name):
                cfg[canonical] = raw[name]
                break
    return cfg


def _s3_client(cfg: dict[str, str]):
    import boto3
    from botocore.config import Config

    kw: dict = {}
    if cfg.get("endpoint_url"):
        # Only for S3-compatible gateways. Set against real AWS it surfaces as a
        # DNS error, which reads like a network outage rather than a config slip.
        kw["endpoint_url"] = cfg["endpoint_url"]
    if cfg.get("region"):
        kw["region_name"] = cfg["region"]
    if cfg.get("access_key") and cfg.get("secret_key"):
        kw["aws_access_key_id"] = cfg["access_key"]
        kw["aws_secret_access_key"] = cfg["secret_key"]
        if cfg.get("session_token"):
            kw["aws_session_token"] = cfg["session_token"]
    if cfg.get("addressing_style"):
        kw["config"] = Config(s3={"addressing_style": cfg["addressing_style"]})
    return boto3.client("s3", **kw)


def s3_clips(
    bucket: str | None = None,
    prefix: str | None = None,
    *,
    limit: int | None = None,
    env_file: str | Path | None = None,
    cache_dir: str | Path | None = None,
    with_transcripts: bool = True,
    workers: int = 16,
) -> list[Clip]:
    """
    Pull audio from S3 (or any S3-compatible endpoint), with labels where present.

    A sibling `<key>.json` holding a `transcription` field labels the clip:
    non-empty means speech, empty means none. Clips keep `has_speech=None` when
    no sibling exists, and then contribute to agreement but not to accuracy.

    Credentials resolve as documented in `.env.example`: explicit `S3_*` keys,
    then `AWS_*`, then boto3's own chain (SSO profile, instance role).
    """
    cfg = _s3_config(env_file)
    bucket = bucket or cfg.get("bucket")
    if not bucket:
        raise ValueError(
            "no S3 bucket given: pass --s3-bucket or set S3_BUCKET in .env "
            "(see .env.example)"
        )
    prefix = prefix if prefix is not None else cfg.get("prefix", "")
    client = _s3_client(cfg)
    cache = Path(cache_dir or Path.home() / ".cache" / "stt-api" / "vad-bench" / bucket)
    cache.mkdir(parents=True, exist_ok=True)

    if prefix and Path(prefix).suffix.lower() in AUDIO_SUFFIXES:
        keys = [prefix]  # a single pasted s3:// object
    else:
        keys = _list_keys(client, bucket, prefix, limit)
    if not keys:
        raise ValueError(f"no audio found under s3://{bucket}/{prefix}")

    def fetch(key: str) -> Clip | None:
        dest = cache / key.replace("/", "_")
        if not dest.exists():
            client.download_file(bucket, key, str(dest))
        clip = Clip(name=Path(key).stem, audio=load_audio(dest))
        if with_transcripts:
            meta = cache / (key.replace("/", "_").rsplit(".", 1)[0] + ".json")
            if not meta.exists():
                try:
                    client.download_file(
                        bucket, key.rsplit(".", 1)[0] + ".json", str(meta)
                    )
                except Exception:  # noqa: BLE001 - no sibling; stay unlabelled
                    return clip
            try:
                doc = _json.loads(meta.read_text())
            except Exception:  # noqa: BLE001
                return clip
            text = (doc.get("transcription") or "").strip()
            clip.transcript = text
            clip.has_speech = bool(text)
        return clip

    with ThreadPoolExecutor(workers) as pool:
        clips = [c for c in pool.map(fetch, keys) if c is not None]
    return clips


def _list_keys(client, bucket: str, prefix: str, limit: int | None) -> list[str]:
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if Path(obj["Key"]).suffix.lower() in AUDIO_SUFFIXES:
                keys.append(obj["Key"])
        if limit and len(keys) >= limit:
            break
    return keys[:limit] if limit else keys


def _noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "silence":
        return np.zeros(n, dtype=np.float32)
    if kind == "white":
        return (rng.standard_normal(n) * 0.003).astype(np.float32)  # ~ -50 dBFS
    if kind == "room":
        # 1/f-ish. Far closer to a real room or an open SIP leg than white noise,
        # and the harder negative of the two.
        x = np.cumsum(rng.standard_normal(n))
        x = x - x.mean()
        return (x / (np.abs(x).max() + 1e-9) * 0.01).astype(np.float32)
    raise ValueError(f"unknown pad kind {kind!r}")


def build_items(
    clips: list[Clip],
    *,
    pad_seconds: float = 1.5,
    pad_kinds: tuple[str, ...] = ("silence", "room"),
    conditions: tuple[str, ...] = ("clean",),
    snr_db: float = 10.0,
    seed: int = 0,
) -> list[Item]:
    """
    Turn clips into scorable items.

    A clip known to hold **no** speech becomes a negative item scored as-is — it
    is already real non-speech audio and padding it would only dilute the result.
    Everything else is sandwiched between pads so onset latency has a known
    reference instant.

    `conditions` may include `clean`, `noisy` (white noise at `snr_db`) and
    `telephony` (8 kHz band-limit plus µ-law — what inbound SIP audio actually
    is, the condition this repo cares about most and the one a tiny model is
    least likely to survive).
    """
    from ...noise_cancellation.benchmark.audio import rms, telephony_degrade

    rng = np.random.default_rng(seed)
    pad_n = int(pad_seconds * SAMPLE_RATE)
    items: list[Item] = []

    def conditioned(base: np.ndarray, ref: np.ndarray, cond: str) -> np.ndarray:
        if cond == "clean":
            return base
        if cond == "noisy":
            noise = rng.standard_normal(len(base)).astype(np.float32)
            target = max(rms(ref), 1e-6) / (10 ** (snr_db / 20.0))
            return base + noise * (target / max(rms(noise), 1e-9))
        if cond == "telephony":
            return telephony_degrade(base, SAMPLE_RATE)
        raise ValueError(f"unknown condition {cond!r}")

    for clip in clips:
        if len(clip.audio) < SAMPLE_RATE // 4:
            continue  # too short to say anything about

        if clip.has_speech is False:
            # Real non-speech, used exactly as recorded.
            for cond in conditions:
                audio = conditioned(clip.audio, clip.audio, cond)
                items.append(
                    Item(
                        name=clip.name,
                        audio=np.clip(audio, -1.0, 1.0).astype(np.float32),
                        condition=cond,
                        kind="negative",
                        regions=[Region(0.0, len(audio) / SAMPLE_RATE, False)],
                    )
                )
            continue

        for kind in pad_kinds:
            head = _noise(kind, pad_n, rng)
            tail = _noise(kind, pad_n, rng)
            base = np.concatenate([head, clip.audio, tail]).astype(np.float32)
            body = Region(
                pad_seconds, pad_seconds + len(clip.audio) / SAMPLE_RATE, True
            )
            regions = [
                Region(0.0, pad_seconds, False),
                body,
                Region(body.end, body.end + pad_seconds, False),
            ]
            for cond in conditions:
                audio = conditioned(base, clip.audio, cond)
                items.append(
                    Item(
                        name=f"{clip.name}/{kind}",
                        audio=np.clip(audio, -1.0, 1.0).astype(np.float32),
                        condition=cond,
                        kind="positive",
                        regions=regions,
                    )
                )
    return items
