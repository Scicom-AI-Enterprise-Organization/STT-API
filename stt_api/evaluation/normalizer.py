"""`Normalizer` — deterministic pass, optionally the LLM, always validated.

The cache is load-bearing, not an optimisation: it is keyed by text hash, so a
reference string normalizes IDENTICALLY across every model you score against it.
Two arms can then never differ because their shared reference was normalized two
different ways.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable, MutableMapping

from .deterministic import deterministic_normalize
from .llm import LLMClient
from .validation import validate

__all__ = ["Normalizer", "Rejection"]

logger = logging.getLogger(__name__)

MODES = ("deterministic", "llm", "both", "pair")


class Rejection(tuple):
    """`(original, candidate, violations)` — one LLM edit the validator refused."""

    __slots__ = ()

    def __new__(cls, original: str, candidate: str, violations: list[str]):
        return super().__new__(cls, (original, candidate, violations))

    original = property(lambda self: self[0])
    candidate = property(lambda self: self[1])
    violations = property(lambda self: self[2])


class Normalizer:
    """Callable: text in, normalized text out.

        norm = Normalizer()                          # rules only, offline, free
        norm = Normalizer("both", client=client)     # rules + LLM residue

    `mode`:
      - `deterministic` — rules only. No network. The default, because it is
        measured to capture ~90% of the recoverable gap.
      - `llm` / `both` — one LLM call per unique text, then the rules on top.
      - `pair` — the leakage study; see `LLMClient.normalize_pair`.

    `cache` may be a path (JSON file, loaded and saved by `save()`) or any
    mutable mapping you already own — a dict, or a Redis-backed shim.
    """

    def __init__(
        self,
        mode: str = "deterministic",
        client: LLMClient | None = None,
        cache: str | Path | MutableMapping[str, str] | None = None,
        drop_fillers: bool = False,
        fillers: Iterable[str] | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode != "deterministic":
            # No client passed: read the process environment. A pip-installed copy
            # has nowhere sensible to put a `.env`, so env vars alone must be
            # enough to turn the LLM pass on.
            if client is None:
                client = LLMClient.from_env()
            if not client.configured:
                raise ValueError(
                    f"mode={mode!r} needs an LLM endpoint. Set OPENAI_BASE_URL and "
                    f"OPENAI_API_KEY (and MODEL_NAME) in the environment, or pass "
                    f"client=LLMClient(base_url=..., api_key=..., model=...). "
                    f"mode='deterministic' needs neither and is the default."
                )
        self.mode = mode
        self.client = client
        self.drop_fillers = drop_fillers
        self.fillers = fillers
        self.cache_path: Path | None = None
        if isinstance(cache, (str, Path)):
            self.cache_path = Path(cache)
            self.cache: MutableMapping[str, str] = (
                json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
            )
        else:
            self.cache = cache if cache is not None else {}
        self.rejected: list[Rejection] = []
        self.errors = 0

    # ------------------------------------------------------------------ llm --
    def _llm(self, text: str) -> str:
        """The model's normalization if it validates, else the original.

        The cache holds the RAW model output and validation runs on every read —
        cache the post-validation text instead and a fix to the validator can
        never be picked up on a re-run, because the rejected edit was already
        overwritten by its own input.
        """
        key = hashlib.sha256(text.encode()).hexdigest()[:24]
        if key in self.cache:
            out = self.cache[key]
        else:
            try:
                out = self.client.normalize(text)
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                logger.warning("LLM error, keeping original: %s", e)
                return text
            self.cache[key] = out
        bad = validate(text, out)
        if bad:
            # A rejected edit is not a failure of the run — it is the guard doing
            # its job. Keeping the original is always safe; accepting a content
            # mutation is not.
            self.rejected.append(Rejection(text, out, bad))
            return text
        return out

    def _pair(self, a: str, b: str) -> tuple[str, str]:
        """Pair-aware: the model sees BOTH transcripts and converges their spelling.

        Each side is still validated against ITS OWN original, so a word copied
        across is rejected exactly as in single-side mode. That validator is the
        only thing between this mode and a meaningless number.
        """
        key = "pair:" + hashlib.sha256(f"{a}\x00{b}".encode()).hexdigest()[:24]
        if key in self.cache:
            na, nb = json.loads(self.cache[key])
        else:
            try:
                na, nb = self.client.normalize_pair(a, b)
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                logger.warning("LLM pair error, keeping originals: %s", e)
                return a, b
            self.cache[key] = json.dumps([na, nb])
        out = []
        for orig, cand in ((a, na), (b, nb)):
            bad = validate(orig, cand)
            if bad:
                self.rejected.append(Rejection(orig, cand, bad))
                cand = orig
            out.append(cand)
        return out[0], out[1]

    # ---------------------------------------------------------------- public --
    def cached(self, text: str) -> bool:
        return hashlib.sha256(text.encode()).hexdigest()[:24] in self.cache

    def rules_only(self, text: str) -> str:
        """The deterministic layer alone, whatever `mode` is set to."""
        return deterministic_normalize(text, self.drop_fillers, self.fillers)

    def __call__(self, text: str) -> str:
        if self.mode in ("llm", "both"):
            text = self._llm(text)
        return self.rules_only(text)

    def save(self) -> None:
        """Persist the cache if it was constructed from a path. No-op otherwise."""
        if self.cache_path is not None:
            self.cache_path.write_text(
                json.dumps(dict(self.cache), ensure_ascii=False, indent=0)
            )
