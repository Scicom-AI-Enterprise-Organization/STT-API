"""The optional LLM layer — an OpenAI-compatible chat client for the residue the
rules cannot settle.

Nothing about any particular endpoint is baked in. Configuration comes from
explicit arguments, from the process environment, or from a `.env` file you point
at — in that order. Standard library only: `urllib`, no SDK.

⚠ THE ONE RULE THAT MAKES THIS MEASUREMENT VALID: the model never sees the
reference and the hypothesis together. Each side is normalized in its own call,
from its own text alone. Show a model both and it will quietly edit the
hypothesis toward the reference — WER drops, the number becomes meaningless, and
nothing in the output looks wrong.

`normalize_pair()` exists to STUDY that effect, not to produce a number. Measured
on Revolab 820, call-centre arm, against a control that pairs each reference with
a *different* clip's hypothesis:

    independent      8.61          each side alone
    pair             8.12  -0.49   sees its own partner
    pair-shuffled    8.59  -0.03   sees a stranger

Convention knowledge is corpus-wide, so the honest part of the gain is what
survives shuffling: **+0.03 pp. The other 0.46 pp — 94% of it — is leakage.**
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

__all__ = ["LLMClient", "PAIR_SYSTEM", "SYSTEM", "load_dotenv"]


SYSTEM = """You normalise the SPELLING of a single Malay/English/Chinese speech transcript.

You are given ONE text. You do not know whether it is a reference or a machine
transcription, and there is no other text to compare it against. Normalise it on its
own terms.

Rules:
- Write every number as digits: "dua puluh tiga" -> "23", "twenty three" -> "23".
  Digits read out one by one stay one string: "kosong satu dua" -> "012".
- Canonicalise hesitation fillers to: herm (um/uh/hmm/erm), ah, la (lah), ya (ye/yeah), ok (okay).
- Lowercase. Remove punctuation.
- Keep Chinese characters as Chinese characters. Never romanise them.

Absolutely never:
- add a word that is not there, delete a word that is, or swap a word for a different one
- correct grammar, spelling of content words, or apparent transcription mistakes
- complete an unfinished sentence

If nothing needs changing, return the text unchanged. Reply with ONLY the normalised
text, no quotes and no explanation."""


PAIR_SYSTEM = """You are given TWO transcripts, A and B, of the SAME utterance, written by two
different transcribers. Normalise the SPELLING of each so that where they say the same thing, they
WRITE it the same way.

Rules:
- Numbers as digits: "dua puluh tiga" -> "23", "twenty three" -> "23". Digits read out one by one
  stay one string: "kosong satu dua" -> "012".
- Hesitation fillers: herm (um/uh/hmm/erm), ah, la (lah), ya (ye/yeah), ok (okay).
- Lowercase, no punctuation. Keep Chinese characters Chinese; never romanise them.
- Where A and B use different spellings, word-splits or number formats for THE SAME SPOKEN WORDS,
  converge them on one form.

CRITICAL — where A and B contain DIFFERENT WORDS, they must STAY different. Those are real
transcription disagreements and they are the thing being measured. You are only allowed to change
how a word is spelled, spaced or digitised:
- never copy a word from one side into the other
- never add, delete, or substitute a word to make the two match
- never fix an apparent mistake, complete a truncated sentence, or repair grammar
If B is missing words that A has, LEAVE IT MISSING.

Reply with ONLY this JSON, nothing else: {"a": "<normalised A>", "b": "<normalised B>"}"""


def load_dotenv(path: str | os.PathLike | None) -> dict[str, str]:
    """Parse a `.env` into a dict. Never mutates `os.environ`.

    A library that edits the process environment on import is a library that
    breaks the host application. Pass the result to `LLMClient.from_env(env=...)`
    instead.
    """
    env: dict[str, str] = {}
    if not path:
        return env
    p = Path(path)
    if not p.exists():
        return env
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


@dataclass
class LLMClient:
    """Minimal OpenAI-compatible chat client for the normalization prompts.

    Works against anything that speaks `POST {base_url}/chat/completions` — vLLM,
    OpenRouter, the OpenAI API, an internal proxy.
    """

    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    retries: int = 3
    timeout: int = 120
    max_tokens: int = 1024

    @classmethod
    def from_env(
        cls,
        env: dict[str, str] | None = None,
        env_file: str | os.PathLike | None = None,
        **overrides,
    ) -> "LLMClient":
        """Build from `.env` file < process environment < explicit overrides.

        Reads `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `MODEL_NAME` (or
        `OPENAI_MODEL`). Any of them can be passed directly instead.
        """
        src: dict[str, str] = {}
        src.update(load_dotenv(env_file))
        src.update({k: v for k, v in os.environ.items() if v})
        if env:
            src.update(env)
        cfg = {
            "base_url": src.get("OPENAI_BASE_URL", ""),
            "api_key": src.get("OPENAI_API_KEY", ""),
            "model": src.get("MODEL_NAME") or src.get("OPENAI_MODEL") or "gpt-4o-mini",
        }
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**cfg)

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def _post(self, messages: list[dict], max_tokens: int) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }).encode()
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    f"{self.base_url.rstrip('/')}/chat/completions", data=body,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"].strip()
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.retries} tries: {last}")

    def normalize(self, text: str) -> str:
        """Normalize ONE transcript, from its own text alone. The safe call."""
        return self._post(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            self.max_tokens,
        )

    def normalize_pair(self, a: str, b: str) -> tuple[str, str]:
        """Normalize two transcripts of one utterance together — see the ⚠ above.

        Kept for studying the leakage effect. Never quote a number produced by it.
        """
        txt = self._post(
            [{"role": "system", "content": PAIR_SYSTEM},
             {"role": "user", "content": f"A: {a}\nB: {b}"}],
            max(self.max_tokens, 2048),
        )
        m = re.search(r"\{.*\}", txt, re.S)
        obj = json.loads(m.group(0) if m else txt)
        return str(obj["a"]), str(obj["b"])
