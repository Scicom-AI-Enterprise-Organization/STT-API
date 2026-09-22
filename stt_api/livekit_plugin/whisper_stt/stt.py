"""
Whisper STT for LiveKit Agents, with the two guards a VAD-fed pipeline needs.

The endpoint is an ordinary OpenAI-compatible `/v1/audio/transcriptions`. What
this plugin adds is refusing to turn silence into an interruption.

## The bug this exists to stop

Whisper answers a silent segment with **a single space**, not an empty string.
Measured against the production endpoint, every one of these returned `' '` with
HTTP 200: 2 s of digital silence, 0.5 s of digital silence, 2 s of quiet room
tone, 2 s of louder noise. Across 1,503 real production turns, **180 (12 %)**
came back as exactly `' '`.

A space is not empty, and LiveKit only checks emptiness. The chain, verified by
driving `AudioRecognition._on_stt_event` directly:

    stt/stream_adapter.py:149    elif not t_event.alternatives[0].text: continue
    voice/audio_recognition.py:1215   if not transcript: return
    voice/audio_recognition.py:1217   self._hooks.on_final_transcript(...)
    voice/agent_activity.py:2531      -> self._interrupt_by_audio_activity()

`''` is falsy and stops at the guard. `' '` is truthy, sails through both, and
reaches `on_final_transcript` — which calls `_interrupt_by_audio_activity()`
**with no text check of its own**. Measured outcome per return value:

    ''        -> hook does not fire   -> no interruption
    ' '       -> hook fires           -> interrupts the agent
    '  \n\t'  -> hook fires           -> interrupts the agent
    '.'       -> hook fires           -> interrupts the agent

So the agent gets cut off by silence, and nothing logs an error.

**Return `text=""`, never `alternatives=[]`.** `audio_recognition.py:1207` does
`ev.alternatives[0].text` with no length check, so an empty alternatives list
trades the interruption for an `IndexError`. An alternative carrying empty text
is safe on every path, because every guard tests the text rather than the list.

## The second guard: don't spend the request at all

A blank filter still pays for the round trip. A local check does not, and the
separation in production audio is not marginal — over 1,503 turns:

    speech turns      RMS  p05 -21.6 dBFS   median -18.4 dBFS
    non-speech turns  RMS  p95 -42.3 dBFS   median **-240 dBFS**

That median is not a typo: **169 of the 180 non-speech turns are literally all
zero samples**, every one of them exactly 0.30 s long. At `silence_floor_dbfs`
of -50 the local gate skips **170 of 180 (94.4 %)** of the useless calls and
loses **0 of 1,323** real speech turns.

Worth knowing where those all-zero segments do *not* come from: silero. Fed
0.30 s of digital silence it peaks at p(speech) **0.0089** against its 0.5
threshold, so it cannot be the thing emitting them. Something upstream is
sending fixed 0.30 s zero buffers — a warmup or health probe would look exactly
like this. The gate makes them free either way, but it is worth tracing rather
than assuming the VAD is at fault.

`min_audio_duration` is the same idea in the time axis: 0.10 s of *real* speech
also transcribes to `' '` on this endpoint (0.15 s gives `' We'`), so anything
shorter cannot produce text and need not be sent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # built lazily by __getattr__ below
    WhisperSTT: type
    DropBlankSTT: type

__all__ = ["WhisperSTT", "DropBlankSTT", "is_blank", "load_env"]

DEFAULT_PATH = "/v1/audio/transcriptions"
SILENCE_FLOOR_DBFS = -50.0
"""Conservative: real speech turns bottom out at -21.6 dBFS (p05) and the
non-speech ones sit at -240. Anything in between works; this leaves ~28 dB of
headroom under the quietest real turn measured."""

MIN_AUDIO_DURATION = 0.1
"""0.10 s of real speech already transcribes to `' '`, so shorter segments
cannot produce text and are not worth a request."""


def is_blank(text: str | None) -> bool:
    """
    True when a transcript carries no content.

    `str.strip()` is Unicode-aware, which matters here and is the reason this is
    not a regex. Four turns in the production sample — `' ஹலோ மாயா!'`,
    `' 你好,我想现在付款…'` — are real speech containing **no ASCII letters at
    all**, so an `[A-Za-z0-9]` test would silently discard Tamil and Chinese
    callers. Length after stripping does not.
    """
    return not text or not text.strip()


def load_env(env_file: str | Path | None = None) -> dict[str, str]:
    """
    Read `STT_*` settings from a `.env`, with the process environment winning.

    Reading the file never mutates `os.environ` — the rule `stt_api.evaluation`
    follows, for the same reason: a library that edits the process environment
    on import breaks whatever is hosting it.
    """
    keys = ("STT_URL", "STT_API", "STT_MODEL", "STT_LANGUAGE")
    path = Path(env_file) if env_file else Path(__file__).resolve().parents[3] / ".env"
    cfg: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            if k in keys and v.strip():
                cfg[k] = v.strip().strip('"').strip("'")
    for k in keys:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


@dataclass
class Counters:
    """Cheap observability: how often each guard fired."""

    requests: int = 0
    transcribed: int = 0
    blank_responses: int = 0
    """Requests that came back with nothing but whitespace."""
    skipped_quiet: int = 0
    """Segments below the RMS floor — no request was made."""
    skipped_short: int = 0
    """Segments below the duration floor — no request was made."""

    @property
    def skipped(self) -> int:
        return self.skipped_quiet + self.skipped_short

    def as_dict(self) -> dict[str, int]:
        return {**vars(self), "skipped": self.skipped}


def _import_livekit() -> Any:
    try:
        from livekit.agents import stt as lkstt  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError(
            "stt_api.livekit_plugin.whisper_stt needs livekit-agents. "
            f'Install the extra: pip install ".[livekit]". Import failed: {e}'
        ) from e
    return lkstt


def _rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -240.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-12))


def _make_classes() -> tuple[type, type]:
    import aiohttp
    from livekit import rtc
    from livekit.agents import (
        APIConnectionError,
        APIConnectOptions,
        APIStatusError,
        APITimeoutError,
        stt as lkstt,
    )
    from livekit.agents.types import NOT_GIVEN, NotGivenOr
    from livekit.agents.utils import AudioBuffer, is_given

    def _blank(language: str) -> Any:
        """
        A FINAL_TRANSCRIPT that carries nothing.

        One alternative with empty text, deliberately — not an empty list. See
        the module docstring: `audio_recognition.py:1207` indexes
        `alternatives[0]` without checking length.
        """
        return lkstt.SpeechEvent(
            type=lkstt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[lkstt.SpeechData(language=language, text="")],
        )

    class WhisperSTT(lkstt.STT):
        """
        Whisper behind an OpenAI-compatible endpoint, with silence guards.

            from stt_api.livekit_plugin.whisper_stt import WhisperSTT

            session = AgentSession(
                stt=WhisperSTT(),          # reads STT_URL / STT_API / STT_MODEL
                vad=silero.VAD.load(),     # required: this STT is not streaming
                llm=..., tts=...,
            )

        Non-streaming by design, so `AgentSession` wraps it in LiveKit's
        `StreamAdapter` and drives one request per VAD speech segment
        (`agent.py:522`). That is also why `vad=` is mandatory alongside it.
        """

        def __init__(
            self,
            *,
            base_url: str | None = None,
            api_key: str | None = None,
            model: str | None = None,
            language: str | None = None,
            prompt: str | None = None,
            temperature: float | None = None,
            path: str = DEFAULT_PATH,
            min_audio_duration: float = MIN_AUDIO_DURATION,
            silence_floor_dbfs: float | None = SILENCE_FLOOR_DBFS,
            timeout: float = 30.0,
            http_session: Any = None,
            env_file: str | Path | None = None,
        ) -> None:
            """
            Args:
                base_url: endpoint root, e.g. `https://host/proxy/stt`. Falls
                    back to `STT_URL`.
                api_key: bearer token. Falls back to `STT_API`.
                model: model id as the endpoint names it. Falls back to
                    `STT_MODEL`. Note the endpoint 404s with a *model* error
                    rather than an auth error if this holds the key by mistake.
                language: optional language hint (`ms`, `en`, `zh`, `ta`).
                min_audio_duration: skip the request for anything shorter, in
                    seconds. Shorter segments transcribe to `' '` anyway.
                silence_floor_dbfs: skip the request when the segment's RMS is
                    below this. `None` disables the check. The default skips
                    94 % of useless calls and no real speech, measured.
                timeout: per-request timeout, seconds.
                http_session: an existing `aiohttp.ClientSession` to borrow.
            """
            super().__init__(
                capabilities=lkstt.STTCapabilities(
                    streaming=False, interim_results=False
                )
            )
            cfg = load_env(env_file)
            self._base_url = (base_url or cfg.get("STT_URL", "")).rstrip("/")
            self._api_key = api_key or cfg.get("STT_API", "")
            self._model = model or cfg.get("STT_MODEL", "")
            self._language = language or cfg.get("STT_LANGUAGE") or ""
            if not self._base_url:
                raise ValueError("no STT endpoint: pass base_url= or set STT_URL")
            if not self._model:
                raise ValueError("no STT model: pass model= or set STT_MODEL")

            self._prompt = prompt
            self._temperature = temperature
            self._path = path
            self._min_audio_duration = min_audio_duration
            self._silence_floor_dbfs = silence_floor_dbfs
            self._timeout = timeout
            self._session = http_session
            self._owns_session = http_session is None
            self.counters = Counters()

        @property
        def model(self) -> str:
            return self._model

        @property
        def provider(self) -> str:
            return "stt-api/whisper"

        def _ensure_session(self) -> Any:
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                )
            return self._session

        async def aclose(self) -> None:
            if self._owns_session and self._session is not None:
                await self._session.close()
                self._session = None

        async def _recognize_impl(
            self,
            buffer: AudioBuffer,
            *,
            language: NotGivenOr[str] = NOT_GIVEN,
            conn_options: APIConnectOptions,
        ) -> Any:
            lang = language if is_given(language) else self._language
            frame = rtc.combine_audio_frames(buffer)
            pcm = np.frombuffer(frame.data, dtype=np.int16)
            duration = frame.samples_per_channel / frame.sample_rate

            # --- guard 1: never spend a request that cannot return words ----
            if duration < self._min_audio_duration:
                self.counters.skipped_short += 1
                return _blank(lang)
            if self._silence_floor_dbfs is not None:
                level = _rms_dbfs(pcm.astype(np.float32) / 32768.0)
                if level < self._silence_floor_dbfs:
                    self.counters.skipped_quiet += 1
                    return _blank(lang)

            form = aiohttp.FormData()
            form.add_field(
                "file",
                frame.to_wav_bytes(),
                filename="audio.wav",
                content_type="audio/wav",
            )
            form.add_field("model", self._model)
            form.add_field("response_format", "json")
            if lang:
                form.add_field("language", lang)
            if self._prompt:
                form.add_field("prompt", self._prompt)
            if self._temperature is not None:
                form.add_field("temperature", str(self._temperature))

            self.counters.requests += 1
            try:
                async with self._ensure_session().post(
                    self._base_url + self._path,
                    data=form,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=aiohttp.ClientTimeout(
                        total=conn_options.timeout or self._timeout
                    ),
                ) as resp:
                    if resp.status != 200:
                        raise APIStatusError(
                            (await resp.text())[:300], status_code=resp.status
                        )
                    payload = await resp.json()
            except APIStatusError:
                raise
            except TimeoutError as e:
                raise APITimeoutError() from e
            except Exception as e:
                raise APIConnectionError() from e

            text = payload.get("text") if isinstance(payload, dict) else None

            # --- guard 2: a blank answer must not read as user activity -----
            if is_blank(text):
                self.counters.blank_responses += 1
                return _blank(lang)

            self.counters.transcribed += 1
            detected = payload.get("language") if isinstance(payload, dict) else None
            return lkstt.SpeechEvent(
                type=lkstt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[lkstt.SpeechData(language=detected or lang, text=text)],
            )

    class DropBlankSTT(lkstt.STT):
        """
        Wrap any STT so whitespace-only transcripts stop interrupting.

            from livekit.plugins import openai
            from stt_api.livekit_plugin.whisper_stt import DropBlankSTT

            stt = DropBlankSTT(openai.STT())

        The same defect is in `livekit-plugins-openai`: `_recognize_impl` emits
        `resp.text` with no check at all, and its streaming path guards with
        `if transcript:` — which `' '` passes. This wrapper is for keeping an
        existing provider while getting the guard; `WhisperSTT` above is the
        native option.
        """

        def __init__(self, wrapped: Any) -> None:
            super().__init__(capabilities=wrapped.capabilities)
            self._wrapped = wrapped
            self.counters = Counters()

        @property
        def model(self) -> str:
            return getattr(self._wrapped, "model", "unknown")

        @property
        def provider(self) -> str:
            return getattr(self._wrapped, "provider", "unknown")

        async def _recognize_impl(
            self,
            buffer: AudioBuffer,
            *,
            language: NotGivenOr[str] = NOT_GIVEN,
            conn_options: APIConnectOptions,
        ) -> Any:
            ev = await self._wrapped.recognize(
                buffer, language=language, conn_options=conn_options
            )
            self.counters.requests += 1
            if not ev.alternatives or is_blank(ev.alternatives[0].text):
                self.counters.blank_responses += 1
                lang = ev.alternatives[0].language if ev.alternatives else ""
                return _blank(lang)
            self.counters.transcribed += 1
            return ev

        async def aclose(self) -> None:
            await self._wrapped.aclose()

    return WhisperSTT, DropBlankSTT


def __getattr__(name: str) -> Any:
    # Built on first access so importing the package does not require
    # livekit-agents: is_blank and load_env are useful on their own.
    if name in ("WhisperSTT", "DropBlankSTT"):
        _import_livekit()
        cls_w, cls_d = _make_classes()
        globals()["WhisperSTT"], globals()["DropBlankSTT"] = cls_w, cls_d
        return globals()[name]
    raise AttributeError(name)
