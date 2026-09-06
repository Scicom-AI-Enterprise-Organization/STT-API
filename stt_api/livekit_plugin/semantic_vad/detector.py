"""
A semantic VAD wired into LiveKit's audio-native end-of-turn interface.

LiveKit ships two unrelated EoT paths, and choosing the wrong one is the main
way this integration goes wrong:

    livekit-plugins-turn-detector      transcript text  -> ONNX over ChatContext
    livekit.agents.inference.eot       streaming audio  -> TurnDetector(...).stream()

Only the second can carry an audio-native model, and it is what this module
plugs into. `TurnDetector(version="v1")` streams audio over a **websocket** to
LiveKit's cloud gateway as protobuf; `version="v1-mini"` runs a ~108 MB ctypes
model in-process instead.

Neither is what a self-hosted semantic VAD wants, and the useful discovery is
that neither is required. The seam between them is
`_StreamingTurnDetectionTransport` — a seven-method Protocol (`run`,
`run_inference`, `push_frame`, `flush`, `attach`, `detach`, `session_id`).
Implement that and everything above it, including the audio ingress,
resampling to 16 kHz and request bookkeeping in
`_BaseStreamingTurnDetectorStream`, stays stock. **There is no need to
reimplement LiveKit's protobuf websocket server to self-host an EoT model.**

One number to keep in mind: the stock local transport keeps a **1.2 second**
rolling buffer (`_CLIENT_BUFFER_SECONDS`). That is right for the mini model and
much too short for a semantic model reasoning about a whole utterance, so the
buffer here is sized from the backend's own `window_seconds` (8 s for
smart-turn-v3) rather than inherited.
"""

from __future__ import annotations

import asyncio
import time
import weakref
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # the class is built lazily by __getattr__ below
    SemanticVAD: type

__all__ = ["SemanticVAD"]

_TIMEOUT_HEADROOM = 1.0
"""LiveKit abandons a prediction after DEFAULT_PREDICTION_TIMEOUT = 1.0 s and,
if `local_fallback` is on, degrades to the mini model for the rest of the
session — stickily, it never comes back. Backends should stay well inside it."""


def _import_livekit() -> tuple[Any, ...]:
    """
    Import the private EoT internals, with a comprehensible failure.

    These live under `livekit.agents.inference.eot` and are not public API. That
    is a real risk and it is better stated than hidden: a LiveKit upgrade can
    move them, and the failure should say so rather than surfacing as an
    AttributeError three frames deep. Verified against livekit-agents 1.7.x.
    """
    try:
        from livekit.agents.inference.eot.base import (
            DEFAULT_SAMPLE_RATE,
            TurnDetectorOptions,
            _BaseStreamingTurnDetector,
            _BaseStreamingTurnDetectorStream,
        )
        from livekit.agents.inference.eot.languages import ThresholdOptions
    except ImportError as e:  # pragma: no cover - depends on the installed version
        raise ImportError(
            "stt_api.livekit_plugin.semantic_vad needs livekit-agents with the "
            "audio EoT interface (livekit.agents.inference.eot, 1.7+). "
            f"Import failed: {e}"
        ) from e
    return (
        DEFAULT_SAMPLE_RATE,
        TurnDetectorOptions,
        _BaseStreamingTurnDetector,
        _BaseStreamingTurnDetectorStream,
        ThresholdOptions,
    )


class _SemanticTransport:
    """
    Holds the trailing audio and answers `inference_start` from it.

    LiveKit only calls `run_inference` at genuine pause candidates (its VAD has
    already seen ~200 ms of silence), so this never polls mid-word — which
    matters, because a semantic model asked mid-syllable has nothing useful to
    say.
    """

    def __init__(self, *, backend: Any, sample_rate: int) -> None:
        from livekit.agents import utils

        self._backend = backend
        self._sample_rate = sample_rate
        self._buf = utils.AudioArrayBuffer(
            buffer_size=int(backend.window_seconds * sample_rate),
            sample_rate=sample_rate,
        )
        self._stream_ref: weakref.ref | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def session_id(self) -> str | None:
        return None

    def attach(self, stream: Any) -> None:
        # Weak, exactly as LiveKit's own transport does: the stream owns the
        # transport, and a strong reference back would keep finished sessions
        # alive for the life of the process.
        self._stream_ref = weakref.ref(stream)

    def detach(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()

    def push_frame(self, frame: Any) -> None:
        self._buf.push_frame(frame)

    def flush(self) -> None:
        # Turn boundary: drop the previous speaker's audio rather than letting it
        # bleed into the next turn's decision.
        if len(self._buf) > 0:
            self._buf.shift(len(self._buf))

    def run_inference(self, request_id: str) -> None:
        task = asyncio.create_task(self._predict(request_id, self._buf.read()))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _predict(self, request_id: str, pcm: np.ndarray) -> None:
        prob = 0.0
        started = time.monotonic()
        try:
            # Off the event loop: ONNX and HTTP both block, and stalling the loop
            # would delay the audio ingress that feeds every other prediction.
            prob = float(await asyncio.to_thread(self._backend.predict, pcm))
        except Exception:
            from livekit.agents.log import logger

            # 0.0 means "not finished", so a broken backend makes the agent wait
            # for the endpointing timeout instead of interrupting the caller.
            # Failing toward silence is the safe direction for a voice agent.
            logger.exception("semantic VAD prediction failed; treating as hold")
        duration = time.monotonic() - started

        stream = self._stream_ref() if self._stream_ref is not None else None
        if stream is None:
            return
        stream._resolve_prediction(request_id, prob, inference_duration=duration)

    async def run(self) -> None:
        stream = self._stream_ref() if self._stream_ref is not None else None
        if stream is None:
            return
        await stream._drain_audio_channel()


def _make_detector_class() -> type:
    (
        DEFAULT_SAMPLE_RATE,
        TurnDetectorOptions,
        _BaseStreamingTurnDetector,
        _BaseStreamingTurnDetectorStream,
        ThresholdOptions,
    ) = _import_livekit()

    class SemanticVAD(_BaseStreamingTurnDetector):
        """
        Audio-native end-of-turn detection, self-hosted.

            from stt_api.livekit_plugin.semantic_vad import SemanticVAD, SmartTurnV3

            session = AgentSession(
                stt=..., llm=..., tts=...,
                vad=ctx.proc.userdata["vad"],
                turn_detection=SemanticVAD(backend=SmartTurnV3()),
            )

        The VAD is still required: LiveKit's VAD is what decides *when* to ask,
        and this model is what answers. They are not alternatives.

        `unlikely_threshold` is the bar for committing a turn. Higher waits
        longer and interrupts less. Calibrate it on your own audio — the default
        inherited here comes from LiveKit's mini model and is not tuned for
        whichever backend you plug in.
        """

        def __init__(
            self,
            *,
            backend: Any,
            unlikely_threshold: float | None = None,
            sample_rate: int = DEFAULT_SAMPLE_RATE,
            local_fallback: bool = False,
        ) -> None:
            from livekit.agents.types import NOT_GIVEN

            self._backend = backend
            thresholds = ThresholdOptions(
                "turn-detector-v1-mini",
                unlikely_threshold if unlikely_threshold is not None else NOT_GIVEN,
            )
            super().__init__(
                opts=TurnDetectorOptions(sample_rate=sample_rate, thresholds=thresholds)
            )
            # Off by default: LiveKit's fallback is a *sticky* one-way degrade to
            # its own mini model, which would silently replace the model you
            # deployed for the rest of the session and pull ~108 MB of weights to
            # do it. Opt in only if you would rather have their model than the
            # endpointing timeout.
            self._local_fallback = local_fallback

        @property
        def model(self) -> str:
            return f"semantic-vad/{type(self._backend).__name__}"

        @property
        def provider(self) -> str:
            return "stt-api"

        async def supports_language(self, language: Any) -> bool:
            # An audio model has no per-language vocabulary to miss. Claiming
            # support for everything is right here and matters: LiveKit skips a
            # detector entirely for languages it does not claim, which is how the
            # stock multilingual text model ends up unused on `ms`.
            return True

        def stream(self, *, conn_options: Any = None) -> Any:
            transport = _SemanticTransport(
                backend=self._backend, sample_rate=self._opts.sample_rate
            )
            return _BaseStreamingTurnDetectorStream(
                detector=self,
                opts=self._opts,
                transport=transport,
                model="turn-detector-v1-mini",
                local_fallback=self._local_fallback,
            )

    return SemanticVAD


def __getattr__(name: str) -> Any:
    # Built on first access so that merely importing the package does not require
    # livekit-agents — the backends are useful on their own, for benchmarking.
    if name == "SemanticVAD":
        cls = _make_detector_class()
        globals()["SemanticVAD"] = cls
        return cls
    raise AttributeError(name)
