"""
The voice isolator as a LiveKit Agents audio filter.

    from livekit.agents.voice import room_io
    from stt_api.livekit_plugin.voice_isolator import LocalIsolator, VoiceIsolator

    backend = LocalIsolator()                     # once per process (prewarm)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                sample_rate=16000,                # the model's rate: no resampling
                noise_cancellation=lambda params: VoiceIsolator(backend=backend),
            ),
        ),
    )

`AudioInputOptions.noise_cancellation` takes any `rtc.FrameProcessor[rtc.AudioFrame]`:
an object `rtc.AudioStream` calls once per frame, synchronously, inside the agent
process, before the VAD, the turn detector and the STT see the frame. GTCRN in
`../noise_cancellation/` plugs in the same way.

LiveKit hands the audio over in 50 ms frames by default. The model's context does
not depend on them: every 10 ms hop it steps attends over the last 10 s of the call
and carries two LSTMs over all of it, and the speaker vector comes from the loudest
3 s heard so far. Those are the call's state, carried from frame to frame, so the
output is the same whatever the frame size.
"""

from __future__ import annotations

import logging

import numpy as np
from livekit import rtc
from livekit.rtc.audio_resampler import AudioResamplerQuality

from .backends import Backend, LocalIsolator
from .onnx_stream import SAMPLE_RATE

logger = logging.getLogger(__name__)

MODEL_SAMPLE_RATE = SAMPLE_RATE
"""The only rate the models run at. Other rates are resampled in and back out."""


class VoiceIsolator(rtc.FrameProcessor[rtc.AudioFrame]):
    """
    One instance per audio stream: the call's state (the attention rings, the
    LSTMs, the enrollment, or the server connection) is its own. The backend is
    shared: `LocalIsolator` holds one onnxruntime session per process,
    `RemoteIsolator` only an address.

    `max_lag_ms` bounds the delay a late server can build up: a frame that finds
    the output short is answered with silence in front of what there is (the
    stream is delayed, nothing is dropped), and if a burst then leaves more than
    `max_lag_ms` of output waiting, the oldest of it is dropped. With
    `LocalIsolator` neither happens: the output is ready when `_process` returns.

    A backend that cannot load or connect raises here, at session start, rather
    than passing audio through untouched; that silent degradation is the failure
    this replaces.
    """

    def __init__(
        self,
        backend: Backend | None = None,
        *,
        enabled: bool = True,
        max_lag_ms: float = 250.0,
    ) -> None:
        self._enabled = enabled
        self._backend = backend if backend is not None else LocalIsolator()
        self._stream = self._backend.open()
        self._max_lag_ms = max_lag_ms
        self._out16 = np.zeros(0, np.float32)
        self._native_rate: int | None = None
        self._down: rtc.AudioResampler | None = None
        self._up: rtc.AudioResampler | None = None
        self._out_native = np.zeros(0, np.float32)
        self._warned_channels = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def enrolled_at(self) -> float | None:
        """Seconds into the call when the live enrollment switched on (local
        backend; None before, and for a remote call, whose enrollment is the
        server's)."""
        return getattr(self._stream, "enrolled_at", None)

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        n, ch, rate = frame.samples_per_channel, frame.num_channels, frame.sample_rate
        if ch != 1:
            if not self._warned_channels:
                self._warned_channels = True
                logger.warning(
                    "VoiceIsolator expects mono audio, got %d channels: passing through",
                    ch,
                )
            return frame
        if rate == MODEL_SAMPLE_RATE:
            pcm = (
                np.frombuffer(frame.data, dtype=np.int16, count=n).astype(np.float32)
                / 32768.0
            )
            self._out16 = np.concatenate([self._out16, self._stream.push(pcm)])
            y, self._out16 = self._take(self._out16, n, rate)
        else:
            self._ensure_resamplers(rate)
            for out in self._down.push(frame):  # type: ignore[union-attr]
                a = np.frombuffer(
                    out.data, dtype=np.int16, count=out.samples_per_channel
                )
                self._emit_up(self._stream.push(a.astype(np.float32) / 32768.0))
            y, self._out_native = self._take(self._out_native, n, rate)
        pcm_out = np.clip(np.rint(y * 32768.0), -32768, 32767).astype(np.int16)
        return rtc.AudioFrame(pcm_out.tobytes(), rate, ch, n)

    def _take(
        self, buf: np.ndarray, n: int, rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if buf.size < n:
            buf = np.concatenate([np.zeros(n - buf.size, np.float32), buf])
        head, rest = buf[:n], buf[n:]
        cap = int(self._max_lag_ms * rate / 1000)
        if rest.size > cap:
            rest = rest[rest.size - cap :]
        return head, rest

    def _emit_up(self, y: np.ndarray) -> None:
        if not y.size:
            return
        pcm = np.clip(np.rint(y * 32768.0), -32768, 32767).astype(np.int16)
        for out in self._up.push(bytearray(pcm.tobytes())):  # type: ignore[union-attr]
            a = np.frombuffer(out.data, dtype=np.int16, count=out.samples_per_channel)
            self._out_native = np.concatenate(
                [self._out_native, a.astype(np.float32) / 32768.0]
            )

    def _ensure_resamplers(self, rate: int) -> None:
        if self._native_rate == rate:
            return
        # GTCRN's choices, for latency: LOW down (a real anti-alias filter,
        # 12.5 ms), QUICK up (the signal is already band-limited to 8 kHz).
        self._native_rate = rate
        self._down = rtc.AudioResampler(
            rate, MODEL_SAMPLE_RATE, num_channels=1, quality=AudioResamplerQuality.LOW
        )
        self._up = rtc.AudioResampler(
            MODEL_SAMPLE_RATE, rate, num_channels=1, quality=AudioResamplerQuality.QUICK
        )
        self._out_native = np.zeros(0, np.float32)

    def _close(self) -> None:
        self._stream.close()
        self._out16 = np.zeros(0, np.float32)
        self._out_native = np.zeros(0, np.float32)
        self._down = self._up = None
