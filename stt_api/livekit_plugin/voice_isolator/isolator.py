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

QUIET = 10 ** (-45 / 20)
"""Peak of a 10 ms piece of isolated audio that may be dropped to win back delay:
the output in the caller's pauses, where the model has removed everything else."""


class VoiceIsolator(rtc.FrameProcessor[rtc.AudioFrame]):
    """
    One instance per audio stream: the call's state (the attention rings, the
    LSTMs, the enrollment, or the server connection) is its own. The backend is
    shared: `LocalIsolator` holds one onnxruntime session per process,
    `RemoteIsolator` only an address.

    `max_lag_ms` bounds the delay a late server can build up. A frame that finds
    the output short is answered with silence in front of what there is (the
    stream is delayed by that much, nothing is lost) while the delay stays within
    `max_lag_ms`; past it, the part not back yet goes out as the caller's own
    audio, unfiltered, and its isolated version is dropped when it comes. So an
    overloaded server degrades the call towards no filter rather than towards an
    agent that hears the caller later and later. Delay a stall left behind is won
    back in the caller's pauses: while more isolated audio than needed has waited
    for 2 s, a quiet 10 ms piece is dropped per frame. With `LocalIsolator` none
    of this happens: the output is ready when `_process` returns.

    `wait_ms` (remote backends) lets `_process` wait that long for the server's
    reply to the frame it just sent, instead of returning what came back for
    earlier frames: the reply then goes out with its own frame rather than the next
    one, so the delay drops by about a frame, at the cost of holding the agent's
    audio loop for up to `wait_ms` per frame (the CPU model holds it for its
    compute); never more than 60% of a frame. 0 never waits. `wait_ms` and
    `max_lag_ms` default to the backend's (`RemoteIsolator`: wait 30 ms, delay at
    most 40 ms; measured through LiveKit agents, 32 calls on one GPU in 50 ms
    frames came back 40 ms behind instead of 90, with 0.06% of the audio passed
    unfiltered). While waiting, the delay a late reply added is won
    back by probing: after 2 s without a late frame, one frame waits for 10 ms more
    than it needs, and if that piece comes back in time and is quiet, it is
    dropped. A probe that a late frame soon follows doubles the time to the next
    (up to 32 s), so the delay settles instead of swinging: every late frame puts
    a little silence into the stream.

    A backend that cannot load or connect raises here, at session start, rather
    than passing audio through untouched; that silent degradation is the failure
    this replaces.
    """

    def __init__(
        self,
        backend: Backend | None = None,
        *,
        enabled: bool = True,
        max_lag_ms: float | None = None,
        wait_ms: float | None = None,
    ) -> None:
        self._enabled = enabled
        self._backend = backend if backend is not None else LocalIsolator()
        self._stream = self._backend.open()
        if wait_ms is None:
            wait_ms = getattr(self._backend, "wait_ms", 0.0)
        self._wait_s = wait_ms / 1000.0 if hasattr(self._stream, "wait") else 0.0
        if (
            max_lag_ms is None
        ):  # a cap that fits only a stream whose frames wait for replies
            max_lag_ms = (
                getattr(self._backend, "max_lag_ms", 250.0) if self._wait_s else 250.0
            )
        self._max_lag_ms = max_lag_ms
        # at the frames' rate: isolated audio not yet sent out, the input of the
        # last max_lag_ms + one frame, isolated samples to drop because their time
        # went out unfiltered, and the delay in samples
        self._out = np.zeros(0, np.float32)
        self._hist = np.zeros(0, np.float32)
        self._skip = 0
        self._lag = 0
        self._waits: list[int] = []  # isolated samples left waiting, last 2 s of frames
        self._calm = (
            0  # frames since one came back short (a wait that ran out), when waiting
        )
        self._probe_s = (
            2.0  # how long a stretch without a late frame before the next probe
        )
        self._since_drop = 1 << 30  # frames since a probe last won back a hop
        # audio that went out unfiltered because its isolated version came late
        self._unfiltered_s = 0.0
        self._rate = MODEL_SAMPLE_RATE
        self._native_rate: int | None = None
        self._down: rtc.AudioResampler | None = None
        self._up: rtc.AudioResampler | None = None
        self._warned_channels = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def delay_ms(self) -> float:
        """How far the output stands behind the input right now: the model's 10 ms
        look-ahead (and the resamplers' delay off 16 kHz) plus, with a server, the
        silence put in front of replies that were not back in time (the round trip
        and the server's step), at most `max_lag_ms`."""
        return 1000.0 * (self._stream.delay_s + self._lag / self._rate)

    @property
    def unfiltered_ms(self) -> float:
        """How much of the call so far went out as the caller's own audio because
        its isolated version was later than `max_lag_ms` (0 with `LocalIsolator`)."""
        return 1000.0 * self._unfiltered_s

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
        raw = (
            np.frombuffer(frame.data, dtype=np.int16, count=n).astype(np.float32)
            / 32768.0
        )
        if rate == MODEL_SAMPLE_RATE:
            self._rate = rate
            self._out = np.concatenate([self._out, self._stream.push(raw)])
        else:
            self._ensure_resamplers(rate)
            for out in self._down.push(frame):  # type: ignore[union-attr]
                a = np.frombuffer(
                    out.data, dtype=np.int16, count=out.samples_per_channel
                )
                self._emit_up(self._stream.push(a.astype(np.float32) / 32768.0))
        wait = getattr(self._stream, "wait", None)
        probe = False
        if self._wait_s > 0 and wait is not None:
            hop = rate // 100
            # after 2 s without a late frame, ask for one hop more than this frame needs
            probe = self._lag > 0 and self._calm >= max(
                1, round(self._probe_s * rate / n)
            )
            short = n + self._skip + (hop if probe else 0) - self._out.size
            if short > 0:
                limit = min(self._wait_s, 0.6 * n / rate)  # never most of a frame
                got = wait(int(np.ceil(short * MODEL_SAMPLE_RATE / rate)), limit)
                if rate == MODEL_SAMPLE_RATE:
                    self._out = np.concatenate([self._out, got])
                else:
                    self._emit_up(got)
        y = self._take(raw, rate, probe)
        pcm_out = np.clip(np.rint(y * 32768.0), -32768, 32767).astype(np.int16)
        return rtc.AudioFrame(pcm_out.tobytes(), rate, ch, n)

    def _take(self, raw: np.ndarray, rate: int, probe: bool = False) -> np.ndarray:
        """This frame's output: the isolated audio `_lag` samples behind the input.
        `probe`: the frame waited for one hop more than it needed; if that hop came
        and the first one waiting is quiet, it is dropped (10 ms less delay)."""
        n = raw.size
        cap = int(self._max_lag_ms * rate / 1000)
        keep = cap + n
        h = np.concatenate([self._hist, raw])
        self._hist = h[-keep:] if h.size >= keep else np.pad(h, (keep - h.size, 0))
        buf = self._out
        if self._skip:  # isolated audio for time that already went out unfiltered
            k = min(self._skip, buf.size)
            buf, self._skip = buf[k:], self._skip - k
        hop = rate // 100
        if buf.size >= n:
            if probe:
                if buf.size >= n + hop and np.abs(buf[:hop]).max() < QUIET:
                    buf, self._lag, self._calm = buf[hop:], self._lag - hop, 0
                    self._since_drop = 0
                elif buf.size < n + hop:  # the server could not give the hop in time
                    self._calm = 0
            self._calm += 1
            self._since_drop += 1
            if self._since_drop * n > 8 * self._probe_s * rate and self._probe_s > 2.0:
                self._probe_s, self._since_drop = (
                    self._probe_s / 2,
                    0,
                )  # long settled: probe sooner
            y, buf = buf[:n], buf[n:]
        else:
            if (
                self._since_drop * n < 2 * self._probe_s * rate
            ):  # the last probe went too far
                self._probe_s = min(2 * self._probe_s, 32.0)
            self._calm = 0
            short = n - buf.size
            pad = min(short, max(0, cap - self._lag))  # silence in front: more delay
            late = short - pad  # past max_lag_ms: the input itself
            self._lag += pad
            end = keep - self._lag
            y = np.concatenate(
                [np.zeros(pad, np.float32), buf, self._hist[end - late : end]]
            )
            self._skip += late
            self._unfiltered_s += late / rate
            buf = buf[:0]
        # more delay than the backend needs (left by a stall): if at least two hops
        # have been waiting after every frame for 2 s, drop one quiet hop
        span = max(1, round(2.0 * rate / n))
        self._waits = self._waits[1 - span :] + [buf.size] if span > 1 else [buf.size]
        if (
            len(self._waits) == span
            and min(self._waits) > 2 * hop
            and np.abs(buf[:hop]).max() < QUIET
        ):
            buf, self._lag = buf[hop:], self._lag - hop
            self._waits = [w - hop for w in self._waits]
        self._out = buf
        return y

    def _emit_up(self, y: np.ndarray) -> None:
        if not y.size:
            return
        pcm = np.clip(np.rint(y * 32768.0), -32768, 32767).astype(np.int16)
        for out in self._up.push(bytearray(pcm.tobytes())):  # type: ignore[union-attr]
            a = np.frombuffer(out.data, dtype=np.int16, count=out.samples_per_channel)
            self._out = np.concatenate([self._out, a.astype(np.float32) / 32768.0])

    def _ensure_resamplers(self, rate: int) -> None:
        if self._native_rate == rate:
            return
        # GTCRN's choices, for latency: LOW down (a real anti-alias filter,
        # 12.5 ms), QUICK up (the signal is already band-limited to 8 kHz).
        self._native_rate = self._rate = rate
        self._down = rtc.AudioResampler(
            rate, MODEL_SAMPLE_RATE, num_channels=1, quality=AudioResamplerQuality.LOW
        )
        self._up = rtc.AudioResampler(
            MODEL_SAMPLE_RATE, rate, num_channels=1, quality=AudioResamplerQuality.QUICK
        )
        self._out, self._hist = np.zeros(0, np.float32), np.zeros(0, np.float32)
        self._skip = self._lag = self._calm = 0
        self._probe_s, self._since_drop = 2.0, 1 << 30
        self._waits = []

    def _close(self) -> None:
        self._stream.close()
        self._out, self._hist = np.zeros(0, np.float32), np.zeros(0, np.float32)
        self._down = self._up = None
