# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Clearline-VAD as a LiveKit Agents `agents.vad.VAD`, a drop-in for `silero.VAD`.

Adapted from livekit-plugins-silero 1.8.9 (`livekit/plugins/silero/vad.py`,
Apache-2.0): the stream, its events, the speech buffer and prefix padding are
upstream's, unchanged. What differs is the model (`onnx_model.py`, weights from
`hub.py`), a fixed 16 kHz inference rate (any input rate is resampled to it), the
ExpFilter alpha as an option (upstream hardcodes 0.35), the default operating
point, and inference run inline: a 0.06 ms call costs less than the thread-pool
hop upstream takes for every window.

Like `../pulse_vad/`, this answers "is there speech right now", not "has the
speaker finished": an `AgentSession` still wants a turn detector beside it.

The stream is not upstream's any more. Pending samples sit in one numpy buffer and a
window is resampled in numpy (8, 24, 32 and 48 kHz input; any other rate, or
`inline=False`, takes upstream's loop), and streams of one VAD that are pushed the
same frame objects, as an AgentSession with a non-streaming STT pushes its own VAD
stream and its StreamAdapter's, share one inference; each keeps its own filter,
state machine, speech buffer and events. `legacy_stream=True` is upstream's loop.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from livekit import agents, rtc
from livekit.agents import utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from . import hub, onnx_model

logger = logging.getLogger(__name__)

__all__ = ["ClearlineVAD", "ClearlineVADStream"]

SLOW_INFERENCE_THRESHOLD = 0.2  # late by 200ms


@dataclass
class _VADOptions:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    deactivation_threshold: float
    sample_rate: int
    smoothing_alpha: float
    inline: bool
    legacy_stream: bool = False


# input rates the stream resamples itself: rate -> (L, M) with rate * L / M = 16 kHz, so that one window takes a whole
# number of input samples
_FAST_RATES = {8000: (2, 1), 16000: (1, 1), 24000: (2, 3), 32000: (1, 2), 48000: (1, 3)}
# the low-pass: zero crossings of its sinc on each side and its cut-off as a share of the lower of the two rates.
# Down from 24, 32 or 48 kHz the cut-off is 16 kHz's own Nyquist: flat to 6.5 kHz, 0.6 dB down at 7, 6 at 8 kHz,
# 44 at 9.5 and 70 at 10, so the little that folds back lands in the top mel band well under what was there. A lower cut-off
# keeps that band cleaner and thinner, and cost a far-field set 0.002 of frame AUC (scripts/vad_rate_check.py); with
# this one the plugin ranks frames as on 16 kHz audio itself. Up from 8 kHz: flat to 3.4 kHz, the image past
# 4.3 kHz 50 dB down and more.
_LOWPASS_DOWN = (10, 0.5)
_LOWPASS_UP = (16, 0.48)
_KAISER_BETA = (
    7.0  # its window: about 70 dB down in the stop band, far more than a VAD needs
)
_PROBATION = 8  # frames after it starts (or is flushed) on each of which a stream looks for an engine to share
_RECHECK = 8  # and, while it is alone on its engine, every so many frames after


class _Resampler:
    """One window's worth of int16 samples at the input rate -> the window at 16 kHz in float32, scaled as upstream
    scales it (/ 32767). A polyphase FIR (a Kaiser-windowed sinc under the lower Nyquist): output n is
    sum_i g[(n M) mod L][i] x[floor(n M / L) - i], so each phase is one matrix-vector product over windows of the
    input a fixed stride apart. The last taps - 1 input samples carry over, so windows join without a seam."""

    def __init__(self, rate: int, window: int) -> None:
        self.L, self.M = L, M = _FAST_RATES[rate]
        self.need = window * M // L
        self.out = np.empty(window, np.float32)
        self.phases: list[tuple[int, int, np.ndarray]] = []
        if L == M:
            return
        up = rate * L
        crossings, share = (
            _LOWPASS_DOWN if rate > onnx_model.SAMPLE_RATE else _LOWPASS_UP
        )
        fc = share * min(rate, onnx_model.SAMPLE_RATE)
        half = int(np.ceil(crossings * up / (2 * fc)))
        k = np.arange(-half, half + 1)
        h = np.sinc(2 * fc / up * k) * np.kaiser(2 * half + 1, _KAISER_BETA)
        self.taps = taps = -(-len(h) // L)
        for ph in range(L):
            g = np.zeros(taps)
            sub = h[ph::L]
            g[: len(sub)] = sub
            g /= (
                g.sum() * np.iinfo(np.int16).max
            )  # unit gain at DC in every phase; int16 -> [-1, 1]
            n0 = next(n for n in range(L) if (n * M) % L == ph)
            # reversed: a row of the sliding view is x[e - taps + 1 .. e], oldest first
            self.phases.append(
                (n0, (n0 * M) // L, np.ascontiguousarray(g[::-1], np.float32))
            )
        self._xb = np.zeros(taps - 1 + self.need, np.float32)
        self._view = np.lib.stride_tricks.sliding_window_view(
            self._xb, taps
        )  # row e: the taps samples ending at e

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.phases:
            np.divide(x, np.iinfo(np.int16).max, out=self.out, dtype=np.float32)
            return self.out
        xb, t1 = self._xb, self.taps - 1
        xb[:t1] = xb[self.need :]  # the tail of the window before
        xb[t1:] = x
        L, M = self.L, self.M
        for n0, e0, g in self.phases:
            self.out[n0::L] = np.ascontiguousarray(self._view[e0::M]) @ g
        return self.out


class _Engine:
    """Frames to probabilities, once, for every stream of a VAD that is pushed the same frames: the pending input,
    the resampler, the model and its state. It remembers the last frames it took (by identity) so that another
    stream of the same VAD can tell it is being fed the same audio, and join."""

    RING = 128

    def __init__(self, vad: ClearlineVAD, rate: int) -> None:
        vad._born += 1
        self.born, self.rate = vad._born, rate
        self.model = onnx_model.OnnxModel(onnx_session=vad._onnx_session)
        self.window = self.model.window_size_samples
        self.window_duration = self.window / onnx_model.SAMPLE_RATE
        self.rs = _Resampler(rate, self.window)
        self.need = self.rs.need
        self._pend = np.empty(self.need + rate, np.int16)
        self._n = 0
        self.count = 0  # frames taken
        self.ring: list[rtc.AudioFrame | None] = [None] * self.RING
        # id(frame) -> which frame it was, for the frames in the ring (they are alive, so no id repeats)
        self.at: dict[int, int] = {}
        self.subs: list[ClearlineVADStream] = []
        # the last windows, each with the frame it was completed by: a stream that joins at a frame this engine took
        # a moment ago gets the windows completed since, so two streams started on the same frame emit the same events
        # whichever task ran first
        self.recent: collections.deque[
            tuple[int, float, np.ndarray, rtc.AudioFrame, float]
        ] = collections.deque(maxlen=32)
        self._vad = vad
        self._extra = 0.0

    def index_of(self, frame: rtc.AudioFrame) -> int:
        """Which of the frames taken (0 = the first) this one is, among the last RING; -1 if none."""
        # one lookup, not a scan: a VAD may carry many unrelated streams, each asking every older engine
        k = self.at.get(id(frame), -1)
        return k if k >= 0 and self.ring[k % self.RING] is frame else -1

    def last(self) -> rtc.AudioFrame | None:
        return self.ring[(self.count - 1) % self.RING] if self.count else None

    def take(self, frame: rtc.AudioFrame) -> None:
        """A new frame: every window it completes goes to every stream on this engine."""
        slot = self.count % self.RING
        old = self.ring[slot]
        if old is not None and self.at.get(id(old)) == self.count - self.RING:
            # leaving the ring (unless the same object was taken again since)
            del self.at[id(old)]
        self.ring[slot] = frame
        self.at[id(frame)] = self.count
        self.count += 1
        data = np.frombuffer(frame.data, dtype=np.int16)
        n = self._n + len(data)
        if n > len(self._pend):  # a frame longer than a second: grow
            self._pend = np.concatenate((self._pend[: self._n], np.empty(n, np.int16)))
        self._pend[self._n : n] = data
        self._n = n
        need = self.need
        while self._n >= need:
            start = time.perf_counter()
            win = (
                self._pend[:need].copy()
            )  # the window's input samples: the events and the speech buffers keep them
            p = self.model(self.rs(win))
            self._vad.windows_inferred += 1
            rest = self._n - need
            if rest:
                self._pend[:rest] = self._pend[need : self._n]
            self._n = rest
            duration = time.perf_counter() - start
            self._extra = max(0.0, self._extra + duration - self.window_duration)
            if duration > SLOW_INFERENCE_THRESHOLD:
                logger.warning(
                    "VAD inference is slower than realtime",
                    extra={"delay": self._extra},
                )
            out = rtc.AudioFrame(
                data=win.tobytes(),
                sample_rate=self.rate,
                num_channels=1,
                samples_per_channel=need,
            )
            self.recent.append((self.count - 1, p, win, out, duration))
            for s in tuple(self.subs):
                try:
                    s._on_window(p, win, out, duration)
                except Exception:  # one stream's trouble is not the others'
                    logger.exception("Clearline-VAD: a stream failed on a window")


class ClearlineVAD(agents.vad.VAD):
    """
    Clearline Voice Activity Detection, a drop-in for `livekit.plugins.silero.VAD`.

        from stt_api.livekit_plugin import ClearlineVAD

        def prewarm(proc):
            proc.userdata["vad"] = ClearlineVAD.load()

    `load()` is blocking (it may download the weights): call it in `prewarm`, as
    with silero.
    """

    @classmethod
    def load(
        cls,
        *,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.55,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        window: int = 512,
        smoothing_alpha: NotGivenOr[float] = NOT_GIVEN,
        variant: str = "release",
        inline: bool = True,
        legacy_stream: bool = False,
        num_threads: int = 1,
        onnx_file_path: NotGivenOr[Path | str] = NOT_GIVEN,
        revision: str | None = None,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
        # deprecated
        padding_duration: NotGivenOr[float] = NOT_GIVEN,
    ) -> ClearlineVAD:
        """
        Load Clearline-VAD and prepare it for streaming.

        Args:
            min_speech_duration: speech this long opens a turn.
            min_silence_duration: silence this long closes one.
            prefix_padding_duration: audio kept before the detected onset.
            max_buffered_speech: cap on buffered speech, in seconds.
            activation_threshold: p(speech) to count a window as speech. **Not
                silero's 0.5**: the default is the operating point chosen on
                development data for the pinned graph (`hub.OPERATING_POINT`: 0.825
                for the release at 512-sample windows, 0.875 at 256; 0.825 and 0.75
                for the distilled variant); the two models' probabilities are calibrated
                differently. On telephone legs
                with a busy room behind the caller, raise it (0.9-0.95).
            window: samples per inference at 16 kHz: 512 (32 ms, silero's cadence)
                or 256 (16 ms updates, the same weights).
            smoothing_alpha: LiveKit's ExpFilter alpha on p, **LiveKit's
                convention**: the weight of the *previous* value, so 1.0 would
                freeze the filter and is refused; 0.0 turns it off. Default 0.35,
                chosen with the threshold. (`PulseVAD`'s option reads the other
                way round; do not copy a value across.)
            variant: "release" (the default: best LiveKit turn-taking on the
                development sets) or "distilled" (the better frame classifier, a
                larger graph: level with silero through telephone codecs and ahead
                of it in noisy rooms, ends turns slightly later, about 1.4 times the
                release's CPU; for offline segmentation and noisy rooms).
            inline: run inference on the event loop instead of the thread pool.
            legacy_stream: upstream's stream loop for every stream: LiveKit's resampler
                once per frame, and one inference per stream even when two streams of
                this VAD are pushed the same frames (an AgentSession with a
                non-streaming STT does that). Off by default.
            num_threads: ONNX Runtime intra-op threads; 1 is right for this size.
            onnx_file_path: a local graph instead of the pinned download (also
                `CLEARLINE_VAD_ONNX_PATH`). Its window must match `window`.
            revision: a Hugging Face revision instead of the pinned one. The default
                operating point belongs to the pinned weights.
            deactivation_threshold: exit threshold while speaking. Default
                max(activation_threshold - 0.15, 0.01), LiveKit's gap.
            padding_duration: **deprecated**, use `prefix_padding_duration`.
        """
        if (variant, window) not in hub.VAD_FILES:
            raise ValueError(
                f"no Clearline-VAD graph for variant={variant!r}, window={window}; "
                f"have {sorted(hub.VAD_FILES)}"
            )
        point = hub.OPERATING_POINT[(variant, window)]
        if not is_given(activation_threshold):
            activation_threshold = point["activation_threshold"]
        if not is_given(smoothing_alpha):
            smoothing_alpha = point["smoothing_alpha"]
        if not 0.0 <= smoothing_alpha < 1.0:
            raise ValueError(
                "smoothing_alpha must be in [0, 1): 1.0 would freeze LiveKit's ExpFilter"
            )

        if is_given(padding_duration):
            logger.warning(
                "padding_duration is deprecated and will be removed in 1.5.0, use prefix_padding_duration instead",  # noqa: E501
            )
            prefix_padding_duration = padding_duration

        if is_given(deactivation_threshold) and deactivation_threshold <= 0:
            raise ValueError("deactivation_threshold must be greater than 0")

        path = hub.vad_model_path(
            variant,
            window,
            onnx_file_path if is_given(onnx_file_path) else None,
            revision,
        )
        session = onnx_model.new_inference_session(path, num_threads)
        got = int(session.get_inputs()[0].shape[1])
        if got != window:
            raise ValueError(f"{path} takes {got}-sample windows, not window={window}")
        opts = _VADOptions(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            deactivation_threshold=deactivation_threshold
            or max(activation_threshold - 0.15, 0.01),
            sample_rate=onnx_model.SAMPLE_RATE,
            smoothing_alpha=smoothing_alpha,
            inline=inline,
            legacy_stream=legacy_stream,
        )
        vad = cls(session=session, opts=opts)
        vad._variant = variant
        return vad

    def __init__(
        self,
        *,
        session,
        opts: _VADOptions,
    ) -> None:
        window = int(session.get_inputs()[0].shape[1])
        super().__init__(
            capabilities=agents.vad.VADCapabilities(
                update_interval=window / onnx_model.SAMPLE_RATE
            )
        )
        self._onnx_session = session
        self._opts = opts
        self._streams = weakref.WeakSet[ClearlineVADStream]()
        self._engines: list[
            _Engine
        ] = []  # one per audio source being heard (see _Engine)
        self._born = 0
        self.windows_inferred = 0  # model calls made for this VAD's streams (a stream that shares makes none)
        self._variant = "release"

    @property
    def model(self) -> str:
        return f"clearline-vad-{self._variant}"

    @property
    def provider(self) -> str:
        return "ONNX"

    def stream(self) -> ClearlineVADStream:
        """
        Create a new VADStream for processing audio data.

        Returns:
            VADStream: A stream object for processing audio input and detecting speech.
        """
        stream = ClearlineVADStream(
            self,
            self._opts,
            onnx_model.OnnxModel(onnx_session=self._onnx_session),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update the VAD options.

        This method allows you to update the VAD options after the VAD object has been created.

        Args:
            min_speech_duration (float): Minimum duration of speech to start a new speech chunk.
            min_silence_duration (float): At the end of each speech, wait this duration before ending the speech.
            prefix_padding_duration (float): Duration of padding to add to the beginning of each speech chunk.
            max_buffered_speech (float): Maximum duration of speech to keep in the buffer (in seconds).
            activation_threshold (float): Threshold to consider a frame as speech.
        """  # noqa: E501
        if is_given(min_speech_duration):
            self._opts.min_speech_duration = min_speech_duration
        if is_given(min_silence_duration):
            self._opts.min_silence_duration = min_silence_duration
        if is_given(prefix_padding_duration):
            self._opts.prefix_padding_duration = prefix_padding_duration
        if is_given(max_buffered_speech):
            self._opts.max_buffered_speech = max_buffered_speech
        if is_given(activation_threshold):
            self._opts.activation_threshold = activation_threshold
        if is_given(deactivation_threshold):
            self._opts.deactivation_threshold = deactivation_threshold

        for stream in self._streams:
            stream.update_options(
                min_speech_duration=min_speech_duration,
                min_silence_duration=min_silence_duration,
                prefix_padding_duration=prefix_padding_duration,
                max_buffered_speech=max_buffered_speech,
                activation_threshold=activation_threshold,
                deactivation_threshold=deactivation_threshold,
            )

    @property
    def min_silence_duration(self) -> float | None:
        return self._opts.min_silence_duration


class ClearlineVADStream(agents.vad.VADStream):
    def __init__(
        self, vad: ClearlineVAD, opts: _VADOptions, model: onnx_model.OnnxModel
    ) -> None:
        super().__init__(vad)
        self._opts, self._model = opts, model
        self._loop = asyncio.get_event_loop()
        self._exp_filter = utils.ExpFilter(alpha=max(self._opts.smoothing_alpha, 1e-6))

        self._input_sample_rate = 0
        self._speech_buffer: np.ndarray | None = None
        self._speech_buffer_index = 0
        self._speech_buffer_max_reached = False
        self._prefix_padding_samples = 0  # (input_sample_rate)

        # the fast path (see the top of this file): the engine this stream is on, how many of its frames this stream
        # has been through, and the state machine's variables (upstream keeps them as locals of its loop)
        self._vad = vad
        self._engine: _Engine | None = None
        self._pos = 0
        self._since = 0  # frames since this stream started, or was flushed
        self._recent: collections.deque[rtc.AudioFrame] = collections.deque(
            maxlen=64
        )  # the last of them
        # id(frame) -> its number (1 = the first since), for the frames in _recent
        self._recent_at: dict[int, int] = {}
        self._reset_public_state()

    def _reset_public_state(self) -> None:
        self._pub_speaking = False
        self._pub_speech_duration = 0.0
        self._pub_silence_duration = 0.0
        self._pub_current_sample = 0
        self._pub_timestamp = 0.0
        self._speech_threshold_duration = 0.0
        self._silence_threshold_duration = 0.0

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update the VAD options.

        This method allows you to update the VAD options after the VAD object has been created.

        Args:
            min_speech_duration (float): Minimum duration of speech to start a new speech chunk.
            min_silence_duration (float): At the end of each speech, wait this duration before ending the speech.
            prefix_padding_duration (float): Duration of padding to add to the beginning of each speech chunk.
            max_buffered_speech (float): Maximum duration of speech to keep in the buffer (in seconds).
            activation_threshold (float): Threshold to consider a frame as speech.
            deactivation_threshold (float): Negative threshold (noise or exit threshold). If model's current state is SPEECH, values BELOW this value are considered as NON-SPEECH.
        """  # noqa: E501
        old_max_buffered_speech = self._opts.max_buffered_speech

        if is_given(min_speech_duration):
            self._opts.min_speech_duration = min_speech_duration
        if is_given(min_silence_duration):
            self._opts.min_silence_duration = min_silence_duration
        if is_given(prefix_padding_duration):
            self._opts.prefix_padding_duration = prefix_padding_duration
        if is_given(max_buffered_speech):
            self._opts.max_buffered_speech = max_buffered_speech
        if is_given(activation_threshold):
            self._opts.activation_threshold = activation_threshold
        if is_given(deactivation_threshold):
            self._opts.deactivation_threshold = deactivation_threshold

        if self._input_sample_rate:
            assert self._speech_buffer is not None

            self._prefix_padding_samples = int(
                self._opts.prefix_padding_duration * self._input_sample_rate
            )

            self._speech_buffer.resize(
                int(self._opts.max_buffered_speech * self._input_sample_rate)
                + self._prefix_padding_samples
            )

            # Resizing may discard samples that the write cursor still counted.
            self._speech_buffer_index = min(
                self._speech_buffer_index, len(self._speech_buffer)
            )

            if self._opts.max_buffered_speech > old_max_buffered_speech:
                self._speech_buffer_max_reached = False

    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            async for input_frame in self._input_ch:
                if isinstance(input_frame, self._FlushSentinel):
                    self._fast_reset()
                    continue
                if not isinstance(input_frame, rtc.AudioFrame):
                    continue
                if not self._input_sample_rate:
                    if (
                        self._opts.legacy_stream
                        or not self._opts.inline
                        or input_frame.sample_rate not in _FAST_RATES
                        or input_frame.num_channels != 1
                    ):
                        await self._legacy_task(
                            input_frame
                        )  # upstream's loop, to the end of the stream
                        return
                    self._input_sample_rate = input_frame.sample_rate
                    self._prefix_padding_samples = int(
                        self._opts.prefix_padding_duration * self._input_sample_rate
                    )
                    self._speech_buffer = np.empty(
                        int(self._opts.max_buffered_speech * self._input_sample_rate)
                        + self._prefix_padding_samples,
                        dtype=np.int16,
                    )
                elif self._input_sample_rate != input_frame.sample_rate:
                    logger.error("a frame with another sample rate was already pushed")
                    continue
                self._fast_frame(input_frame)
        finally:
            self._leave_engine()

    # ------------------------------------------------------------------ the fast path
    def _leave_engine(self) -> None:
        e, self._engine = self._engine, None
        if e is not None:
            if self in e.subs:
                e.subs.remove(self)
            if not e.subs and e in self._vad._engines:
                self._vad._engines.remove(e)

    def _join(self, e: _Engine, pos: int) -> None:
        self._leave_engine()
        e.subs.append(self)
        self._engine, self._pos = e, pos

    def _fast_reset(self) -> None:
        """A flush: upstream resets the model, the filter, the buffers and the state machine. This stream leaves its
        engine (alone on it, that drops the model's state with it) and starts again."""
        self._leave_engine()
        self._since = 0
        self._recent.clear()
        self._recent_at.clear()
        self._exp_filter = utils.ExpFilter(alpha=max(self._opts.smoothing_alpha, 1e-6))
        self._speech_buffer_index = 0
        self._speech_buffer_max_reached = False
        if self._speech_buffer is not None:
            self._speech_buffer.fill(0)
        self._reset_public_state()

    def _fast_frame(self, frame: rtc.AudioFrame) -> None:
        e = self._engine
        r = self._recent
        if (
            len(r) == r.maxlen
            and self._recent_at.get(id(r[0])) == self._since - len(r) + 1
        ):
            del self._recent_at[id(r[0])]  # the oldest is about to leave
        r.append(frame)
        self._since += 1
        self._recent_at[id(frame)] = self._since
        engines = self._vad._engines
        if (
            (e is None or len(e.subs) == 1)
            and len(engines) > (e is not None)
            and (self._since <= _PROBATION or self._since % _RECHECK == 0)
        ):
            # A stream alone on its engine (or with none yet) looks for an OLDER engine of this VAD that is hearing the
            # same frames, on each of its first frames and now and then after. Only ever the younger moves, so a
            # stream that has been running keeps its model's state and its windows. Two ways to find one:
            #   it has taken this very frame already (this stream is behind it): join where this frame was, and take
            #     the windows completed since;
            #   its last frame is one this stream has been through (this stream is ahead, as when the other's task is
            #     the slower of the two): hand it the frames it has not had, in order, then join.
            for o in engines:
                if (
                    o is e
                    or o.rate != frame.sample_rate
                    or (e is not None and o.born > e.born)
                ):
                    continue
                k = o.index_of(frame)
                if k >= 0:
                    self._join(o, k + 1)
                    for at, p, win, out, duration in tuple(o.recent):
                        if (
                            at >= k
                        ):  # completed by this frame or a later one, before this stream was here
                            self._on_window(p, win, out, duration)
                    return
                last = o.last()
                j = (
                    self._recent_at.get(id(last), 0) - (self._since - len(r) + 1)
                    if last is not None
                    else -1
                )
                if 0 <= j <= len(r) - 2 and r[j] is last:
                    for f in list(r)[j + 1 : -1]:
                        o.take(f)
                    self._join(o, o.count)
                    e = o
                    break
        if e is None:
            e = _Engine(self._vad, frame.sample_rate)
            engines.append(e)
            self._join(e, 0)
        if self._pos < e.count:
            # the engine took this frame already, for another stream; the window it may have completed has been here
            k = self._pos
            if k >= e.count - e.RING and e.ring[k % e.RING] is not frame:
                # not the same audio after all: this stream goes its own way from here
                self._leave_engine()
                e = _Engine(self._vad, frame.sample_rate)
                engines.append(e)
                self._join(e, 0)
            else:
                self._pos += 1
                return
        self._pos += 1
        e.take(frame)

    def _on_window(
        self,
        p: float,
        win: np.ndarray,
        frame: rtc.AudioFrame,
        inference_duration: float,
    ) -> None:
        """One window's probability: upstream's bookkeeping, events and state machine, line for line."""
        opts = self._opts
        assert self._speech_buffer is not None and self._engine is not None
        if opts.smoothing_alpha > 0:
            p = self._exp_filter.apply(exp=1.0, sample=p)
        window_duration = self._engine.window_duration
        self._pub_current_sample += self._engine.window
        self._pub_timestamp += window_duration

        # copy the inference window to the speech buffer
        available_space = len(self._speech_buffer) - self._speech_buffer_index
        to_copy_buffer = min(len(win), available_space)
        if to_copy_buffer > 0:
            self._speech_buffer[
                self._speech_buffer_index : self._speech_buffer_index + to_copy_buffer
            ] = win[:to_copy_buffer]
            self._speech_buffer_index += to_copy_buffer
        elif not self._speech_buffer_max_reached:
            # reached self._opts.max_buffered_speech (padding is included)
            self._speech_buffer_max_reached = True
            logger.warning(
                "max_buffered_speech reached, ignoring further data for the current speech input"
            )

        if self._pub_speaking:
            self._pub_speech_duration += window_duration
        else:
            self._pub_silence_duration += window_duration

        self._event_ch.send_nowait(
            agents.vad.VADEvent(
                type=agents.vad.VADEventType.INFERENCE_DONE,
                samples_index=self._pub_current_sample,
                timestamp=self._pub_timestamp,
                silence_duration=self._pub_silence_duration,
                speech_duration=self._pub_speech_duration,
                probability=p,
                inference_duration=inference_duration,
                frames=[frame],
                speaking=self._pub_speaking,
                raw_accumulated_silence=self._silence_threshold_duration,
                raw_accumulated_speech=self._speech_threshold_duration,
            )
        )

        if p >= opts.activation_threshold or (
            self._pub_speaking and p > opts.deactivation_threshold
        ):
            self._speech_threshold_duration += window_duration
            self._silence_threshold_duration = 0.0

            if (
                not self._pub_speaking
                and self._speech_threshold_duration >= opts.min_speech_duration
            ):
                self._pub_speaking = True
                self._pub_silence_duration = 0.0
                self._pub_speech_duration = self._speech_threshold_duration
                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.START_OF_SPEECH,
                        samples_index=self._pub_current_sample,
                        timestamp=self._pub_timestamp,
                        silence_duration=self._pub_silence_duration,
                        speech_duration=self._pub_speech_duration,
                        frames=[self._copy_speech_buffer()],
                        speaking=True,
                    )
                )
        else:
            self._silence_threshold_duration += window_duration
            self._speech_threshold_duration = 0.0

            if not self._pub_speaking:
                self._reset_write_cursor()

            if (
                self._pub_speaking
                and self._silence_threshold_duration >= opts.min_silence_duration
            ):
                self._pub_speaking = False
                self._pub_silence_duration = self._silence_threshold_duration
                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.END_OF_SPEECH,
                        samples_index=self._pub_current_sample,
                        timestamp=self._pub_timestamp,
                        silence_duration=self._pub_silence_duration,
                        speech_duration=max(
                            0.0,
                            self._pub_speech_duration
                            - self._silence_threshold_duration,
                        ),
                        frames=[self._copy_speech_buffer()],
                        speaking=False,
                    )
                )
                self._pub_speech_duration = 0.0
                self._reset_write_cursor()

    def _reset_write_cursor(self) -> None:
        assert self._speech_buffer is not None
        if self._speech_buffer_index <= self._prefix_padding_samples:
            return
        padding_data = self._speech_buffer[
            self._speech_buffer_index
            - self._prefix_padding_samples : self._speech_buffer_index
        ]
        self._speech_buffer_max_reached = False
        self._speech_buffer[: self._prefix_padding_samples] = padding_data
        self._speech_buffer_index = self._prefix_padding_samples

    def _copy_speech_buffer(self) -> rtc.AudioFrame:
        assert self._speech_buffer is not None
        return rtc.AudioFrame(
            sample_rate=self._input_sample_rate,
            num_channels=1,
            samples_per_channel=self._speech_buffer_index,
            data=self._speech_buffer[: self._speech_buffer_index].tobytes(),
        )

    # ------------------------------------------------------------------ upstream's loop
    async def _frames_from(self, first: rtc.AudioFrame):
        yield first
        async for f in self._input_ch:
            yield f

    async def _legacy_task(self, first_frame: rtc.AudioFrame) -> None:
        inference_f32_data = np.empty(self._model.window_size_samples, dtype=np.float32)

        # "pub_" means public, these values are exposed to the users through events
        pub_speaking = False
        pub_speech_duration = 0.0
        pub_silence_duration = 0.0
        pub_current_sample = 0
        pub_timestamp = 0.0

        speech_threshold_duration = 0.0
        silence_threshold_duration = 0.0

        input_frames: list[rtc.AudioFrame] = []
        inference_frames: list[rtc.AudioFrame] = []
        resampler: rtc.AudioResampler | None = None

        # used to avoid drift when the sample_rate ratio is not an integer
        input_copy_remaining_fract = 0.0

        extra_inference_time = 0.0

        def _reset_state() -> None:
            nonlocal pub_speaking, pub_speech_duration, pub_silence_duration
            nonlocal pub_current_sample, pub_timestamp
            nonlocal speech_threshold_duration, silence_threshold_duration
            nonlocal input_frames, inference_frames, resampler
            nonlocal input_copy_remaining_fract, extra_inference_time

            self._model.reset()
            self._exp_filter = utils.ExpFilter(
                alpha=max(self._opts.smoothing_alpha, 1e-6)
            )

            self._speech_buffer_index = 0
            self._speech_buffer_max_reached = False
            if self._speech_buffer is not None:
                self._speech_buffer.fill(0)

            pub_speaking = False
            pub_speech_duration = 0.0
            pub_silence_duration = 0.0
            pub_current_sample = 0
            pub_timestamp = 0.0
            speech_threshold_duration = 0.0
            silence_threshold_duration = 0.0

            input_frames = []
            inference_frames = []
            input_copy_remaining_fract = 0.0
            extra_inference_time = 0.0

            if (
                self._input_sample_rate
                and self._input_sample_rate != self._opts.sample_rate
            ):
                resampler = rtc.AudioResampler(
                    input_rate=self._input_sample_rate,
                    output_rate=self._opts.sample_rate,
                    quality=rtc.AudioResamplerQuality.QUICK,
                )
            else:
                resampler = None

        async for input_frame in self._frames_from(first_frame):
            if isinstance(input_frame, self._FlushSentinel):
                _reset_state()
                continue

            if not isinstance(input_frame, rtc.AudioFrame):
                continue

            if not self._input_sample_rate:
                self._input_sample_rate = input_frame.sample_rate

                # alloc the buffers now that we know the input sample rate
                self._prefix_padding_samples = int(
                    self._opts.prefix_padding_duration * self._input_sample_rate
                )

                self._speech_buffer = np.empty(
                    int(self._opts.max_buffered_speech * self._input_sample_rate)
                    + self._prefix_padding_samples,
                    dtype=np.int16,
                )

                if self._input_sample_rate != self._opts.sample_rate:
                    # resampling needed: the input sample rate isn't the same as the model's
                    # sample rate used for inference
                    resampler = rtc.AudioResampler(
                        input_rate=self._input_sample_rate,
                        output_rate=self._opts.sample_rate,
                        quality=rtc.AudioResamplerQuality.QUICK,  # VAD doesn't need high quality
                    )

            elif self._input_sample_rate != input_frame.sample_rate:
                logger.error("a frame with another sample rate was already pushed")
                continue

            assert self._speech_buffer is not None

            input_frames.append(input_frame)
            if resampler is not None:
                # the resampler may have a bit of latency, but it is OK to ignore since it should be
                # negligible
                inference_frames.extend(resampler.push(input_frame))
            else:
                inference_frames.append(input_frame)

            while True:
                start_time = time.perf_counter()

                available_inference_samples = sum(
                    [frame.samples_per_channel for frame in inference_frames]
                )
                if available_inference_samples < self._model.window_size_samples:
                    break  # not enough samples to run inference

                input_frame = utils.combine_frames(input_frames)
                inference_frame = utils.combine_frames(inference_frames)

                # convert data to f32
                np.divide(
                    inference_frame.data[: self._model.window_size_samples],
                    np.iinfo(np.int16).max,
                    out=inference_f32_data,
                    dtype=np.float32,
                )

                # run the inference
                self._vad.windows_inferred += 1
                if self._opts.inline:
                    p = self._model(inference_f32_data)
                else:
                    p = await self._loop.run_in_executor(
                        None, self._model, inference_f32_data
                    )
                if self._opts.smoothing_alpha > 0:
                    p = self._exp_filter.apply(exp=1.0, sample=p)

                window_duration = (
                    self._model.window_size_samples / self._opts.sample_rate
                )

                pub_current_sample += self._model.window_size_samples
                pub_timestamp += window_duration

                resampling_ratio = self._input_sample_rate / self._model.sample_rate
                to_copy = (
                    self._model.window_size_samples * resampling_ratio
                    + input_copy_remaining_fract
                )
                to_copy_int = int(to_copy)
                input_copy_remaining_fract = to_copy - to_copy_int

                # copy the inference window to the speech buffer
                available_space = len(self._speech_buffer) - self._speech_buffer_index
                to_copy_buffer = min(to_copy_int, available_space)
                if to_copy_buffer > 0:
                    self._speech_buffer[
                        self._speech_buffer_index : self._speech_buffer_index
                        + to_copy_buffer
                    ] = input_frame.data[:to_copy_buffer]
                    self._speech_buffer_index += to_copy_buffer
                elif not self._speech_buffer_max_reached:
                    # reached self._opts.max_buffered_speech (padding is included)
                    self._speech_buffer_max_reached = True
                    logger.warning(
                        "max_buffered_speech reached, ignoring further data for the current speech input"  # noqa: E501
                    )

                inference_duration = time.perf_counter() - start_time
                extra_inference_time = max(
                    0.0,
                    extra_inference_time + inference_duration - window_duration,
                )
                if inference_duration > SLOW_INFERENCE_THRESHOLD:
                    logger.warning(
                        "VAD inference is slower than realtime",
                        extra={"delay": extra_inference_time},
                    )

                def _reset_write_cursor() -> None:
                    assert self._speech_buffer is not None

                    if self._speech_buffer_index <= self._prefix_padding_samples:
                        return

                    padding_data = self._speech_buffer[
                        self._speech_buffer_index
                        - self._prefix_padding_samples : self._speech_buffer_index
                    ]

                    self._speech_buffer_max_reached = False
                    self._speech_buffer[: self._prefix_padding_samples] = padding_data
                    self._speech_buffer_index = self._prefix_padding_samples

                def _copy_speech_buffer() -> rtc.AudioFrame:
                    # copy the data from speech_buffer
                    assert self._speech_buffer is not None
                    speech_data = self._speech_buffer[
                        : self._speech_buffer_index
                    ].tobytes()

                    return rtc.AudioFrame(
                        sample_rate=self._input_sample_rate,
                        num_channels=1,
                        samples_per_channel=self._speech_buffer_index,
                        data=speech_data,
                    )

                if pub_speaking:
                    pub_speech_duration += window_duration
                else:
                    pub_silence_duration += window_duration

                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.INFERENCE_DONE,
                        samples_index=pub_current_sample,
                        timestamp=pub_timestamp,
                        silence_duration=pub_silence_duration,
                        speech_duration=pub_speech_duration,
                        probability=p,
                        inference_duration=inference_duration,
                        frames=[
                            rtc.AudioFrame(
                                data=input_frame.data[:to_copy_int].tobytes(),
                                sample_rate=self._input_sample_rate,
                                num_channels=1,
                                samples_per_channel=to_copy_int,
                            )
                        ],
                        speaking=pub_speaking,
                        raw_accumulated_silence=silence_threshold_duration,
                        raw_accumulated_speech=speech_threshold_duration,
                    )
                )

                if p >= self._opts.activation_threshold or (
                    pub_speaking and p > self._opts.deactivation_threshold
                ):
                    speech_threshold_duration += window_duration
                    silence_threshold_duration = 0.0

                    if not pub_speaking:
                        if speech_threshold_duration >= self._opts.min_speech_duration:
                            pub_speaking = True
                            pub_silence_duration = 0.0
                            pub_speech_duration = speech_threshold_duration

                            self._event_ch.send_nowait(
                                agents.vad.VADEvent(
                                    type=agents.vad.VADEventType.START_OF_SPEECH,
                                    samples_index=pub_current_sample,
                                    timestamp=pub_timestamp,
                                    silence_duration=pub_silence_duration,
                                    speech_duration=pub_speech_duration,
                                    frames=[_copy_speech_buffer()],
                                    speaking=True,
                                )
                            )

                else:
                    silence_threshold_duration += window_duration
                    speech_threshold_duration = 0.0

                    if not pub_speaking:
                        _reset_write_cursor()

                    if (
                        pub_speaking
                        and silence_threshold_duration
                        >= self._opts.min_silence_duration
                    ):
                        pub_speaking = False
                        pub_silence_duration = silence_threshold_duration

                        self._event_ch.send_nowait(
                            agents.vad.VADEvent(
                                type=agents.vad.VADEventType.END_OF_SPEECH,
                                samples_index=pub_current_sample,
                                timestamp=pub_timestamp,
                                silence_duration=pub_silence_duration,
                                speech_duration=max(
                                    0.0,
                                    pub_speech_duration - silence_threshold_duration,
                                ),
                                frames=[_copy_speech_buffer()],
                                speaking=False,
                            )
                        )

                        pub_speech_duration = 0.0

                        _reset_write_cursor()

                # remove the frames that were used for inference from the input and inference frames
                input_frames = []
                inference_frames = []

                # add the remaining data
                if len(input_frame.data) - to_copy_int > 0:
                    data = input_frame.data[to_copy_int:].tobytes()
                    input_frames.append(
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=self._input_sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,
                        )
                    )

                if len(inference_frame.data) - self._model.window_size_samples > 0:
                    data = inference_frame.data[
                        self._model.window_size_samples :
                    ].tobytes()
                    inference_frames.append(
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=self._opts.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,
                        )
                    )
