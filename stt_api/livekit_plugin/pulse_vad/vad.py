"""
PulseVAD as a LiveKit Agents `agents.vad.VAD` — a drop-in for `silero.VAD`.

This is the **third** LiveKit interface in this package and the only one that
answers "is someone speaking right now". Keep the three apart:

    ../turn_detector/   transcript text  -> has the speaker finished?
    ../semantic_vad/    streaming audio  -> has the speaker finished?
    this module         streaming audio  -> is there speech in this 200 ms?

`AgentSession(vad=...)` wants this one, and it still wants it when you also pass
`turn_detection=SemanticVAD(...)`: the VAD is what decides *when* to ask the turn
detector. They are not alternatives.

Two design points where this deliberately departs from `livekit-plugins-silero`,
both measured rather than assumed:

**The window and the update rate are decoupled.** PulseVAD's input is a fixed
200 ms patch; silero's is 32 ms. Reporting every 200 ms would be a real
regression — LiveKit's duration defaults are calibrated against silero's cadence,
and `min_speech_duration=0.05` would be satisfied by a single window, so one
spurious inference would open a turn. Instead the 200 ms window **slides** over a
`hop_duration` of 32 ms by default, matching silero's update interval at 6.25x
the inference count. That costs 0.097 ms x 31.25/s = **0.3 % of one core**.

**No exponential smoothing by default.** Silero applies `ExpFilter(alpha=0.35)`
because a 32 ms window is jittery. Consecutive PulseVAD windows already overlap
84 %, so the output is smooth on its own: measured over `test_audio/`, raw
threshold crossings run 0.3-1.8 per second, and α=0.35 cuts that to 0.3-1.5
while adding **32-160 ms** to onset latency. `min_speech_duration` already
debounces. Pass `smoothing_alpha` if your audio disagrees.

What you give up against silero is onset latency: a 200 ms causal window cannot
report speech until speech fills enough of it. Measured from speech start to the
first window over threshold, 2.1k fp32 at 0.35: **median 140 ms, worst 908 ms**
on clips with a breathy onset; the 81k teacher manages 156-236 ms and is worth
the 0.098 ms if barge-in latency matters more than CPU. See README.md.
"""

from __future__ import annotations

import asyncio
import time
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .frontend import SAMPLE_RATE, WINDOW_SAMPLES
from .model import CEILINGS, PulseVADModel, Precision, Variant

if TYPE_CHECKING:  # both classes are built lazily by __getattr__ below
    PulseVAD: type
    PulseVADStream: type

__all__ = ["PulseVAD", "PulseVADStream"]

SLOW_INFERENCE_THRESHOLD = 0.2
"""Match silero: warn once the loop is 200 ms behind realtime."""

DEFAULT_ACTIVATION_THRESHOLD = 0.35
"""
Not silero's 0.5, and the difference is not cosmetic.

p(speech) from the 2.1k model spans [0.002, 0.711] rather than [0, 1] (see
`model.CEILINGS`), so silero's 0.5 sits at 70 % of the usable range — high enough
that 0.5 and 0.7 give visibly different segmentations. 0.35 is the midpoint of
the measured range for both checkpoints, and over 1,200 windows of digital
silence, white noise at -40/-20 dBFS and pink-ish noise it fired on **0 %**
(2.1k) and **1-2 %** (81k) — isolated windows that `min_speech_duration` drops.
"""


@dataclass
class _VADOptions:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    deactivation_threshold: float
    smoothing_alpha: float
    sample_rate: int
    hop_samples: int


def _import_livekit():
    try:
        from livekit import agents, rtc  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError(
            "stt_api.livekit_plugin.pulse_vad needs livekit-agents. "
            f'Install the extra: pip install ".[livekit]". Import failed: {e}'
        ) from e
    return agents, rtc


def _make_classes() -> tuple[type, type]:
    agents, rtc = _import_livekit()
    from livekit.agents import utils
    from livekit.agents.log import logger
    from livekit.agents.types import NOT_GIVEN, NotGivenOr
    from livekit.agents.utils import is_given

    class PulseVAD(agents.vad.VAD):
        """
        Frame-level voice activity detection from a 2,118-parameter CNN.

            from stt_api.livekit_plugin.pulse_vad import PulseVAD

            def prewarm(proc):
                proc.userdata["vad"] = PulseVAD.load()

            session = AgentSession(vad=ctx.proc.userdata["vad"], stt=..., llm=...)

        `load()` is blocking — call it in `prewarm`, as with silero.
        """

        @classmethod
        def load(
            cls,
            *,
            model: Variant = "2.1k",
            precision: Precision = "fp32",
            min_speech_duration: float = 0.05,
            min_silence_duration: float = 0.55,
            prefix_padding_duration: float = 0.5,
            max_buffered_speech: float = 60.0,
            activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
            deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
            smoothing_alpha: float = 1.0,
            hop_duration: float = 0.032,
            num_threads: int = 1,
            providers: list[str] | None = None,
            model_path: str | None = None,
        ) -> PulseVAD:
            """
            Load PulseVAD and prepare it for streaming.

            Args:
                model: `"2.1k"` (2,118 params, the ship model) or `"81k"` (the
                    teacher — 5x the compute, materially better onset latency).
                precision: `"fp32"` or `"int8"`. fp32 by default: int8 measured
                    *no faster* off-microcontroller (0.019 ms vs 0.018 ms) and
                    ships a larger file. `"81k"` has no int8 build.
                min_speech_duration: speech this long opens a turn.
                min_silence_duration: silence this long closes one.
                prefix_padding_duration: audio kept before the detected onset.
                max_buffered_speech: cap on buffered speech, in seconds.
                activation_threshold: p(speech) to count a window as speech.
                    **Must sit below the checkpoint's ceiling** — 0.711 for 2.1k
                    fp32, 0.895 for 81k — or the VAD can never fire. Checked here.
                deactivation_threshold: exit threshold while already speaking.
                    Defaults to `max(activation - 0.1, 0.01)`; the gap is smaller
                    than silero's 0.15 because the output range is ~0.7 wide, not 1.
                smoothing_alpha: exponential smoothing on p. 1.0 (default) is off
                    — see the module docstring.
                hop_duration: how often to run inference. The 200 ms window is
                    fixed by the model; this is how far it slides between runs.
                num_threads: ORT threads. 1 is right for 0.018 ms of work.
                providers: ORT execution providers. CPU by default.
                model_path: override the vendored weights (also
                    `PULSEVAD_ONNX_PATH`).
            """
            if not 0.0 < smoothing_alpha <= 1.0:
                raise ValueError(
                    f"smoothing_alpha must be in (0, 1], got {smoothing_alpha}"
                )

            hop_samples = int(round(hop_duration * SAMPLE_RATE))
            if not 0 < hop_samples <= WINDOW_SAMPLES:
                raise ValueError(
                    f"hop_duration must be in (0, {WINDOW_SAMPLES / SAMPLE_RATE}] seconds "
                    f"(the model's window), got {hop_duration}"
                )

            loaded = PulseVADModel(
                model,
                precision=precision,
                num_threads=num_threads,
                providers=providers,
                model_path=model_path,
            )

            # The whole point of CEILINGS: a threshold above the model's range is
            # a VAD that never fires, and it fails silently — no exception, no
            # log line, just an agent that never hears anyone. Coming from silero
            # (which saturates at 1.0) `activation_threshold=0.8` is an entirely
            # ordinary thing to write.
            if activation_threshold >= loaded.ceiling:
                raise ValueError(
                    f"activation_threshold={activation_threshold} is at or above the "
                    f"{model}/{precision} ceiling of {loaded.ceiling}: p(speech) never "
                    f"reaches it, so this VAD could never report speech. PulseVAD does "
                    f"not saturate at 1.0 the way silero does — measured range is "
                    f"[{0.002 if model == '2.1k' else 0.006}, {loaded.ceiling}]. "
                    f"Try {DEFAULT_ACTIVATION_THRESHOLD} (the midpoint), or "
                    f"model='81k' whose ceiling is {CEILINGS[('81k', 'fp32')]}."
                )
            if is_given(deactivation_threshold) and deactivation_threshold <= 0:
                raise ValueError("deactivation_threshold must be greater than 0")

            opts = _VADOptions(
                min_speech_duration=min_speech_duration,
                min_silence_duration=min_silence_duration,
                prefix_padding_duration=prefix_padding_duration,
                max_buffered_speech=max_buffered_speech,
                activation_threshold=activation_threshold,
                deactivation_threshold=(
                    deactivation_threshold
                    if is_given(deactivation_threshold)
                    else max(activation_threshold - 0.1, 0.01)
                ),
                smoothing_alpha=smoothing_alpha,
                sample_rate=SAMPLE_RATE,
                hop_samples=hop_samples,
            )
            return cls(model=loaded, opts=opts)

        def __init__(self, *, model: PulseVADModel, opts: _VADOptions) -> None:
            super().__init__(
                capabilities=agents.vad.VADCapabilities(
                    update_interval=opts.hop_samples / opts.sample_rate
                )
            )
            self._model = model
            self._opts = opts
            self._streams = weakref.WeakSet[PulseVADStream]()

        @property
        def model(self) -> str:
            return f"pulsevad-{self._model.model}-{self._model.precision}"

        @property
        def provider(self) -> str:
            return "ONNX"

        @property
        def ceiling(self) -> float:
            """Largest p(speech) the loaded checkpoint is known to emit."""
            return self._model.ceiling

        @property
        def min_silence_duration(self) -> float:
            return self._opts.min_silence_duration

        def stream(self) -> PulseVADStream:
            stream = PulseVADStream(self, self._opts, self._model)
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
            """Change options on the VAD and every live stream."""
            if (
                is_given(activation_threshold)
                and activation_threshold >= self._model.ceiling
            ):
                raise ValueError(
                    f"activation_threshold={activation_threshold} is at or above the "
                    f"{self._model.model}/{self._model.precision} ceiling of "
                    f"{self._model.ceiling}; the VAD would stop reporting speech entirely."
                )
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

    class PulseVADStream(agents.vad.VADStream):
        def __init__(
            self, vad: PulseVAD, opts: _VADOptions, model: PulseVADModel
        ) -> None:
            super().__init__(vad)
            self._opts, self._model = opts, model
            self._loop = asyncio.get_event_loop()
            self._input_sample_rate = 0
            self._speech_buffer: np.ndarray | None = None
            self._speech_buffer_max_reached = False
            self._prefix_padding_samples = 0

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
                if self._opts.max_buffered_speech > old_max_buffered_speech:
                    self._speech_buffer_max_reached = False

        @agents.utils.log_exceptions(logger=logger)
        async def _main_task(self) -> None:
            hop = self._opts.hop_samples
            hop_duration = hop / self._opts.sample_rate
            hop_f32 = np.empty(hop, dtype=np.float32)

            # The model's 200 ms of context. Primed with silence rather than
            # waiting for a full window: PulseVAD scores digital silence at 0.008,
            # so the opening windows of a session read as non-speech and the ring
            # fills with real audio within 200 ms. A partially-filled window is
            # the same input shape as a genuine onset after a pause, which is
            # squarely in distribution.
            window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

            speech_buffer_index = 0
            pub_speaking = False
            pub_speech_duration = 0.0
            pub_silence_duration = 0.0
            pub_current_sample = 0
            pub_timestamp = 0.0
            speech_threshold_duration = 0.0
            silence_threshold_duration = 0.0
            smoothed: float | None = None

            input_frames: list[rtc.AudioFrame] = []
            inference_frames: list[rtc.AudioFrame] = []
            resampler: rtc.AudioResampler | None = None
            input_copy_remaining_fract = 0.0
            extra_inference_time = 0.0

            def _reset_state() -> None:
                nonlocal speech_buffer_index, pub_speaking, pub_speech_duration
                nonlocal pub_silence_duration, pub_current_sample, pub_timestamp
                nonlocal speech_threshold_duration, silence_threshold_duration
                nonlocal input_frames, inference_frames, resampler, smoothed
                nonlocal input_copy_remaining_fract, extra_inference_time

                # `flush()` is a hard segment boundary: the previous speaker's
                # audio must not bleed into the next decision, so the context
                # window goes back to silence too.
                window.fill(0.0)
                smoothed = None
                speech_buffer_index = 0
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

            async for input_frame in self._input_ch:
                if isinstance(input_frame, self._FlushSentinel):
                    _reset_state()
                    continue
                if not isinstance(input_frame, rtc.AudioFrame):
                    continue

                if not self._input_sample_rate:
                    self._input_sample_rate = input_frame.sample_rate
                    self._prefix_padding_samples = int(
                        self._opts.prefix_padding_duration * self._input_sample_rate
                    )
                    self._speech_buffer = np.empty(
                        int(self._opts.max_buffered_speech * self._input_sample_rate)
                        + self._prefix_padding_samples,
                        dtype=np.int16,
                    )
                    if self._input_sample_rate != self._opts.sample_rate:
                        resampler = rtc.AudioResampler(
                            input_rate=self._input_sample_rate,
                            output_rate=self._opts.sample_rate,
                            quality=rtc.AudioResamplerQuality.QUICK,
                        )
                elif self._input_sample_rate != input_frame.sample_rate:
                    logger.error("a frame with another sample rate was already pushed")
                    continue

                assert self._speech_buffer is not None

                input_frames.append(input_frame)
                if resampler is not None:
                    inference_frames.extend(resampler.push(input_frame))
                else:
                    inference_frames.append(input_frame)

                while True:
                    start_time = time.perf_counter()

                    available = sum(f.samples_per_channel for f in inference_frames)
                    if available < hop:
                        break  # not enough new audio to advance the window

                    input_frame = utils.combine_frames(input_frames)
                    inference_frame = utils.combine_frames(inference_frames)

                    # Slide the 200 ms context on by one hop. Unlike silero — whose
                    # window *is* its hop — consecutive inputs here overlap by
                    # WINDOW_SAMPLES - hop (168 ms at the default), which is what
                    # buys a silero-rate update from a 200 ms model.
                    np.divide(
                        inference_frame.data[:hop],
                        np.iinfo(np.int16).max,
                        out=hop_f32,
                        dtype=np.float32,
                    )
                    window[:-hop] = window[hop:]
                    window[-hop:] = hop_f32

                    p = await self._loop.run_in_executor(None, self._model, window)
                    if self._opts.smoothing_alpha < 1.0:
                        alpha = self._opts.smoothing_alpha
                        smoothed = (
                            p
                            if smoothed is None
                            else alpha * p + (1 - alpha) * smoothed
                        )
                        p = smoothed

                    pub_current_sample += hop
                    pub_timestamp += hop_duration

                    resampling_ratio = self._input_sample_rate / self._opts.sample_rate
                    to_copy = hop * resampling_ratio + input_copy_remaining_fract
                    to_copy_int = int(to_copy)
                    input_copy_remaining_fract = to_copy - to_copy_int

                    available_space = len(self._speech_buffer) - speech_buffer_index
                    to_copy_buffer = min(to_copy_int, available_space)
                    if to_copy_buffer > 0:
                        self._speech_buffer[
                            speech_buffer_index : speech_buffer_index + to_copy_buffer
                        ] = input_frame.data[:to_copy_buffer]
                        speech_buffer_index += to_copy_buffer
                    elif not self._speech_buffer_max_reached:
                        self._speech_buffer_max_reached = True
                        logger.warning(
                            "max_buffered_speech reached, ignoring further data for the "
                            "current speech input"
                        )

                    inference_duration = time.perf_counter() - start_time
                    extra_inference_time = max(
                        0.0, extra_inference_time + inference_duration - hop_duration
                    )
                    if inference_duration > SLOW_INFERENCE_THRESHOLD:
                        logger.warning(
                            "VAD inference is slower than realtime",
                            extra={"delay": extra_inference_time},
                        )

                    def _reset_write_cursor() -> None:
                        nonlocal speech_buffer_index
                        assert self._speech_buffer is not None
                        if speech_buffer_index <= self._prefix_padding_samples:
                            return
                        padding_data = self._speech_buffer[
                            speech_buffer_index
                            - self._prefix_padding_samples : speech_buffer_index
                        ]
                        self._speech_buffer_max_reached = False
                        self._speech_buffer[
                            : self._prefix_padding_samples
                        ] = padding_data
                        speech_buffer_index = self._prefix_padding_samples

                    def _copy_speech_buffer() -> rtc.AudioFrame:
                        assert self._speech_buffer is not None
                        return rtc.AudioFrame(
                            sample_rate=self._input_sample_rate,
                            num_channels=1,
                            samples_per_channel=speech_buffer_index,  # noqa: B023
                            data=self._speech_buffer[:speech_buffer_index].tobytes(),  # noqa: B023
                        )

                    if pub_speaking:
                        pub_speech_duration += hop_duration
                    else:
                        pub_silence_duration += hop_duration

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
                        speech_threshold_duration += hop_duration
                        silence_threshold_duration = 0.0
                        if not pub_speaking and (
                            speech_threshold_duration >= self._opts.min_speech_duration
                        ):
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
                        silence_threshold_duration += hop_duration
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
                                        pub_speech_duration
                                        - silence_threshold_duration,
                                    ),
                                    frames=[_copy_speech_buffer()],
                                    speaking=False,
                                )
                            )
                            pub_speech_duration = 0.0
                            _reset_write_cursor()

                    # Consume exactly one hop from each queue, keeping the rest.
                    input_frames = []
                    inference_frames = []
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
                    if len(inference_frame.data) - hop > 0:
                        data = inference_frame.data[hop:].tobytes()
                        inference_frames.append(
                            rtc.AudioFrame(
                                data=data,
                                sample_rate=self._opts.sample_rate,
                                num_channels=1,
                                samples_per_channel=len(data) // 2,
                            )
                        )

    return PulseVAD, PulseVADStream


def __getattr__(name: str):
    # Built on first access so importing the package does not require
    # livekit-agents: frontend.py and model.py are useful on their own.
    if name in ("PulseVAD", "PulseVADStream"):
        cls_vad, cls_stream = _make_classes()
        globals()["PulseVAD"], globals()["PulseVADStream"] = cls_vad, cls_stream
        return globals()[name]
    raise AttributeError(name)
