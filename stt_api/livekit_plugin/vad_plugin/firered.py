"""LiveKit VAD plugin using FireRedVAD (DFSMN-based)."""

from __future__ import annotations

import asyncio
import statistics
import time
import weakref
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from livekit import agents, rtc
from livekit.agents import utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .fireredvad import FireRedStreamVad, FireRedStreamVadConfig
from .constants import FRAME_SHIFT_SAMPLE, SAMPLE_RATE

from .log import logger

SLOW_INFERENCE_THRESHOLD = 0.2


@dataclass
class _FireRedVADOptions:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    sample_rate: int
    profiling: bool = False


@dataclass
class ProfilingStats:
    inference_count: int = 0
    latencies: list[float] = field(default_factory=list)
    total_time: float = 0.0

    def summary(self) -> str:
        if not self.latencies:
            return "No inferences recorded."
        sorted_lat = sorted(self.latencies)
        mean_ms = statistics.mean(sorted_lat) * 1000
        median_ms = statistics.median(sorted_lat) * 1000
        p95_idx = int(len(sorted_lat) * 0.95)
        p99_idx = int(len(sorted_lat) * 0.99)
        p95_ms = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] * 1000
        p99_ms = sorted_lat[min(p99_idx, len(sorted_lat) - 1)] * 1000
        lines = [
            "=" * 50,
            "FIRERED VAD PROFILING RESULTS",
            "=" * 50,
            f"  Inferences:  {self.inference_count}",
            f"  Total time:  {self.total_time:.4f}s",
            f"  Mean:        {mean_ms:.3f}ms",
            f"  Median:      {median_ms:.3f}ms",
            f"  p95:         {p95_ms:.3f}ms",
            f"  p99:         {p99_ms:.3f}ms",
            f"  Min:         {min(sorted_lat) * 1000:.3f}ms",
            f"  Max:         {max(sorted_lat) * 1000:.3f}ms",
            "=" * 50,
        ]
        return "\n".join(lines)


class VAD(agents.vad.VAD):
    """LiveKit VAD plugin using FireRedVAD (Deep FSMN architecture)."""

    # FireRedVAD operates at 10ms frame shift (100 frames/s)
    # update_interval matches this
    FRAME_SHIFT_S = FRAME_SHIFT_SAMPLE / SAMPLE_RATE  # 0.01s

    PROFILING_WARMUP = 10
    PROFILING_ITERATIONS = 200

    @classmethod
    def load(
        cls,
        *,
        min_speech_duration: float = 0.06,
        min_silence_duration: float = 0.4,
        prefix_padding_duration: float = 0.08,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        sample_rate: Literal[16000] = 16000,
        use_gpu: bool = False,
        profiling: bool = False,
    ) -> VAD:
        """Load FireRedVAD model.

        Args:
            min_speech_duration: Min speech to trigger START_OF_SPEECH (seconds).
            min_silence_duration: Silence duration to trigger END_OF_SPEECH (seconds).
            prefix_padding_duration: Padding before speech start (seconds).
            max_buffered_speech: Max speech buffer (seconds).
            activation_threshold: Speech probability threshold.
            sample_rate: Only 16kHz supported by FireRedVAD.
            use_gpu: Use GPU for inference.
            profiling: Run profiling on load and track live inference.
        """
        if sample_rate != 16000:
            raise ValueError("FireRedVAD only supports 16kHz sample rate")

        # Convert seconds to frame counts (10ms per frame)
        fps = 100  # frames per second
        config = FireRedStreamVadConfig(
            use_gpu=use_gpu,
            smooth_window_size=5,
            speech_threshold=activation_threshold,
            pad_start_frame=max(1, int(prefix_padding_duration * fps)),
            min_speech_frame=max(1, int(min_speech_duration * fps)),
            max_speech_frame=int(max_buffered_speech * fps),
            min_silence_frame=max(1, int(min_silence_duration * fps)),
        )

        firered = FireRedStreamVad.from_pretrained(config=config)

        opts = _FireRedVADOptions(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            sample_rate=sample_rate,
            profiling=profiling,
        )

        vad = cls(firered=firered, config=config, opts=opts)

        if profiling:
            stats = vad.run_profiling()
            logger.info("FireRedVAD profiling on load:\n%s", stats.summary())

        return vad

    def __init__(
        self,
        *,
        firered: FireRedStreamVad,
        config: FireRedStreamVadConfig,
        opts: _FireRedVADOptions,
    ) -> None:
        super().__init__(
            capabilities=agents.vad.VADCapabilities(update_interval=self.FRAME_SHIFT_S)
        )
        self._firered = firered
        self._config = config
        self._opts = opts
        self._streams = weakref.WeakSet[FireRedVADStream]()
        self._profiling_stats: ProfilingStats | None = None

    @property
    def model(self) -> str:
        return "firered"

    @property
    def provider(self) -> str:
        return "FireRedTeam"

    @property
    def profiling_stats(self) -> ProfilingStats | None:
        return self._profiling_stats

    def run_profiling(self, iterations: int | None = None) -> ProfilingStats:
        """Run sample data through FireRedVAD and measure inference latency."""
        n = iterations or self.PROFILING_ITERATIONS
        rng = np.random.default_rng(42)

        # FireRedVAD processes audio chunks → fbank features → model
        # Simulate with random int16 audio chunks (160 samples = 10ms at 16kHz)
        chunk_size = FRAME_SHIFT_SAMPLE  # 160 samples

        stats = ProfilingStats()

        # Create a fresh instance for profiling
        self._firered.reset()

        # Warmup
        for _ in range(self.PROFILING_WARMUP):
            chunk = rng.integers(-3000, 3000, size=chunk_size, dtype=np.int16)
            feats, _ = self._firered.audio_feat.extract(chunk)
            if feats.size(0) > 0:
                with torch.no_grad():
                    self._firered.vad_model.forward(
                        feats.unsqueeze(0), caches=self._firered.model_caches
                    )

        self._firered.reset()

        # Benchmark
        total_start = time.perf_counter()
        for i in range(n):
            if i % 3 == 0:
                chunk = np.zeros(chunk_size, dtype=np.int16)
            elif i % 3 == 1:
                chunk = rng.integers(-500, 500, size=chunk_size, dtype=np.int16)
            else:
                chunk = rng.integers(-20000, 20000, size=chunk_size, dtype=np.int16)

            t0 = time.perf_counter()
            feats, _ = self._firered.audio_feat.extract(chunk)
            if feats.size(0) > 0:
                with torch.no_grad():
                    probs, self._firered.model_caches = self._firered.vad_model.forward(
                        feats.unsqueeze(0), caches=self._firered.model_caches
                    )
            elapsed = time.perf_counter() - t0
            stats.latencies.append(elapsed)
            stats.inference_count += 1

        stats.total_time = time.perf_counter() - total_start
        self._firered.reset()
        self._profiling_stats = stats
        return stats

    def stream(self) -> FireRedVADStream:
        stream = FireRedVADStream(self, self._opts, self._firered, self._config)
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


class FireRedVADStream(agents.vad.VADStream):
    def __init__(
        self,
        vad: VAD,
        opts: _FireRedVADOptions,
        firered: FireRedStreamVad,
        config: FireRedStreamVadConfig,
    ) -> None:
        super().__init__(vad)
        self._opts = opts
        self._firered = firered
        self._config = config
        self._loop = asyncio.get_event_loop()

        self._input_sample_rate = 0
        self._speech_buffer: np.ndarray | None = None
        self._speech_buffer_max_reached = False
        self._prefix_padding_samples = 0

        # Audio accumulation buffer (int16 samples at 16kHz)
        self._audio_buf = np.empty(0, dtype=np.int16)

        # FireRedVAD stateful objects (per-stream copy)
        self._model_caches = None
        self._postprocessor = firered.postprocessor
        self._postprocessor.reset()
        self._firered_ref = firered

        # Profiling
        self._profiling = opts.profiling
        self._stream_stats = ProfilingStats() if self._profiling else None
        self._stream_start_time: float | None = None

    @property
    def stream_stats(self) -> ProfilingStats | None:
        return self._stream_stats

    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        if self._profiling:
            self._stream_start_time = time.perf_counter()

        speech_buffer_index: int = 0
        pub_speaking = False
        pub_speech_duration = 0.0
        pub_silence_duration = 0.0
        pub_current_sample = 0
        pub_timestamp = 0.0

        resampler: rtc.AudioResampler | None = None

        frame_shift = FRAME_SHIFT_SAMPLE  # 160 samples at 16kHz = 10ms

        async for input_frame in self._input_ch:
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

                if self._input_sample_rate != SAMPLE_RATE:
                    resampler = rtc.AudioResampler(
                        input_rate=self._input_sample_rate,
                        output_rate=SAMPLE_RATE,
                        quality=rtc.AudioResamplerQuality.QUICK,
                    )

            elif self._input_sample_rate != input_frame.sample_rate:
                logger.error("a frame with another sample rate was already pushed")
                continue

            assert self._speech_buffer is not None

            # Resample if needed, accumulate int16 samples
            if resampler is not None:
                resampled = resampler.push(input_frame)
                for rf in resampled:
                    self._audio_buf = np.append(
                        self._audio_buf, np.array(rf.data, dtype=np.int16)
                    )
            else:
                self._audio_buf = np.append(
                    self._audio_buf, np.array(input_frame.data, dtype=np.int16)
                )

            # Copy to speech buffer
            input_data = np.array(input_frame.data, dtype=np.int16)
            available_space = len(self._speech_buffer) - speech_buffer_index
            to_copy = min(len(input_data), available_space)
            if to_copy > 0:
                self._speech_buffer[
                    speech_buffer_index: speech_buffer_index + to_copy
                ] = input_data[:to_copy]
                speech_buffer_index += to_copy
            elif not self._speech_buffer_max_reached:
                self._speech_buffer_max_reached = True
                logger.warning("max_buffered_speech reached")

            # Process complete 10ms frames
            while len(self._audio_buf) >= frame_shift:
                start_time = time.perf_counter()

                chunk = self._audio_buf[:frame_shift]
                self._audio_buf = self._audio_buf[frame_shift:]

                # Extract fbank features
                feats, _ = self._firered_ref.audio_feat.extract(chunk)

                p = 0.0
                if feats.size(0) > 0:
                    with torch.no_grad():
                        probs, self._model_caches = (
                            self._firered_ref.vad_model.forward(
                                feats.unsqueeze(0), caches=self._model_caches
                            )
                        )
                        # probs shape: (1, T, 1) — take last frame probability
                        p = probs[0, -1, 0].item()

                # Run through postprocessor
                frame_result = self._postprocessor.process_one_frame(p)

                window_duration = FRAME_SHIFT_SAMPLE / SAMPLE_RATE  # 0.01s
                pub_current_sample += frame_shift
                pub_timestamp += window_duration

                inference_duration = time.perf_counter() - start_time

                if self._profiling and self._stream_stats is not None:
                    self._stream_stats.latencies.append(inference_duration)
                    self._stream_stats.inference_count += 1

                if pub_speaking:
                    pub_speech_duration += window_duration
                else:
                    pub_silence_duration += window_duration

                # Emit INFERENCE_DONE
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
                                data=chunk.tobytes(),
                                sample_rate=SAMPLE_RATE,
                                num_channels=1,
                                samples_per_channel=frame_shift,
                            )
                        ],
                        speaking=pub_speaking,
                    )
                )

                def _copy_speech_buffer() -> rtc.AudioFrame:
                    assert self._speech_buffer is not None
                    speech_data = self._speech_buffer[:speech_buffer_index].tobytes()
                    return rtc.AudioFrame(
                        sample_rate=self._input_sample_rate,
                        num_channels=1,
                        samples_per_channel=speech_buffer_index,
                        data=speech_data,
                    )

                def _reset_write_cursor() -> None:
                    nonlocal speech_buffer_index
                    assert self._speech_buffer is not None
                    if speech_buffer_index <= self._prefix_padding_samples:
                        return
                    padding_data = self._speech_buffer[
                        speech_buffer_index - self._prefix_padding_samples: speech_buffer_index
                    ]
                    self._speech_buffer_max_reached = False
                    self._speech_buffer[: self._prefix_padding_samples] = padding_data
                    speech_buffer_index = self._prefix_padding_samples

                # Map FireRedVAD state transitions to LiveKit events
                if frame_result.is_speech_start and not pub_speaking:
                    pub_speaking = True
                    pub_silence_duration = 0.0
                    pub_speech_duration = 0.0

                    self._event_ch.send_nowait(
                        agents.vad.VADEvent(
                            type=agents.vad.VADEventType.START_OF_SPEECH,
                            samples_index=pub_current_sample,
                            timestamp=pub_timestamp,
                            silence_duration=0.0,
                            speech_duration=0.0,
                            frames=[_copy_speech_buffer()],
                            speaking=True,
                        )
                    )

                elif frame_result.is_speech_end and pub_speaking:
                    pub_speaking = False
                    pub_silence_duration = 0.0

                    self._event_ch.send_nowait(
                        agents.vad.VADEvent(
                            type=agents.vad.VADEventType.END_OF_SPEECH,
                            samples_index=pub_current_sample,
                            timestamp=pub_timestamp,
                            silence_duration=pub_silence_duration,
                            speech_duration=pub_speech_duration,
                            frames=[_copy_speech_buffer()],
                            speaking=False,
                        )
                    )

                    pub_speech_duration = 0.0
                    _reset_write_cursor()

                elif not pub_speaking:
                    _reset_write_cursor()

        # Log profiling stats when stream ends
        if (
            self._profiling
            and self._stream_stats is not None
            and self._stream_start_time is not None
        ):
            self._stream_stats.total_time = time.perf_counter() - self._stream_start_time
            logger.info(
                "FireRedVAD stream profiling:\n%s", self._stream_stats.summary()
            )
