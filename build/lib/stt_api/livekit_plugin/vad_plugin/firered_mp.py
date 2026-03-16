"""LiveKit VAD plugin using FireRedVAD with ProcessPoolExecutor.

Each worker process loads its own FireRedVAD model. Since FireRedVAD is stateful
(model caches), we pass caches back and forth via IPC. This trades IPC overhead
for true parallelism (bypasses GIL).
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import statistics
import time
import weakref
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from livekit import agents, rtc
from livekit.agents import utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from stt_api.vad.fireredvad import FireRedStreamVadConfig
from stt_api.vad.fireredvad.constants import FRAME_SHIFT_SAMPLE, SAMPLE_RATE

from .log import logger

SLOW_INFERENCE_THRESHOLD = 0.2

VAD_WORKERS = int(os.environ.get("VAD_WORKERS", "4"))

# --- Worker process globals ---
_worker_firered = None
_worker_feat_extractor = None
_worker_vad_model = None


def _init_worker(use_gpu: bool):
    """Initialize FireRedVAD in each worker process."""
    global _worker_feat_extractor, _worker_vad_model

    from stt_api.vad.fireredvad import FireRedStreamVad, FireRedStreamVadConfig
    from stt_api.vad.fireredvad.audio_feat import AudioFeat
    from stt_api.vad.fireredvad.detect_model import DetectModel
    from huggingface_hub import snapshot_download

    folder_name = snapshot_download(repo_id="FireRedTeam/FireRedVAD")
    model_dir = os.path.join(folder_name, "Stream-VAD")
    cmvn_path = os.path.join(model_dir, "cmvn.ark")

    _worker_feat_extractor = AudioFeat(cmvn_path)
    _worker_vad_model = DetectModel.from_pretrained(model_dir)
    if use_gpu:
        _worker_vad_model.cuda()
    else:
        _worker_vad_model.cpu()

    logger.info(f"FireRedVAD worker {os.getpid()} initialized")


def _run_inference(chunk: np.ndarray, caches) -> tuple[float, list]:
    """Run fbank extraction + DFSMN inference in worker process."""
    global _worker_feat_extractor, _worker_vad_model

    feats, _ = _worker_feat_extractor.extract(chunk)
    p = 0.0
    new_caches = caches

    if feats.size(0) > 0:
        with torch.no_grad():
            probs, new_caches = _worker_vad_model.forward(
                feats.unsqueeze(0), caches=caches
            )
            p = probs[0, -1, 0].item()

    return p, new_caches


_vad_executor: ProcessPoolExecutor | None = None


def _get_vad_executor(use_gpu: bool = False) -> ProcessPoolExecutor:
    global _vad_executor
    if _vad_executor is None:
        ctx = multiprocessing.get_context("spawn")
        _vad_executor = ProcessPoolExecutor(
            max_workers=VAD_WORKERS,
            initializer=_init_worker,
            initargs=(use_gpu,),
            mp_context=ctx,
        )
        logger.info(f"FireRedVAD ProcessPool initialized with {VAD_WORKERS} workers")
    return _vad_executor


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
            "FIRERED VAD (MP) PROFILING RESULTS",
            "=" * 50,
            f"  Workers:     {VAD_WORKERS}",
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
    """LiveKit VAD using FireRedVAD with ProcessPoolExecutor."""

    FRAME_SHIFT_S = FRAME_SHIFT_SAMPLE / SAMPLE_RATE

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
        if sample_rate != 16000:
            raise ValueError("FireRedVAD only supports 16kHz sample rate")

        # Start process pool
        _get_vad_executor(use_gpu=use_gpu)

        fps = 100
        config = FireRedStreamVadConfig(
            use_gpu=use_gpu,
            smooth_window_size=5,
            speech_threshold=activation_threshold,
            pad_start_frame=max(1, int(prefix_padding_duration * fps)),
            min_speech_frame=max(1, int(min_speech_duration * fps)),
            max_speech_frame=int(max_buffered_speech * fps),
            min_silence_frame=max(1, int(min_silence_duration * fps)),
        )

        # Also load in main process for profiling
        from stt_api.vad.fireredvad import FireRedStreamVad
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
            logger.info("FireRedVAD (MP) profiling on load:\n%s", stats.summary())

        return vad

    def __init__(self, *, firered, config, opts) -> None:
        super().__init__(
            capabilities=agents.vad.VADCapabilities(update_interval=self.FRAME_SHIFT_S)
        )
        self._firered = firered
        self._config = config
        self._opts = opts
        self._streams = weakref.WeakSet[FireRedVADMPStream]()
        self._profiling_stats: ProfilingStats | None = None

    @property
    def model(self) -> str:
        return "firered-mp"

    @property
    def provider(self) -> str:
        return "FireRedTeam"

    @property
    def profiling_stats(self) -> ProfilingStats | None:
        return self._profiling_stats

    def run_profiling(self, iterations: int = 200) -> ProfilingStats:
        """Profile by submitting inference to worker processes."""
        rng = np.random.default_rng(42)
        chunk_size = FRAME_SHIFT_SAMPLE
        executor = _get_vad_executor()
        loop = asyncio.new_event_loop()

        stats = ProfilingStats()
        caches = None

        # Warmup
        for _ in range(10):
            chunk = rng.integers(-3000, 3000, size=chunk_size, dtype=np.int16)
            future = executor.submit(_run_inference, chunk, caches)
            _, caches = future.result()

        caches = None  # reset

        total_start = time.perf_counter()
        for i in range(iterations):
            if i % 3 == 0:
                chunk = np.zeros(chunk_size, dtype=np.int16)
            elif i % 3 == 1:
                chunk = rng.integers(-500, 500, size=chunk_size, dtype=np.int16)
            else:
                chunk = rng.integers(-20000, 20000, size=chunk_size, dtype=np.int16)

            t0 = time.perf_counter()
            future = executor.submit(_run_inference, chunk, caches)
            _, caches = future.result()
            elapsed = time.perf_counter() - t0
            stats.latencies.append(elapsed)
            stats.inference_count += 1

        stats.total_time = time.perf_counter() - total_start
        self._profiling_stats = stats
        loop.close()
        return stats

    def stream(self) -> FireRedVADMPStream:
        stream = FireRedVADMPStream(self, self._opts, self._firered, self._config)
        self._streams.add(stream)
        return stream

    def update_options(self, **kwargs) -> None:
        for key, val in kwargs.items():
            if is_given(val) and hasattr(self._opts, key):
                setattr(self._opts, key, val)


class FireRedVADMPStream(agents.vad.VADStream):
    def __init__(self, vad, opts, firered, config) -> None:
        super().__init__(vad)
        self._opts = opts
        self._config = config
        self._firered = firered
        self._loop = asyncio.get_event_loop()

        self._input_sample_rate = 0
        self._speech_buffer: np.ndarray | None = None
        self._speech_buffer_max_reached = False
        self._prefix_padding_samples = 0
        self._audio_buf = np.empty(0, dtype=np.int16)

        # Per-stream stateful caches (passed to/from worker)
        self._model_caches = None
        self._postprocessor = firered.postprocessor
        self._postprocessor.reset()

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
        frame_shift = FRAME_SHIFT_SAMPLE
        executor = _get_vad_executor()

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

            if resampler is not None:
                for rf in resampler.push(input_frame):
                    self._audio_buf = np.append(
                        self._audio_buf, np.array(rf.data, dtype=np.int16)
                    )
            else:
                self._audio_buf = np.append(
                    self._audio_buf, np.array(input_frame.data, dtype=np.int16)
                )

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

            while len(self._audio_buf) >= frame_shift:
                start_time = time.perf_counter()

                chunk = self._audio_buf[:frame_shift].copy()
                self._audio_buf = self._audio_buf[frame_shift:]

                # Run inference in worker process
                p, self._model_caches = await self._loop.run_in_executor(
                    executor, _run_inference, chunk, self._model_caches
                )

                frame_result = self._postprocessor.process_one_frame(p)

                window_duration = FRAME_SHIFT_SAMPLE / SAMPLE_RATE
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
                    return rtc.AudioFrame(
                        sample_rate=self._input_sample_rate,
                        num_channels=1,
                        samples_per_channel=speech_buffer_index,
                        data=self._speech_buffer[:speech_buffer_index].tobytes(),
                    )

                def _reset_write_cursor() -> None:
                    nonlocal speech_buffer_index
                    assert self._speech_buffer is not None
                    if speech_buffer_index <= self._prefix_padding_samples:
                        return
                    padding = self._speech_buffer[
                        speech_buffer_index - self._prefix_padding_samples: speech_buffer_index
                    ]
                    self._speech_buffer_max_reached = False
                    self._speech_buffer[: self._prefix_padding_samples] = padding
                    speech_buffer_index = self._prefix_padding_samples

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

        if (
            self._profiling
            and self._stream_stats is not None
            and self._stream_start_time is not None
        ):
            self._stream_stats.total_time = time.perf_counter() - self._stream_start_time
            logger.info("FireRedVAD (MP) stream:\n%s", self._stream_stats.summary())
