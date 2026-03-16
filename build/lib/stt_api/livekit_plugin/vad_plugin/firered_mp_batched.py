"""LiveKit VAD plugin using FireRedVAD with batched ProcessPoolExecutor.

Instead of sending one 10ms frame per IPC call, accumulates frames into
larger batches (e.g., 200ms) and sends the whole batch to a worker.
The worker does fbank extraction + DFSMN inference on all frames at once
and returns all probabilities in one shot — amortizing IPC overhead.
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

VAD_WORKERS = int(os.environ.get("VAD_WORKERS", "4"))

# Batch size in seconds — how much audio to accumulate before sending to worker
BATCH_DURATION_MS = int(os.environ.get("VAD_BATCH_MS", "200"))

# --- Worker process globals ---
_worker_feat_extractor = None
_worker_vad_model = None


def _init_worker(use_gpu: bool):
    """Initialize FireRedVAD in each worker process."""
    global _worker_feat_extractor, _worker_vad_model

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

    logger.info(f"FireRedVAD batched worker {os.getpid()} initialized")


def _run_batch_inference(audio_chunk: np.ndarray, caches) -> tuple[list[float], list]:
    """Run fbank + DFSMN on an entire audio chunk. Returns per-frame probabilities."""
    global _worker_feat_extractor, _worker_vad_model

    feats, _ = _worker_feat_extractor.extract(audio_chunk)
    probs_list = []
    new_caches = caches

    if feats.size(0) > 0:
        with torch.no_grad():
            probs, new_caches = _worker_vad_model.forward(
                feats.unsqueeze(0), caches=caches
            )
            # probs shape: (1, T, 1) — extract all frame probabilities
            probs_list = probs[0, :, 0].tolist()

    return probs_list, new_caches


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
        logger.info(
            f"FireRedVAD batched ProcessPool: {VAD_WORKERS} workers, "
            f"batch={BATCH_DURATION_MS}ms"
        )
    return _vad_executor


@dataclass
class ProfilingStats:
    inference_count: int = 0
    latencies: list[float] = field(default_factory=list)
    total_time: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)

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
        avg_batch = statistics.mean(self.batch_sizes) if self.batch_sizes else 0
        lines = [
            "=" * 50,
            f"FIRERED VAD (MP BATCHED, {BATCH_DURATION_MS}ms)",
            "=" * 50,
            f"  Workers:       {VAD_WORKERS}",
            f"  Batch size:    {BATCH_DURATION_MS}ms ({avg_batch:.0f} frames avg)",
            f"  IPC calls:     {len(self.latencies)}",
            f"  Total frames:  {self.inference_count}",
            f"  Total time:    {self.total_time:.4f}s",
            f"  Per-IPC call latency:",
            f"    Mean:    {mean_ms:.3f}ms",
            f"    Median:  {median_ms:.3f}ms",
            f"    p95:     {p95_ms:.3f}ms",
            f"    p99:     {p99_ms:.3f}ms",
            f"    Min:     {min(sorted_lat) * 1000:.3f}ms",
            f"    Max:     {max(sorted_lat) * 1000:.3f}ms",
        ]
        if self.inference_count > 0:
            per_frame = self.total_time / self.inference_count * 1000
            lines.append(f"  Per-frame (amortized): {per_frame:.3f}ms")
        lines.append("=" * 50)
        return "\n".join(lines)


class VAD(agents.vad.VAD):
    """LiveKit VAD using FireRedVAD with batched ProcessPoolExecutor."""

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
        return vad

    def __init__(self, *, firered, config, opts) -> None:
        super().__init__(
            capabilities=agents.vad.VADCapabilities(update_interval=self.FRAME_SHIFT_S)
        )
        self._firered = firered
        self._config = config
        self._opts = opts
        self._streams = weakref.WeakSet[_BatchedStream]()

    @property
    def model(self) -> str:
        return "firered-mp-batched"

    @property
    def provider(self) -> str:
        return "FireRedTeam"

    def stream(self) -> _BatchedStream:
        stream = _BatchedStream(self, self._opts, self._firered, self._config)
        self._streams.add(stream)
        return stream

    def update_options(self, **kwargs) -> None:
        for key, val in kwargs.items():
            if is_given(val) and hasattr(self._opts, key):
                setattr(self._opts, key, val)


@dataclass
class _FireRedVADOptions:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    sample_rate: int
    profiling: bool = False


class _BatchedStream(agents.vad.VADStream):
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

        # Audio accumulator — we collect BATCH_DURATION_MS worth before IPC
        self._audio_buf = np.empty(0, dtype=np.int16)
        self._batch_samples = int(SAMPLE_RATE * BATCH_DURATION_MS / 1000)

        self._model_caches = None
        self._postprocessor = firered.postprocessor
        self._postprocessor.reset()

        self._profiling = opts.profiling
        self._stream_stats = ProfilingStats() if self._profiling else None
        self._stream_start_time: float | None = None

    @property
    def stream_stats(self) -> ProfilingStats | None:
        return self._stream_stats

    async def _process_batch(self, batch: np.ndarray, executor) -> list[float]:
        """Send a batch to worker, get back per-frame probabilities."""
        probs, self._model_caches = await self._loop.run_in_executor(
            executor, _run_batch_inference, batch, self._model_caches
        )
        return probs

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

            # Accumulate resampled audio
            if resampler is not None:
                for rf in resampler.push(input_frame):
                    self._audio_buf = np.append(
                        self._audio_buf, np.array(rf.data, dtype=np.int16)
                    )
            else:
                self._audio_buf = np.append(
                    self._audio_buf, np.array(input_frame.data, dtype=np.int16)
                )

            # Copy to speech buffer
            input_data = np.array(input_frame.data, dtype=np.int16)
            avail = len(self._speech_buffer) - speech_buffer_index
            to_copy = min(len(input_data), avail)
            if to_copy > 0:
                self._speech_buffer[
                    speech_buffer_index: speech_buffer_index + to_copy
                ] = input_data[:to_copy]
                speech_buffer_index += to_copy

            # Only send to worker when we have a full batch
            if len(self._audio_buf) < self._batch_samples:
                continue

            # Take the batch
            batch = self._audio_buf[: self._batch_samples].copy()
            self._audio_buf = self._audio_buf[self._batch_samples:]

            # One IPC call for the entire batch
            t0 = time.perf_counter()
            probs = await self._process_batch(batch, executor)
            ipc_duration = time.perf_counter() - t0

            if self._profiling and self._stream_stats is not None:
                self._stream_stats.latencies.append(ipc_duration)
                self._stream_stats.inference_count += len(probs)
                self._stream_stats.batch_sizes.append(len(probs))

            # Process each frame's probability through postprocessor
            window_duration = FRAME_SHIFT_SAMPLE / SAMPLE_RATE

            for p in probs:
                frame_result = self._postprocessor.process_one_frame(p)

                pub_current_sample += frame_shift
                pub_timestamp += window_duration

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
                        inference_duration=ipc_duration / len(probs),
                        frames=[],
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
            logger.info("FireRedVAD (MP batched):\n%s", self._stream_stats.summary())
