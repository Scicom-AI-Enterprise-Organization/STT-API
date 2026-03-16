"""LiveKit VAD plugin using FireRedVAD with persistent worker processes.

Follows the STT-API pattern: caches NEVER leave the worker process.
Audio goes in → worker does everything → events come out.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import statistics
import time
import uuid
import weakref
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Literal

import numpy as np

from livekit import agents, rtc
from livekit.agents import utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from stt_api.vad.fireredvad import FireRedStreamVadConfig
from stt_api.vad.fireredvad.constants import FRAME_SHIFT_SAMPLE, SAMPLE_RATE

from .log import logger

VAD_WORKERS = int(os.environ.get("VAD_WORKERS", "4"))


@dataclass
class ProfilingStats:
    inference_count: int = 0
    latencies: list[float] = field(default_factory=list)
    total_time: float = 0.0
    ipc_round_trips: int = 0

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
            "FIRERED VAD (PERSISTENT WORKERS)",
            "=" * 50,
            f"  Workers:          {VAD_WORKERS}",
            f"  Total frames:     {self.inference_count}",
            f"  IPC round-trips:  {self.ipc_round_trips}",
            f"  Total time:       {self.total_time:.4f}s",
            f"  Per round-trip latency:",
            f"    Mean:   {mean_ms:.3f}ms",
            f"    Median: {median_ms:.3f}ms",
            f"    p95:    {p95_ms:.3f}ms",
            f"    p99:    {p99_ms:.3f}ms",
            f"    Min:    {min(sorted_lat) * 1000:.3f}ms",
            f"    Max:    {max(sorted_lat) * 1000:.3f}ms",
        ]
        if self.inference_count > 0:
            per_frame = self.total_time / self.inference_count * 1000
            lines.append(f"  Per-frame (amortized): {per_frame:.3f}ms")
        lines.append("=" * 50)
        return "\n".join(lines)


# --- Messages between main process and workers ---
# Input: (stream_id, "audio", audio_chunk_int16)
#        (stream_id, "reset", None)
#        (stream_id, "end", None)
#        ("shutdown", None, None)
#
# Output: (stream_id, frame_results_list)
#   where each result = (probability, is_speech_start, is_speech_end)


def _worker_loop(
    input_q: Queue,
    output_q: Queue,
    use_gpu: bool,
    config_dict: dict,
):
    """Persistent worker process. Caches stay here forever."""
    import torch
    from stt_api.vad.fireredvad import FireRedStreamVad, FireRedStreamVadConfig
    from stt_api.vad.fireredvad.audio_feat import AudioFeat
    from stt_api.vad.fireredvad.detect_model import DetectModel
    from stt_api.vad.fireredvad.stream_vad_postprocessor import StreamVadPostprocessor
    from huggingface_hub import snapshot_download

    # Load model ONCE in this process
    folder_name = snapshot_download(repo_id="FireRedTeam/FireRedVAD")
    model_dir = os.path.join(folder_name, "Stream-VAD")
    cmvn_path = os.path.join(model_dir, "cmvn.ark")

    feat_extractor = AudioFeat(cmvn_path)
    vad_model = DetectModel.from_pretrained(model_dir)
    if use_gpu:
        vad_model.cuda()
    else:
        vad_model.cpu()

    # Per-stream state: caches + postprocessor (all stays in THIS process)
    stream_caches: dict[str, list | None] = {}
    stream_postprocessors: dict[str, StreamVadPostprocessor] = {}

    config = FireRedStreamVadConfig(**config_dict)

    while True:
        msg = input_q.get()
        if msg is None:
            break

        stream_id, cmd, data = msg

        if cmd == "shutdown":
            break

        elif cmd == "reset":
            stream_caches[stream_id] = None
            stream_postprocessors[stream_id] = StreamVadPostprocessor(
                config.smooth_window_size,
                config.speech_threshold,
                config.pad_start_frame,
                config.min_speech_frame,
                config.max_speech_frame,
                config.min_silence_frame,
            )
            output_q.put((stream_id, []))

        elif cmd == "end":
            stream_caches.pop(stream_id, None)
            stream_postprocessors.pop(stream_id, None)
            output_q.put((stream_id, []))

        elif cmd == "audio":
            audio_chunk = data  # np.ndarray int16
            caches = stream_caches.get(stream_id)
            pp = stream_postprocessors.get(stream_id)

            if pp is None:
                # Auto-init if not reset
                stream_caches[stream_id] = None
                caches = None
                pp = StreamVadPostprocessor(
                    config.smooth_window_size,
                    config.speech_threshold,
                    config.pad_start_frame,
                    config.min_speech_frame,
                    config.max_speech_frame,
                    config.min_silence_frame,
                )
                stream_postprocessors[stream_id] = pp

            # Feature extraction + inference (ALL in this process)
            feats, _ = feat_extractor.extract(audio_chunk)
            results = []

            if feats.size(0) > 0:
                with torch.no_grad():
                    probs, new_caches = vad_model.forward(
                        feats.unsqueeze(0), caches=caches
                    )
                stream_caches[stream_id] = new_caches

                # Run each frame through postprocessor
                for t in range(probs.size(1)):
                    p = probs[0, t, 0].item()
                    fr = pp.process_one_frame(p)
                    results.append((p, fr.is_speech_start, fr.is_speech_end))

            output_q.put((stream_id, results))


class _WorkerPool:
    """Manages persistent worker processes."""

    def __init__(self, num_workers: int, use_gpu: bool, config_dict: dict):
        self._workers = []
        self._input_queues: list[Queue] = []
        self._output_queues: list[Queue] = []
        self._next_worker = 0

        ctx = multiprocessing.get_context("spawn")

        for _ in range(num_workers):
            in_q = ctx.Queue()
            out_q = ctx.Queue()
            p = ctx.Process(
                target=_worker_loop,
                args=(in_q, out_q, use_gpu, config_dict),
                daemon=True,
            )
            p.start()
            self._workers.append(p)
            self._input_queues.append(in_q)
            self._output_queues.append(out_q)

        logger.info(f"FireRedVAD worker pool started with {num_workers} processes")

    def assign_worker(self) -> int:
        """Round-robin assign a stream to a worker."""
        idx = self._next_worker % len(self._workers)
        self._next_worker += 1
        return idx

    def send(self, worker_idx: int, stream_id: str, cmd: str, data=None):
        self._input_queues[worker_idx].put((stream_id, cmd, data))

    def recv(self, worker_idx: int) -> tuple[str, list]:
        return self._output_queues[worker_idx].get()

    def shutdown(self):
        for q in self._input_queues:
            q.put(None)
        for w in self._workers:
            w.join(timeout=5)


_pool: _WorkerPool | None = None


def _get_pool(use_gpu: bool, config_dict: dict) -> _WorkerPool:
    global _pool
    if _pool is None:
        _pool = _WorkerPool(VAD_WORKERS, use_gpu, config_dict)
    return _pool


class VAD(agents.vad.VAD):
    """LiveKit VAD using FireRedVAD with persistent worker processes."""

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

        config_dict = {
            "use_gpu": config.use_gpu,
            "smooth_window_size": config.smooth_window_size,
            "speech_threshold": config.speech_threshold,
            "pad_start_frame": config.pad_start_frame,
            "min_speech_frame": config.min_speech_frame,
            "max_speech_frame": config.max_speech_frame,
            "min_silence_frame": config.min_silence_frame,
        }

        # Start persistent workers
        _get_pool(use_gpu, config_dict)

        opts = _Opts(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            sample_rate=sample_rate,
            profiling=profiling,
        )

        return cls(config=config, config_dict=config_dict, opts=opts, use_gpu=use_gpu)

    def __init__(self, *, config, config_dict, opts, use_gpu) -> None:
        super().__init__(
            capabilities=agents.vad.VADCapabilities(update_interval=self.FRAME_SHIFT_S)
        )
        self._config = config
        self._config_dict = config_dict
        self._opts = opts
        self._use_gpu = use_gpu
        self._streams = weakref.WeakSet[_WorkerStream]()

    @property
    def model(self) -> str:
        return "firered-worker"

    @property
    def provider(self) -> str:
        return "FireRedTeam"

    def stream(self) -> _WorkerStream:
        pool = _get_pool(self._use_gpu, self._config_dict)
        stream = _WorkerStream(self, self._opts, pool)
        self._streams.add(stream)
        return stream

    def update_options(self, **kwargs) -> None:
        for key, val in kwargs.items():
            if is_given(val) and hasattr(self._opts, key):
                setattr(self._opts, key, val)


@dataclass
class _Opts:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    sample_rate: int
    profiling: bool = False


# How much audio to batch before sending to worker
BATCH_DURATION_MS = int(os.environ.get("VAD_BATCH_MS", "200"))


class _WorkerStream(agents.vad.VADStream):
    def __init__(self, vad: VAD, opts: _Opts, pool: _WorkerPool) -> None:
        super().__init__(vad)
        self._opts = opts
        self._pool = pool
        self._loop = asyncio.get_event_loop()

        self._stream_id = str(uuid.uuid4())
        self._worker_idx = pool.assign_worker()

        self._input_sample_rate = 0
        self._speech_buffer: np.ndarray | None = None
        self._speech_buffer_max_reached = False
        self._prefix_padding_samples = 0
        self._audio_buf = np.empty(0, dtype=np.int16)
        self._batch_samples = int(SAMPLE_RATE * BATCH_DURATION_MS / 1000)

        self._profiling = opts.profiling
        self._stream_stats = ProfilingStats() if self._profiling else None
        self._stream_start_time: float | None = None

    @property
    def stream_stats(self) -> ProfilingStats | None:
        return self._stream_stats

    async def _send_and_recv(self, cmd: str, data=None) -> list:
        """Send to worker and receive response, non-blocking."""
        self._pool.send(self._worker_idx, self._stream_id, cmd, data)
        # Use run_in_executor to avoid blocking the event loop on queue.get()
        _, results = await self._loop.run_in_executor(
            None, self._pool.recv, self._worker_idx
        )
        return results

    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        if self._profiling:
            self._stream_start_time = time.perf_counter()

        # Reset worker state for this stream
        await self._send_and_recv("reset")

        speech_buffer_index: int = 0
        pub_speaking = False
        pub_speech_duration = 0.0
        pub_silence_duration = 0.0
        pub_current_sample = 0
        pub_timestamp = 0.0

        resampler: rtc.AudioResampler | None = None
        frame_shift = FRAME_SHIFT_SAMPLE

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
            avail = len(self._speech_buffer) - speech_buffer_index
            to_copy = min(len(input_data), avail)
            if to_copy > 0:
                self._speech_buffer[
                    speech_buffer_index: speech_buffer_index + to_copy
                ] = input_data[:to_copy]
                speech_buffer_index += to_copy

            # Batch: only send when we have enough audio
            if len(self._audio_buf) < self._batch_samples:
                continue

            batch = self._audio_buf[: self._batch_samples].copy()
            self._audio_buf = self._audio_buf[self._batch_samples:]

            # ONE IPC round-trip: audio in → (probs + events) out
            # Caches STAY in the worker
            t0 = time.perf_counter()
            results = await self._send_and_recv("audio", batch)
            ipc_duration = time.perf_counter() - t0

            if self._profiling and self._stream_stats is not None:
                self._stream_stats.latencies.append(ipc_duration)
                self._stream_stats.inference_count += len(results)
                self._stream_stats.ipc_round_trips += 1

            window_duration = FRAME_SHIFT_SAMPLE / SAMPLE_RATE

            for p, is_start, is_end in results:
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
                        inference_duration=ipc_duration / max(len(results), 1),
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

                if is_start and not pub_speaking:
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
                elif is_end and pub_speaking:
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

        # Clean up worker state
        await self._send_and_recv("end")

        if (
            self._profiling
            and self._stream_stats is not None
            and self._stream_start_time is not None
        ):
            self._stream_stats.total_time = time.perf_counter() - self._stream_start_time
            logger.info("FireRedVAD (worker):\n%s", self._stream_stats.summary())
