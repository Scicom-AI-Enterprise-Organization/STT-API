"""
VAD Benchmark: Sequential vs Parallel Processing

This script benchmarks two approaches:
1. Sequential: Current approach - process frames one by one
2. Parallel: Use ProcessPoolExecutor to process audio in parallel

Usage:
    # Run with default settings
    python benchmark_vad.py

    # Run with specific audio file
    python benchmark_vad.py --audio test_audio/masak.mp3

    # Run with specific number of workers
    python benchmark_vad.py --workers 4

Environment Variables (for parallel mode):
    OMP_NUM_THREADS=1       # Limit ONNX threads per process
    OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1
"""

import os
import sys
import time
import argparse
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import librosa

# Configuration
SAMPLE_RATE = 16000
FRAME_SIZE = 512
MAX_CHUNK_LENGTH = 25.0  # seconds
MINIMUM_SILENT_MS = 200
MINIMUM_TRIGGER_VAD_MS = 1500

# Global VAD model (loaded per process)
_silero_model = None


def get_silero_model():
    """Get or load the Silero VAD model (cached per process)."""
    global _silero_model
    if _silero_model is None:
        from silero_vad import load_silero_vad

        _silero_model = load_silero_vad(onnx=True)
    return _silero_model


def init_worker():
    """Initialize VAD model in worker process."""
    get_silero_model()
    print(f"Worker {os.getpid()} initialized with Silero VAD")


def process_vad_sequential(audio_data: np.ndarray) -> List[Tuple]:
    """
    Process audio with VAD sequentially (current approach).

    Returns:
        List of (wav_data, start_ts, end_ts, silence_ratio) tuples
    """
    silero = get_silero_model()
    silero.reset_states()

    chunks = []
    wav_data = np.array([], dtype=np.float32)
    last_timestamp = 0.0
    total_silent = 0
    total_silent_frames = 0
    total_frames = 0

    num_frames = len(audio_data) // FRAME_SIZE

    for i in range(num_frames):
        start_idx = i * FRAME_SIZE
        end_idx = start_idx + FRAME_SIZE
        frame = audio_data[start_idx:end_idx]

        if len(frame) < FRAME_SIZE:
            frame = np.pad(frame, (0, FRAME_SIZE - len(frame)), mode="constant")

        total_frames += 1

        frame_pt = torch.from_numpy(frame).unsqueeze(0)
        vad_score = silero(frame_pt, sr=SAMPLE_RATE).numpy()[0][0]
        vad = vad_score > 0.5

        if vad:
            total_silent = 0
        else:
            total_silent += len(frame)
            total_silent_frames += 1

        wav_data = np.concatenate([wav_data, frame])
        audio_len = len(wav_data) / SAMPLE_RATE
        audio_len_ms = audio_len * 1000
        silent_len = (total_silent / SAMPLE_RATE) * 1000
        silence_ratio = total_silent_frames / total_frames if total_frames > 0 else 0

        vad_trigger = (
            audio_len_ms >= MINIMUM_TRIGGER_VAD_MS and silent_len >= MINIMUM_SILENT_MS
        )

        if vad_trigger or audio_len >= MAX_CHUNK_LENGTH:
            start_ts = last_timestamp
            end_ts = last_timestamp + audio_len
            chunks.append((len(wav_data), start_ts, end_ts, silence_ratio))

            last_timestamp = end_ts
            total_silent = 0
            total_silent_frames = 0
            total_frames = 0
            wav_data = np.array([], dtype=np.float32)

    # Handle remaining samples
    if len(wav_data) > 0:
        audio_len = len(wav_data) / SAMPLE_RATE
        silence_ratio = total_silent_frames / total_frames if total_frames > 0 else 0
        start_ts = last_timestamp
        end_ts = last_timestamp + audio_len
        chunks.append((len(wav_data), start_ts, end_ts, silence_ratio))

    return chunks


def process_audio_chunk_worker(args: Tuple) -> List[Tuple]:
    """
    Worker function to process a portion of audio.
    Called by ProcessPoolExecutor.
    """
    audio_segment, start_offset = args

    silero = get_silero_model()
    silero.reset_states()

    chunks = []
    wav_data = np.array([], dtype=np.float32)
    last_timestamp = start_offset
    total_silent = 0
    total_silent_frames = 0
    total_frames = 0

    num_frames = len(audio_segment) // FRAME_SIZE

    for i in range(num_frames):
        start_idx = i * FRAME_SIZE
        end_idx = start_idx + FRAME_SIZE
        frame = audio_segment[start_idx:end_idx]

        if len(frame) < FRAME_SIZE:
            frame = np.pad(frame, (0, FRAME_SIZE - len(frame)), mode="constant")

        total_frames += 1

        frame_pt = torch.from_numpy(frame).unsqueeze(0)
        vad_score = silero(frame_pt, sr=SAMPLE_RATE).numpy()[0][0]
        vad = vad_score > 0.5

        if vad:
            total_silent = 0
        else:
            total_silent += len(frame)
            total_silent_frames += 1

        wav_data = np.concatenate([wav_data, frame])
        audio_len = len(wav_data) / SAMPLE_RATE
        audio_len_ms = audio_len * 1000
        silent_len = (total_silent / SAMPLE_RATE) * 1000
        silence_ratio = total_silent_frames / total_frames if total_frames > 0 else 0

        vad_trigger = (
            audio_len_ms >= MINIMUM_TRIGGER_VAD_MS and silent_len >= MINIMUM_SILENT_MS
        )

        if vad_trigger or audio_len >= MAX_CHUNK_LENGTH:
            start_ts = last_timestamp
            end_ts = last_timestamp + audio_len
            chunks.append((len(wav_data), start_ts, end_ts, silence_ratio))

            last_timestamp = end_ts
            total_silent = 0
            total_silent_frames = 0
            total_frames = 0
            wav_data = np.array([], dtype=np.float32)

    # Handle remaining samples
    if len(wav_data) > 0:
        audio_len = len(wav_data) / SAMPLE_RATE
        silence_ratio = total_silent_frames / total_frames if total_frames > 0 else 0
        start_ts = last_timestamp
        end_ts = last_timestamp + audio_len
        chunks.append((len(wav_data), start_ts, end_ts, silence_ratio))

    return chunks


def process_vad_parallel(audio_data: np.ndarray, num_workers: int = 4) -> List[Tuple]:
    """
    Process audio with VAD using multiple processes.

    Splits audio into segments, processes each in parallel, then combines results.

    Note: This approach has a limitation - VAD state is not shared between segments,
    which may cause slightly different chunking at segment boundaries.

    Returns:
        List of (wav_data_len, start_ts, end_ts, silence_ratio) tuples
    """
    audio_duration = len(audio_data) / SAMPLE_RATE
    segment_duration = audio_duration / num_workers
    segment_samples = int(segment_duration * SAMPLE_RATE)

    # Split audio into segments for each worker
    segments = []
    for i in range(num_workers):
        start_sample = i * segment_samples
        end_sample = (
            start_sample + segment_samples if i < num_workers - 1 else len(audio_data)
        )
        start_offset = start_sample / SAMPLE_RATE
        segments.append((audio_data[start_sample:end_sample], start_offset))

    # Process segments in parallel
    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=init_worker
    ) as executor:
        results = list(executor.map(process_audio_chunk_worker, segments))

    # Combine results
    all_chunks = []
    for segment_chunks in results:
        all_chunks.extend(segment_chunks)

    return all_chunks


def run_benchmark(audio_path: str, num_workers: int = 4, num_runs: int = 3):
    """Run the benchmark comparing sequential vs parallel VAD."""

    print(f"\n{'='*60}")
    print("VAD BENCHMARK: Sequential vs Parallel")
    print(f"{'='*60}")

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio_data, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio_duration = len(audio_data) / SAMPLE_RATE
    print(f"Audio duration: {audio_duration:.2f}s ({len(audio_data)} samples)")
    print(f"Number of workers for parallel: {num_workers}")
    print(f"Number of runs: {num_runs}")

    # Check thread settings
    omp_threads = os.environ.get("OMP_NUM_THREADS", "not set")
    openblas_threads = os.environ.get("OPENBLAS_NUM_THREADS", "not set")
    print("\nThread settings:")
    print(f"  OMP_NUM_THREADS: {omp_threads}")
    print(f"  OPENBLAS_NUM_THREADS: {openblas_threads}")

    # Warmup
    print("\n--- Warmup ---")
    print("Running warmup (sequential)...")
    _ = process_vad_sequential(audio_data[: SAMPLE_RATE * 10])  # First 10 seconds
    print("Warmup complete.")

    # Benchmark Sequential
    print("\n--- Sequential VAD ---")
    sequential_times = []
    sequential_chunks = None

    for i in range(num_runs):
        start = time.perf_counter()
        chunks = process_vad_sequential(audio_data)
        elapsed = time.perf_counter() - start
        sequential_times.append(elapsed)
        sequential_chunks = chunks
        print(f"  Run {i+1}: {elapsed:.3f}s ({len(chunks)} chunks)")

    seq_avg = sum(sequential_times) / len(sequential_times)
    seq_min = min(sequential_times)
    print(f"  Average: {seq_avg:.3f}s, Min: {seq_min:.3f}s")

    # Benchmark Parallel
    print(f"\n--- Parallel VAD ({num_workers} workers) ---")
    parallel_times = []
    parallel_chunks = None

    for i in range(num_runs):
        start = time.perf_counter()
        chunks = process_vad_parallel(audio_data, num_workers=num_workers)
        elapsed = time.perf_counter() - start
        parallel_times.append(elapsed)
        parallel_chunks = chunks
        print(f"  Run {i+1}: {elapsed:.3f}s ({len(chunks)} chunks)")

    par_avg = sum(parallel_times) / len(parallel_times)
    par_min = min(parallel_times)
    print(f"  Average: {par_avg:.3f}s, Min: {par_min:.3f}s")

    # Summary
    speedup = seq_avg / par_avg if par_avg > 0 else 0

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\nAudio duration: {audio_duration:.2f}s")
    print("\n| Method     | Avg Time | Min Time | Chunks | Speedup |")
    print("|------------|----------|----------|--------|---------|")
    print(
        f"| Sequential | {seq_avg:>7.3f}s | {seq_min:>7.3f}s | {len(sequential_chunks):>6} | 1.00x   |"
    )
    print(
        f"| Parallel   | {par_avg:>7.3f}s | {par_min:>7.3f}s | {len(parallel_chunks):>6} | {speedup:>5.2f}x  |"
    )

    if speedup > 1:
        print(f"\n✅ Parallel is {speedup:.2f}x FASTER than sequential")
    elif speedup < 1:
        print(f"\n❌ Parallel is {1/speedup:.2f}x SLOWER than sequential")
    else:
        print("\n⚠️ No significant difference")

    # RTF comparison
    seq_rtf = seq_avg / audio_duration
    par_rtf = par_avg / audio_duration
    print("\nVAD RTF (lower is better):")
    print(f"  Sequential: {seq_rtf:.4f} ({1/seq_rtf:.1f}x faster than real-time)")
    print(f"  Parallel:   {par_rtf:.4f} ({1/par_rtf:.1f}x faster than real-time)")

    print(f"\n{'='*60}")

    return {
        "sequential": {
            "avg": seq_avg,
            "min": seq_min,
            "chunks": len(sequential_chunks),
        },
        "parallel": {"avg": par_avg, "min": par_min, "chunks": len(parallel_chunks)},
        "speedup": speedup,
        "audio_duration": audio_duration,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark VAD: Sequential vs Parallel"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="test_audio/masak.mp3",
        help="Path to audio file for benchmarking",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel mode",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    run_benchmark(args.audio, num_workers=args.workers, num_runs=args.runs)
