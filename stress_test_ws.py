"""
STT-API WebSocket Stress Test

Simulates multiple concurrent WebSocket clients streaming audio from a file,
measures transcription latency, throughput, and success rate.

Usage:
    # Direct testing:
    STT_API_URL=http://localhost:9091 AUDIO_FILE=test_audio/masak.mp3 python stress_test_ws.py

    # Docker:
    docker compose -f stress-test.yaml run --rm stress-test uv run python stress_test_ws.py

Environment Variables:
    WARMUP_COUNT: Number of warmup connections (default: 2)
    CONCURRENCY: Number of concurrent WebSocket clients (default: 10)
    STT_API_URL: URL of the STT API (default: http://stt-api:9091)
    AUDIO_FILE: Path to audio file for testing (default: /app/test_audio/masak.mp3)
    LANGUAGE: Language hint (default: ms)
    CHUNK_MS: Milliseconds of audio per WebSocket send (default: 100)
    SAMPLE_RATE: Audio sample rate in Hz (default: 16000)
"""

import os
import asyncio
import time
import json
import statistics
import numpy as np
import librosa
import websockets

WARMUP_COUNT = int(os.environ.get("WARMUP_COUNT", "2"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "10"))
STT_API_URL = os.environ.get("STT_API_URL", "http://stt-api:9091")
AUDIO_FILE = os.environ.get("AUDIO_FILE", "/app/test_audio/masak.mp3")
LANGUAGE = os.environ.get("LANGUAGE", "ms")
CHUNK_MS = int(os.environ.get("CHUNK_MS", "100"))
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))


def get_audio_duration(file_path: str) -> float:
    try:
        return librosa.get_duration(path=file_path)
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
        return 0.0


def load_audio(file_path: str) -> np.ndarray:
    audio_data, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio_data.astype(np.float32)


def build_ws_url() -> str:
    base = STT_API_URL.replace("http://", "ws://").replace("https://", "wss://")
    return f"{base}/ws?language={LANGUAGE}"


async def stream_ws_websockets(audio_data: np.ndarray, client_id: int) -> dict:
    """Single WebSocket client using the websockets library."""
    ws_url = build_ws_url()
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
    results = []
    segments_received = 0
    silent_count = 0
    error_msg = None

    start_time = time.time()
    first_transcription_time = None
    last_message_time = None

    try:
        async with websockets.connect(ws_url) as ws:
            receive_task = None
            done_sending = asyncio.Event()

            async def receiver():
                nonlocal segments_received, silent_count, first_transcription_time, last_message_time
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        last_message_time = time.time()
                        data = json.loads(msg)
                        if data.get("type") == "transcription":
                            if first_transcription_time is None:
                                first_transcription_time = time.time()
                            segs = data.get("segments", [])
                            segments_received += len(segs) if segs else 1
                            results.append(data)
                        elif data.get("type") == "silent":
                            silent_count += 1
                        elif data.get("error"):
                            results.append(data)
                except asyncio.TimeoutError:
                    pass
                except websockets.exceptions.ConnectionClosed:
                    pass

            receive_task = asyncio.create_task(receiver())

            offset = 0
            while offset < len(audio_data):
                end = min(offset + chunk_samples, len(audio_data))
                chunk = audio_data[offset:end]
                await ws.send(chunk.tobytes())
                offset = end
                await asyncio.sleep(0.0)

            await receive_task

    except Exception as e:
        error_msg = str(e)

    end_time = last_message_time if last_message_time else time.time()
    total_time = end_time - start_time
    ttft = (first_transcription_time - start_time) if first_transcription_time else None

    return {
        "client_id": client_id,
        "total_time": total_time,
        "ttft": ttft,
        "segments_received": segments_received,
        "silent_count": silent_count,
        "results": results,
        "success": error_msg is None and segments_received > 0,
        "error": error_msg,
    }


async def stream_single(audio_data: np.ndarray, client_id: int) -> dict:
    return await stream_ws_websockets(audio_data, client_id)


async def run_stress_test(concurrency: int, audio_data: np.ndarray) -> list:
    tasks = [
        stream_single(audio_data, i)
        for i in range(concurrency)
    ]
    return await asyncio.gather(*tasks)


def print_report(results: list, audio_duration: float, concurrency: int):
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print("\n" + "=" * 60)
    print("STT-API WEBSOCKET STRESS TEST REPORT")
    print("=" * 60)

    print("\n--- Test Configuration ---")
    print(f"Concurrency: {concurrency}")
    print(f"Audio Duration: {audio_duration:.2f}s")
    print(f"Language: {LANGUAGE}")
    print(f"Chunk Size: {CHUNK_MS}ms")
    print(f"Total Clients: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success Rate: {len(successful)/len(results)*100:.1f}%")

    if failed:
        print("\n--- Failed Clients ---")
        for r in failed[:5]:
            print(f"  Client {r['client_id']}: {r['error']}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    if successful:
        total_times = [r["total_time"] for r in successful]
        total_segments = sum(r["segments_received"] for r in successful)
        total_silent = sum(r["silent_count"] for r in successful)
        ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]

        times_sorted = sorted(total_times)
        count = len(times_sorted)

        avg = statistics.mean(times_sorted)
        p50 = statistics.median(times_sorted)
        p90 = times_sorted[int(0.90 * count) - 1] if count >= 10 else times_sorted[-1]
        p95 = times_sorted[int(0.95 * count) - 1] if count >= 20 else times_sorted[-1]
        p99 = times_sorted[int(0.99 * count) - 1] if count >= 100 else times_sorted[-1]

        print("\n--- Total Session Time ---")
        print(f"Min: {min(times_sorted):.3f}s")
        print(f"Max: {max(times_sorted):.3f}s")
        print(f"Avg: {avg:.3f}s")
        print(f"P50: {p50:.3f}s")
        print(f"P90: {p90:.3f}s")
        print(f"P95: {p95:.3f}s")
        print(f"P99: {p99:.3f}s")

        if ttfts:
            ttfts_sorted = sorted(ttfts)
            ttft_count = len(ttfts_sorted)
            ttft_avg = statistics.mean(ttfts_sorted)
            ttft_p50 = statistics.median(ttfts_sorted)
            ttft_p90 = ttfts_sorted[int(0.90 * ttft_count) - 1] if ttft_count >= 10 else ttfts_sorted[-1]

            print("\n--- Time to First Transcription (TTFT) ---")
            print(f"Min: {min(ttfts_sorted):.3f}s")
            print(f"Max: {max(ttfts_sorted):.3f}s")
            print(f"Avg: {ttft_avg:.3f}s")
            print(f"P50: {ttft_p50:.3f}s")
            print(f"P90: {ttft_p90:.3f}s")

        print("\n--- Segments ---")
        print(f"Total Transcription Segments: {total_segments}")
        print(f"Total Silent Segments: {total_silent}")
        print(f"Avg Segments/Client: {total_segments / len(successful):.1f}")

        if audio_duration > 0:
            rtfs = [t / audio_duration for t in times_sorted]
            print("\n--- Real-Time Factor (RTF) ---")
            print("(RTF < 1.0 means faster than real-time)")
            print(f"Min RTF: {min(rtfs):.3f}")
            print(f"Max RTF: {max(rtfs):.3f}")
            print(f"Avg RTF: {statistics.mean(rtfs):.3f}")
            print(f"P50 RTF: {statistics.median(rtfs):.3f}")

        wall_time = max(times_sorted)
        print("\n--- Throughput ---")
        print(f"Total Wall Time: {wall_time:.3f}s")
        print(f"Clients/second: {len(successful) / wall_time:.2f}")
        print(f"Audio seconds processed/second: {len(successful) * audio_duration / wall_time:.2f}")

    print("\n" + "=" * 60)


async def main():
    if websockets is None and aiohttp is None:
        print("Error: Either 'websockets' or 'aiohttp' must be installed")
        print("  pip install websockets  OR  pip install aiohttp")
        return

    lib = "websockets" if websockets is not None else "aiohttp"
    print(f"Using WebSocket library: {lib}")

    print(f"Loading audio file: {AUDIO_FILE}")
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file not found: {AUDIO_FILE}")
        if os.path.exists("test_audio"):
            print("Available test audio files:")
            for f in os.listdir("test_audio"):
                print(f"  - test_audio/{f}")
        return

    audio_data = load_audio(AUDIO_FILE)
    audio_duration = get_audio_duration(AUDIO_FILE)

    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Audio samples: {len(audio_data)}")
    print(f"API URL: {STT_API_URL}")
    print(f"WebSocket URL: {build_ws_url()}")

    # Warmup
    print(f"\n--- Warmup ({WARMUP_COUNT} clients) ---")
    for i in range(WARMUP_COUNT):
        results = await run_stress_test(1, audio_data)
        r = results[0]
        status = "ok" if r["success"] else "FAIL"
        segs = r["segments_received"]
        ttft_str = f", TTFT: {r['ttft']:.3f}s" if r["ttft"] else ""
        print(f"  Warmup {i+1}: {r['total_time']:.3f}s, {segs} segments{ttft_str} [{status}]")
        if r["error"]:
            print(f"    Error: {r['error']}")

    # Stress test
    print(f"\n--- Running Stress Test ({CONCURRENCY} concurrent clients) ---")
    start = time.time()
    results = await run_stress_test(CONCURRENCY, audio_data)
    wall_time = time.time() - start
    print(f"Completed in {wall_time:.3f}s")

    # Report
    print_report(results, audio_duration, CONCURRENCY)


if __name__ == "__main__":
    asyncio.run(main())
