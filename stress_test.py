"""
STT-API Stress Test (Container Testing)

Usage:
    # Add stress-test service to docker-compose and run:
    docker compose run --rm stress-test

    # Or for direct testing:
    STT_API_URL=http://localhost:9091 AUDIO_FILE=test_audio/masak.mp3 python stress_test.py

    # Test with diarization:
    DIARIZATION_MODE=online SPEAKER_SIMILARITY=0.75 docker compose run --rm stress-test

Environment Variables:
    WARMUP_COUNT: Number of warmup requests (default: 3)
    CONCURRENCY: Number of concurrent requests (default: 50)
    STT_API_URL: URL of the STT API (default: http://stt-api:9091)
    AUDIO_FILE: Path to audio file for testing (default: /app/test_audio/masak.mp3)
    DIARIZATION_MODE: Diarization mode - none, online, offline (default: none)
    SPEAKER_SIMILARITY: Speaker clustering threshold for online mode (default: 0.75)
    SPEAKER_MAX_N: Maximum speakers for online mode (default: 10)
"""

import os
import asyncio
import time
import aiohttp
import statistics
import librosa

WARMUP_COUNT = int(os.environ.get("WARMUP_COUNT", "3"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "50"))
STT_API_URL = os.environ.get("STT_API_URL", "http://stt-api:9091")
AUDIO_FILE = os.environ.get("AUDIO_FILE", "/app/test_audio/masak.mp3")
DIARIZATION_MODE = os.environ.get("DIARIZATION_MODE", "none")
SPEAKER_SIMILARITY = os.environ.get("SPEAKER_SIMILARITY", "0.75")
SPEAKER_MAX_N = os.environ.get("SPEAKER_MAX_N", "10")


def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds using librosa."""
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
        return 0.0


async def transcribe_request(
    session: aiohttp.ClientSession, audio_bytes: bytes, filename: str
) -> tuple:
    """
    Send a single transcription request.

    Returns:
        Tuple of (processing_time, success, error_message)
    """
    url = f"{STT_API_URL}/audio/transcriptions"

    start_time = time.time()

    try:
        data = aiohttp.FormData()
        data.add_field(
            "file", audio_bytes, filename=filename, content_type="audio/mpeg"
        )
        data.add_field("language", "ms")
        data.add_field("response_format", "verbose_json")
        data.add_field("diarization", DIARIZATION_MODE)
        if DIARIZATION_MODE in ("online", "offline"):
            data.add_field("speaker_similarity", SPEAKER_SIMILARITY)
            data.add_field("speaker_max_n", SPEAKER_MAX_N)

        async with session.post(url, data=data) as response:
            if response.status == 200:
                await response.json()  # Consume response body
                processing_time = time.time() - start_time
                return (processing_time, True, None)
            else:
                error_text = await response.text()
                processing_time = time.time() - start_time
                return (
                    processing_time,
                    False,
                    f"HTTP {response.status}: {error_text[:100]}",
                )
    except Exception as e:
        processing_time = time.time() - start_time
        return (processing_time, False, str(e))


async def run_stress_test(concurrency: int, audio_bytes: bytes, filename: str) -> list:
    """Run stress test with specified concurrency."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            transcribe_request(session, audio_bytes, filename)
            for _ in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
    return results


def print_report(results: list, audio_duration: float, concurrency: int):
    """Print stress test report with SLO metrics."""
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print("\n" + "=" * 50)
    print("STT-API STRESS TEST REPORT")
    print("=" * 50)

    print("\n--- Test Configuration ---")
    print(f"Concurrency: {concurrency}")
    print(f"Audio Duration: {audio_duration:.2f}s")
    print(f"Diarization: {DIARIZATION_MODE}")
    if DIARIZATION_MODE in ("online", "offline"):
        print(f"  Speaker Similarity: {SPEAKER_SIMILARITY}")
        print(f"  Max Speakers: {SPEAKER_MAX_N}")
    print(f"Total Requests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success Rate: {len(successful)/len(results)*100:.1f}%")

    if failed:
        print("\n--- Failed Requests ---")
        for i, (time_taken, success, error) in enumerate(failed[:5]):
            print(f"  {i+1}. {error}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    if successful:
        times = [r[0] for r in successful]
        times_sorted = sorted(times)
        count = len(times_sorted)

        avg = sum(times_sorted) / count
        p50 = statistics.median(times_sorted)
        p90 = times_sorted[int(0.90 * count) - 1] if count >= 10 else times_sorted[-1]
        p95 = times_sorted[int(0.95 * count) - 1] if count >= 20 else times_sorted[-1]
        p99 = times_sorted[int(0.99 * count) - 1] if count >= 100 else times_sorted[-1]

        print("\n--- Latency Report ---")
        print(f"Min Time: {min(times_sorted):.3f}s")
        print(f"Max Time: {max(times_sorted):.3f}s")
        print(f"Avg Time: {avg:.3f}s")
        print(f"P50 (Median): {p50:.3f}s")
        print(f"P90: {p90:.3f}s")
        print(f"P95: {p95:.3f}s")
        print(f"P99: {p99:.3f}s")

        # RTF calculations
        if audio_duration > 0:
            rtfs = [t / audio_duration for t in times_sorted]
            avg_rtf = avg / audio_duration
            rtf_p50 = p50 / audio_duration
            rtf_p90 = p90 / audio_duration
            rtf_p95 = p95 / audio_duration
            rtf_p99 = p99 / audio_duration

            print("\n--- Real-Time Factor (RTF) Report ---")
            print("(RTF < 1.0 means faster than real-time)")
            print(f"Min RTF: {min(rtfs):.3f}")
            print(f"Max RTF: {max(rtfs):.3f}")
            print(f"Avg RTF: {avg_rtf:.3f}")
            print(f"P50 RTF: {rtf_p50:.3f}")
            print(f"P90 RTF: {rtf_p90:.3f}")
            print(f"P95 RTF: {rtf_p95:.3f}")
            print(f"P99 RTF: {rtf_p99:.3f}")

        # Throughput
        total_time = max(times_sorted)
        throughput = len(successful) / total_time
        print("\n--- Throughput ---")
        print(f"Total Wall Time: {total_time:.3f}s")
        print(f"Requests/second: {throughput:.2f}")
        print(f"Audio seconds processed/second: {throughput * audio_duration:.2f}")

    print("\n" + "=" * 50)


async def main():
    # Load audio file once
    print(f"Loading audio file: {AUDIO_FILE}")
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file not found: {AUDIO_FILE}")
        print("Available test audio files:")
        if os.path.exists("test_audio"):
            for f in os.listdir("test_audio"):
                print(f"  - test_audio/{f}")
        return

    with open(AUDIO_FILE, "rb") as f:
        audio_bytes = f.read()

    audio_duration = get_audio_duration(AUDIO_FILE)
    filename = os.path.basename(AUDIO_FILE)

    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"API URL: {STT_API_URL}")
    print(f"Diarization mode: {DIARIZATION_MODE}")

    # Warmup
    print(f"\n--- Warmup ({WARMUP_COUNT} requests) ---")
    for i in range(WARMUP_COUNT):
        results = await run_stress_test(1, audio_bytes, filename)
        status = "✓" if results[0][1] else "✗"
        print(f"  Warmup {i+1}: {results[0][0]:.3f}s {status}")

    # Stress test
    print(f"\n--- Running Stress Test ({CONCURRENCY} concurrent requests) ---")
    start = time.time()
    results = await run_stress_test(CONCURRENCY, audio_bytes, filename)
    wall_time = time.time() - start
    print(f"Completed in {wall_time:.3f}s")

    # Report
    print_report(results, audio_duration, CONCURRENCY)


if __name__ == "__main__":
    asyncio.run(main())
