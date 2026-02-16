"""
STT-API Force Alignment Stress Test

Sends concurrent POST requests to /force_align with audio + transcript pairs,
measures latency, throughput, RTF, and success rate.

Usage:
    # Direct testing:
    STT_API_URL=http://localhost:9091 python stress_test_force_alignment.py

    # Docker:
    docker compose -f stress-test-force-alignment.yaml run --rm stress-test-force-alignment

Environment Variables:
    WARMUP_COUNT: Number of warmup requests (default: 2)
    CONCURRENCY: Number of concurrent requests (default: 10)
    STT_API_URL: URL of the STT API (default: http://stt-api:9091)
"""

import os
import asyncio
import time
import aiohttp
import statistics
import librosa

WARMUP_COUNT = int(os.environ.get("WARMUP_COUNT", "2"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "10"))
STT_API_URL = os.environ.get("STT_API_URL", "http://stt-api:9091")

AUDIO_PAIRS = {
    "/app/test_audio/husein-chinese.mp3": ("是的先生，我能帮您什么吗?", "chi"),
    "/app/test_audio/husein-english.mp3": ("Yes sir, what can I help you?", "eng"),
    "/app/test_audio/husein-tamil.mp3": ("ஆமா ஐயா, நான் உங்களுக்கு என்ன உதவி செய்ய வேண்டும்?", "ta"),
    "/app/test_audio/husein-malay.mp3": ("Ya encik, apa yang saya boleh tolong?", "ms"),
}


def get_audio_duration(file_path: str) -> float:
    try:
        return librosa.get_duration(path=file_path)
    except Exception as e:
        print(f"Warning: Could not get audio duration for {file_path}: {e}")
        return 0.0


def load_audio_pairs() -> list:
    """Load all audio files and return list of (audio_bytes, transcript, language, filename, duration)."""
    pairs = []
    for file_path, (transcript, language) in AUDIO_PAIRS.items():
        if not os.path.exists(file_path):
            print(f"Warning: Audio file not found: {file_path}, skipping")
            continue
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        duration = get_audio_duration(file_path)
        filename = os.path.basename(file_path)
        pairs.append((audio_bytes, transcript, language, filename, duration))
    return pairs


async def force_align_request(
    session: aiohttp.ClientSession,
    audio_bytes: bytes,
    transcript: str,
    language: str,
    filename: str,
    request_id: int,
) -> dict:
    url = f"{STT_API_URL}/force_align"
    start_time = time.time()

    try:
        data = aiohttp.FormData()
        data.add_field(
            "file", audio_bytes, filename=filename, content_type="audio/mpeg"
        )
        data.add_field("language", language)
        data.add_field("transcript", transcript)

        async with session.post(url, data=data) as response:
            if response.status == 200:
                result = await response.json()
                processing_time = time.time() - start_time
                num_words = len(result.get("words_alignment", []))
                audio_length = result.get("length", 0)
                return {
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "success": True,
                    "error": None,
                    "filename": filename,
                    "num_words": num_words,
                    "audio_length": audio_length,
                }
            else:
                error_text = await response.text()
                processing_time = time.time() - start_time
                return {
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "success": False,
                    "error": f"HTTP {response.status}: {error_text[:200]}",
                    "filename": filename,
                    "num_words": 0,
                    "audio_length": 0,
                }
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "request_id": request_id,
            "processing_time": processing_time,
            "success": False,
            "error": str(e),
            "filename": filename,
            "num_words": 0,
            "audio_length": 0,
        }


async def run_stress_test(concurrency: int, audio_pairs: list) -> list:
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for i in range(concurrency):
            pair = audio_pairs[i % len(audio_pairs)]
            audio_bytes, transcript, language, filename, duration = pair
            tasks.append(
                force_align_request(
                    session, audio_bytes, transcript, language, filename, i
                )
            )
        return await asyncio.gather(*tasks)


def print_report(results: list, audio_pairs: list, concurrency: int):
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    avg_audio_duration = sum(p[4] for p in audio_pairs) / len(audio_pairs)

    print("\n" + "=" * 60)
    print("FORCE ALIGNMENT STRESS TEST REPORT")
    print("=" * 60)

    print("\n--- Test Configuration ---")
    print(f"Concurrency: {concurrency}")
    print(f"Audio Files: {len(audio_pairs)}")
    for pair in audio_pairs:
        print(f"  {pair[3]}: {pair[4]:.2f}s ({pair[2]})")
    print(f"Avg Audio Duration: {avg_audio_duration:.2f}s")
    print(f"Total Requests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success Rate: {len(successful)/len(results)*100:.1f}%")

    if failed:
        print("\n--- Failed Requests ---")
        for r in failed[:5]:
            print(f"  Request {r['request_id']} ({r['filename']}): {r['error']}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    if successful:
        times = [r["processing_time"] for r in successful]
        times_sorted = sorted(times)
        count = len(times_sorted)

        avg = statistics.mean(times_sorted)
        p50 = statistics.median(times_sorted)
        p90 = times_sorted[int(0.90 * count) - 1] if count >= 10 else times_sorted[-1]
        p95 = times_sorted[int(0.95 * count) - 1] if count >= 20 else times_sorted[-1]
        p99 = times_sorted[int(0.99 * count) - 1] if count >= 100 else times_sorted[-1]

        print("\n--- Latency Report ---")
        print(f"Min: {min(times_sorted):.3f}s")
        print(f"Max: {max(times_sorted):.3f}s")
        print(f"Avg: {avg:.3f}s")
        print(f"P50: {p50:.3f}s")
        print(f"P90: {p90:.3f}s")
        print(f"P95: {p95:.3f}s")
        print(f"P99: {p99:.3f}s")

        if avg_audio_duration > 0:
            rtfs = [t / avg_audio_duration for t in times_sorted]
            print("\n--- Real-Time Factor (RTF) ---")
            print("(RTF < 1.0 means faster than real-time)")
            print(f"Min RTF: {min(rtfs):.3f}")
            print(f"Max RTF: {max(rtfs):.3f}")
            print(f"Avg RTF: {avg / avg_audio_duration:.3f}")
            print(f"P50 RTF: {p50 / avg_audio_duration:.3f}")
            print(f"P90 RTF: {p90 / avg_audio_duration:.3f}")
            print(f"P95 RTF: {p95 / avg_audio_duration:.3f}")
            print(f"P99 RTF: {p99 / avg_audio_duration:.3f}")

        total_words = sum(r["num_words"] for r in successful)
        total_audio = sum(r["audio_length"] for r in successful)
        print("\n--- Alignment Stats ---")
        print(f"Total Words Aligned: {total_words}")
        print(f"Total Audio Aligned: {total_audio:.2f}s")
        print(f"Avg Words/Request: {total_words / len(successful):.1f}")

        wall_time = max(times_sorted)
        print("\n--- Throughput ---")
        print(f"Total Wall Time: {wall_time:.3f}s")
        print(f"Requests/second: {len(successful) / wall_time:.2f}")
        print(f"Audio seconds aligned/second: {total_audio / wall_time:.2f}")

    print("\n" + "=" * 60)


async def main():
    print(f"API URL: {STT_API_URL}")
    print(f"Loading audio files...")

    audio_pairs = load_audio_pairs()
    if not audio_pairs:
        print("Error: No audio files found")
        print("Expected files:")
        for path in AUDIO_PAIRS:
            print(f"  {path}")
        return

    print(f"Loaded {len(audio_pairs)} audio-transcript pairs:")
    for audio_bytes, transcript, language, filename, duration in audio_pairs:
        print(f"  {filename}: {duration:.2f}s [{language}] \"{transcript[:50]}\"")

    # Warmup
    print(f"\n--- Warmup ({WARMUP_COUNT} requests) ---")
    for i in range(WARMUP_COUNT):
        pair = audio_pairs[i % len(audio_pairs)]
        results = await run_stress_test(1, [pair])
        r = results[0]
        status = "ok" if r["success"] else "FAIL"
        words = r["num_words"]
        print(f"  Warmup {i+1} ({r['filename']}): {r['processing_time']:.3f}s, {words} words [{status}]")
        if r["error"]:
            print(f"    Error: {r['error']}")

    # Stress test
    print(f"\n--- Running Stress Test ({CONCURRENCY} concurrent requests) ---")
    start = time.time()
    results = await run_stress_test(CONCURRENCY, audio_pairs)
    wall_time = time.time() - start
    print(f"Completed in {wall_time:.3f}s")

    # Report
    print_report(results, audio_pairs, CONCURRENCY)


if __name__ == "__main__":
    asyncio.run(main())
