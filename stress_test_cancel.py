"""
STT-API Cancellation Test

Sends concurrent POST requests to /audio/transcriptions, then aborts them
after a configurable delay to verify the server stops processing.

Watch the server logs to confirm tasks are cancelled and not running to completion.

Usage:
    # Direct testing:
    STT_API_URL=http://localhost:9091 AUDIO_FILE=test_audio/masak.mp3 python stress_test_cancel.py

    # Cancel after 2 seconds:
    CANCEL_AFTER_S=2.0 python stress_test_cancel.py

    # Cancel after 0 seconds (immediate):
    CANCEL_AFTER_S=0.0 python stress_test_cancel.py

Environment Variables:
    CONCURRENCY: Number of concurrent requests to send (default: 5)
    STT_API_URL: URL of the STT API (default: http://stt-api:9091)
    AUDIO_FILE: Path to audio file for testing (default: /app/test_audio/masak.mp3)
    CANCEL_AFTER_S: Seconds to wait before cancelling each request (default: 3.0)
    LANGUAGE: Language hint (default: ms)
    WARMUP_COUNT: Number of warmup requests that run to completion (default: 1)
"""

import os
import asyncio
import time
import aiohttp
import librosa

CONCURRENCY = int(os.environ.get("CONCURRENCY", "5"))
STT_API_URL = os.environ.get("STT_API_URL", "http://stt-api:9091")
AUDIO_FILE = os.environ.get("AUDIO_FILE", "/app/test_audio/masak.mp3")
CANCEL_AFTER_S = float(os.environ.get("CANCEL_AFTER_S", "3.0"))
LANGUAGE = os.environ.get("LANGUAGE", "ms")
WARMUP_COUNT = int(os.environ.get("WARMUP_COUNT", "1"))


def get_audio_duration(file_path: str) -> float:
    try:
        return librosa.get_duration(path=file_path)
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
        return 0.0


async def _do_request(session: aiohttp.ClientSession, audio_bytes: bytes, filename: str):
    """The actual POST request. Meant to be wrapped in a task and cancelled."""
    url = f"{STT_API_URL}/audio/transcriptions"
    data = aiohttp.FormData()
    data.add_field("file", audio_bytes, filename=filename, content_type="audio/mpeg")
    data.add_field("language", LANGUAGE)
    data.add_field("response_format", "verbose_json")

    async with session.post(url, data=data) as response:
        return await response.json()


async def send_and_cancel(
    session: aiohttp.ClientSession,
    audio_bytes: bytes,
    filename: str,
    cancel_after: float,
    request_id: int,
) -> dict:
    """
    Send a transcription request and cancel the task after cancel_after seconds.
    Cancelling the task closes the underlying TCP connection,
    which the server detects as client disconnect.
    """
    start_time = time.time()

    task = asyncio.create_task(_do_request(session, audio_bytes, filename))

    await asyncio.sleep(cancel_after)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        elapsed = time.time() - start_time
        print(f"  Request {request_id}: cancelled after {elapsed:.3f}s")
        return {
            "request_id": request_id,
            "elapsed": elapsed,
            "status": "cancelled",
            "cancel_after": cancel_after,
        }

    # Task completed before we could cancel it
    elapsed = time.time() - start_time
    print(f"  Request {request_id}: completed before cancel at {elapsed:.3f}s")
    return {
        "request_id": request_id,
        "elapsed": elapsed,
        "status": "completed_before_cancel",
        "cancel_after": cancel_after,
    }


async def send_full_request(
    session: aiohttp.ClientSession,
    audio_bytes: bytes,
    filename: str,
    request_id: int,
) -> dict:
    """Send a request and wait for full completion."""
    url = f"{STT_API_URL}/audio/transcriptions"
    start_time = time.time()

    data = aiohttp.FormData()
    data.add_field("file", audio_bytes, filename=filename, content_type="audio/mpeg")
    data.add_field("language", LANGUAGE)
    data.add_field("response_format", "verbose_json")

    try:
        async with session.post(url, data=data) as response:
            if response.status == 200:
                result = await response.json()
                elapsed = time.time() - start_time
                text_preview = result.get("text", "")[:80]
                return {
                    "request_id": request_id,
                    "elapsed": elapsed,
                    "status": "completed",
                    "text_preview": text_preview,
                }
            else:
                error_text = await response.text()
                elapsed = time.time() - start_time
                return {
                    "request_id": request_id,
                    "elapsed": elapsed,
                    "status": "http_error",
                    "error": f"HTTP {response.status}: {error_text[:100]}",
                }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "request_id": request_id,
            "elapsed": elapsed,
            "status": "error",
            "error": str(e),
        }


async def main():
    print(f"Loading audio file: {AUDIO_FILE}")
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file not found: {AUDIO_FILE}")
        if os.path.exists("test_audio"):
            print("Available test audio files:")
            for f in os.listdir("test_audio"):
                print(f"  - test_audio/{f}")
        return

    with open(AUDIO_FILE, "rb") as f:
        audio_bytes = f.read()

    audio_duration = get_audio_duration(AUDIO_FILE)
    filename = os.path.basename(AUDIO_FILE)

    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"API URL: {STT_API_URL}")
    print(f"Cancel after: {CANCEL_AFTER_S}s")
    print(f"Concurrency: {CONCURRENCY}")

    timeout = aiohttp.ClientTimeout(total=600)

    # Warmup - full requests
    if WARMUP_COUNT > 0:
        print(f"\n--- Warmup ({WARMUP_COUNT} full requests) ---")
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i in range(WARMUP_COUNT):
                result = await send_full_request(session, audio_bytes, filename, i)
                print(f"  Warmup {i+1}: {result['elapsed']:.3f}s [{result['status']}]")
                if result.get("text_preview"):
                    print(f"    Text: {result['text_preview']}...")

    # Phase 1: Send requests and cancel them
    print(f"\n--- Phase 1: Sending {CONCURRENCY} requests, cancelling after {CANCEL_AFTER_S}s ---")
    phase1_start = time.time()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            send_and_cancel(session, audio_bytes, filename, CANCEL_AFTER_S, i)
            for i in range(CONCURRENCY)
        ]
        cancel_results = await asyncio.gather(*tasks)

    phase1_time = time.time() - phase1_start
    print(f"\nAll {CONCURRENCY} requests cancelled in {phase1_time:.3f}s")

    # Phase 2: Wait a bit, then send a follow-up request to check server health
    print("\n--- Phase 2: Waiting 3s then sending follow-up request ---")
    await asyncio.sleep(3.0)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        result = await send_full_request(session, audio_bytes, filename, 0)
        print(f"  Follow-up: {result['elapsed']:.3f}s [{result['status']}]")
        if result.get("text_preview"):
            print(f"    Text: {result['text_preview']}...")

    # Report
    print("\n" + "=" * 60)
    print("CANCELLATION TEST REPORT")
    print("=" * 60)

    print(f"\nAudio Duration: {audio_duration:.2f}s")
    print(f"Cancel After: {CANCEL_AFTER_S}s")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Phase 1 Wall Time: {phase1_time:.3f}s")

    cancelled = [r for r in cancel_results if r["status"] == "cancelled"]
    errors = [r for r in cancel_results if r["status"] == "error"]
    print(f"\nCancelled: {len(cancelled)}/{CONCURRENCY}")
    print(f"Errors: {len(errors)}/{CONCURRENCY}")

    if cancelled:
        times = sorted(r["elapsed"] for r in cancelled)
        print(f"Cancel Time Min: {min(times):.3f}s")
        print(f"Cancel Time Max: {max(times):.3f}s")
        print(f"Cancel Time Avg: {sum(times)/len(times):.3f}s")

    if errors:
        print("\nErrors:")
        for r in errors[:5]:
            print(f"  Request {r['request_id']}: {r.get('error', 'unknown')}")

    print(f"\nFollow-up Request: {result['elapsed']:.3f}s [{result['status']}]")

    if result["status"] == "completed":
        expected_normal = audio_duration * 0.5  # rough estimate
        if result["elapsed"] < expected_normal * 3:
            print("Server recovered normally after cancellations")
        else:
            print(f"WARNING: Follow-up took {result['elapsed']:.1f}s, server may still be processing cancelled requests")
    else:
        print(f"WARNING: Follow-up failed: {result.get('error', 'unknown')}")

    print("\nCheck server logs for 'Client disconnected, aborting transcription' messages")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
