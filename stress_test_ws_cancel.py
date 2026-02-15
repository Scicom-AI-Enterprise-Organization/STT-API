"""
STT-API WebSocket Cancellation Test

Streams audio over WebSocket connections, then disconnects after a configurable
delay to verify the server cancels in-flight transcription tasks.

Watch the server logs for 'Transcription cancelled for client ...' messages.

Usage:
    # Direct testing:
    STT_API_URL=http://localhost:9091 AUDIO_FILE=test_audio/masak.mp3 python stress_test_ws_cancel.py

    # Disconnect after 5 seconds:
    CANCEL_AFTER_S=5.0 python stress_test_ws_cancel.py

    # Disconnect immediately after sending all audio:
    CANCEL_AFTER_S=0.0 python stress_test_ws_cancel.py

Environment Variables:
    CONCURRENCY: Number of concurrent WebSocket clients (default: 5)
    STT_API_URL: URL of the STT API (default: http://stt-api:9091)
    AUDIO_FILE: Path to audio file for testing (default: /app/test_audio/masak.mp3)
    CANCEL_AFTER_S: Seconds to wait after sending all audio before disconnecting (default: 3.0)
    LANGUAGE: Language hint (default: ms)
    CHUNK_MS: Milliseconds of audio per WebSocket send (default: 100)
    SAMPLE_RATE: Audio sample rate in Hz (default: 16000)
    WARMUP_COUNT: Number of warmup clients that run to completion (default: 1)
"""

import os
import asyncio
import time
import json
import numpy as np
import librosa
import websockets

CONCURRENCY = int(os.environ.get("CONCURRENCY", "5"))
STT_API_URL = os.environ.get("STT_API_URL", "http://stt-api:9091")
AUDIO_FILE = os.environ.get("AUDIO_FILE", "/app/test_audio/masak.mp3")
CANCEL_AFTER_S = float(os.environ.get("CANCEL_AFTER_S", "3.0"))
LANGUAGE = os.environ.get("LANGUAGE", "ms")
CHUNK_MS = int(os.environ.get("CHUNK_MS", "100"))
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
WARMUP_COUNT = int(os.environ.get("WARMUP_COUNT", "1"))


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


async def stream_full(audio_data: np.ndarray, client_id: int) -> dict:
    """Stream all audio and wait for all transcriptions to complete."""
    ws_url = build_ws_url()
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
    segments_received = 0
    start_time = time.time()
    first_transcription_time = None
    last_message_time = None
    error_msg = None

    try:
        async with websockets.connect(ws_url) as ws:
            async def receiver():
                nonlocal segments_received, first_transcription_time, last_message_time
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        last_message_time = time.time()
                        data = json.loads(msg)
                        if data.get("type") == "transcription":
                            if first_transcription_time is None:
                                first_transcription_time = time.time()
                            segs = data.get("segments", [])
                            segments_received += len(segs) if segs else 1
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

            # Signal end of audio so server flushes remaining and closes
            await ws.send(json.dumps({"type": "end"}))

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
        "success": error_msg is None and segments_received > 0,
        "error": error_msg,
    }


async def stream_and_cancel(
    audio_data: np.ndarray, client_id: int, cancel_after: float
) -> dict:
    """
    Start streaming audio, then disconnect after cancel_after seconds from the START.
    The server should cancel any in-flight transcriptions.
    """
    ws_url = build_ws_url()
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
    segments_before_cancel = 0
    start_time = time.time()
    error_msg = None
    chunks_sent = 0
    total_chunks = (len(audio_data) + chunk_samples - 1) // chunk_samples

    try:
        async with websockets.connect(ws_url) as ws:
            async def receiver():
                nonlocal segments_before_cancel
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                        data = json.loads(msg)
                        if data.get("type") == "transcription":
                            segments_before_cancel += 1
                except asyncio.TimeoutError:
                    pass
                except websockets.exceptions.ConnectionClosed:
                    pass

            receive_task = asyncio.create_task(receiver())

            async def sender():
                nonlocal chunks_sent
                offset = 0
                while offset < len(audio_data):
                    end = min(offset + chunk_samples, len(audio_data))
                    chunk = audio_data[offset:end]
                    await ws.send(chunk.tobytes())
                    chunks_sent += 1
                    offset = end
                    await asyncio.sleep(0.0)

            send_task = asyncio.create_task(sender())

            # Wait cancel_after seconds then kill everything
            await asyncio.sleep(cancel_after)

            send_task.cancel()
            receive_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

            await ws.close()

    except Exception as e:
        error_msg = str(e)

    elapsed = time.time() - start_time

    print(
        f"  Client {client_id}: sent {chunks_sent}/{total_chunks} chunks in {elapsed:.3f}s, "
        f"{segments_before_cancel} segments received before cancel"
    )

    return {
        "client_id": client_id,
        "elapsed": elapsed,
        "chunks_sent": chunks_sent,
        "total_chunks": total_chunks,
        "segments_before_cancel": segments_before_cancel,
        "status": "cancelled" if error_msg is None else "error",
        "error": error_msg,
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

    audio_data = load_audio(AUDIO_FILE)
    audio_duration = get_audio_duration(AUDIO_FILE)

    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Audio samples: {len(audio_data)}")
    print(f"API URL: {STT_API_URL}")
    print(f"WebSocket URL: {build_ws_url()}")
    print(f"Cancel after: {CANCEL_AFTER_S}s")
    print(f"Concurrency: {CONCURRENCY}")

    # Warmup - full requests
    if WARMUP_COUNT > 0:
        print(f"\n--- Warmup ({WARMUP_COUNT} full clients) ---")
        for i in range(WARMUP_COUNT):
            result = await stream_full(audio_data, i)
            status = "ok" if result["success"] else "FAIL"
            segs = result["segments_received"]
            ttft_str = f", TTFT: {result['ttft']:.3f}s" if result["ttft"] else ""
            print(f"  Warmup {i+1}: {result['total_time']:.3f}s, {segs} segments{ttft_str} [{status}]")
            if result.get("error"):
                print(f"    Error: {result['error']}")

    # Phase 1: Send and cancel
    print(f"\n--- Phase 1: Sending {CONCURRENCY} clients, cancelling after {CANCEL_AFTER_S}s ---")
    phase1_start = time.time()

    tasks = [
        stream_and_cancel(audio_data, i, CANCEL_AFTER_S)
        for i in range(CONCURRENCY)
    ]
    cancel_results = await asyncio.gather(*tasks)

    phase1_time = time.time() - phase1_start
    print(f"\nAll {CONCURRENCY} clients disconnected in {phase1_time:.3f}s")

    # Phase 2: Follow-up to check server health
    print("\n--- Phase 2: Waiting 3s then sending follow-up client ---")
    await asyncio.sleep(3.0)

    result = await stream_full(audio_data, 0)
    status = "ok" if result["success"] else "FAIL"
    segs = result["segments_received"]
    ttft_str = f", TTFT: {result['ttft']:.3f}s" if result["ttft"] else ""
    print(f"  Follow-up: {result['total_time']:.3f}s, {segs} segments{ttft_str} [{status}]")

    # Report
    print("\n" + "=" * 60)
    print("WEBSOCKET CANCELLATION TEST REPORT")
    print("=" * 60)

    print(f"\nAudio Duration: {audio_duration:.2f}s")
    print(f"Cancel After: {CANCEL_AFTER_S}s")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Phase 1 Wall Time: {phase1_time:.3f}s")

    cancelled = [r for r in cancel_results if r["status"] == "cancelled"]
    errors = [r for r in cancel_results if r["status"] == "error"]
    total_segs = sum(r["segments_before_cancel"] for r in cancel_results)

    print(f"\nCancelled: {len(cancelled)}/{CONCURRENCY}")
    print(f"Errors: {len(errors)}/{CONCURRENCY}")
    print(f"Total segments received before cancel: {total_segs}")

    if cancelled:
        elapsed_times = sorted(r["elapsed"] for r in cancelled)
        chunks_sent = [r["chunks_sent"] for r in cancelled]
        total_chunks = cancelled[0]["total_chunks"]
        print(f"\nElapsed Min: {min(elapsed_times):.3f}s")
        print(f"Elapsed Max: {max(elapsed_times):.3f}s")
        print(f"Chunks Sent: {min(chunks_sent)}-{max(chunks_sent)} / {total_chunks}")

    if errors:
        print("\nErrors:")
        for r in errors[:5]:
            print(f"  Client {r['client_id']}: {r.get('error', 'unknown')}")

    print(f"\nFollow-up: {result['total_time']:.3f}s, {segs} segments [{status}]")

    if result["success"]:
        print("Server recovered normally after cancellations")
    else:
        print(f"WARNING: Follow-up failed: {result.get('error', 'unknown')}")

    print("\nCheck server logs for 'Transcription cancelled for client ...' messages")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
