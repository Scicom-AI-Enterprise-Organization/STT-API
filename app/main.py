import os
import re
import json
import logging
import io
import tempfile
import asyncio
import time
import multiprocessing
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import aiohttp
from aiohttp import FormData
import urllib.parse
from fastapi import FastAPI, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import librosa
import soundfile as sf
from app.diarization import load_speaker_model, run_online_diarization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STT_API_URL = os.environ.get("STT_API_URL", "https://stt-engine-rtx.aies.scicom.dev")
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
MAX_CHUNK_LENGTH = float(os.environ.get("MAX_CHUNK_LENGTH", "25"))
MINIMUM_SILENT_MS = int(os.environ.get("MINIMUM_SILENT_MS", "200"))
MINIMUM_TRIGGER_VAD_MS = int(os.environ.get("MINIMUM_TRIGGER_VAD_MS", "1500"))
REJECT_SEGMENT_VAD_RATIO = float(os.environ.get("REJECT_SEGMENT_VAD_RATIO", "0.9"))
ENABLE_ONLINE_DIARIZATION = (
    os.environ.get("ENABLE_ONLINE_DIARIZATION", "true").lower() == "true"
)
OSD_API_URL = os.environ.get("OSD_API_URL", "http://osd:8000")

MAX_CONCURRENT_UPSTREAM = int(os.environ.get("MAX_CONCURRENT_UPSTREAM", "100"))
upstream_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPSTREAM)
logger.info(f"Upstream semaphore initialized with limit: {MAX_CONCURRENT_UPSTREAM}")

MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "50"))
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
logger.info(f"Request semaphore initialized with limit: {MAX_CONCURRENT_REQUESTS}")

CHUNK_BATCH_SIZE = int(os.environ.get("CHUNK_BATCH_SIZE", "64"))

VAD_WORKERS = int(os.environ.get("VAD_WORKERS", "8"))

ENABLE_FORCE_ALIGNMENT = os.environ.get("ENABLE_FORCE_ALIGNMENT", "true").lower() == "true"

_vad_executor = None
_diarization_executor = None

sample_rate = SAMPLE_RATE
maxlen = MAX_CHUNK_LENGTH
frame_size = 512
replaces = [
    "<|startoftranscript|>",
    "<|endoftext|>",
    "<|transcribe|>",
    "<|transcribeprecise|>",
]
pattern = r"<\|\-?\d+\.?\d*\|>"
pattern_unified = r"(?:<\|speaker:(\d+)\|>)?<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>"

_worker_silero = None


def init_vad_worker():
    """Initialize VAD model in worker process."""
    global _worker_silero
    from silero_vad import load_silero_vad

    _worker_silero = load_silero_vad(onnx=True)
    logger.info(f"Worker {os.getpid()} initialized with Silero VAD")


def init_diarization_worker():
    """Initialize diarization model in worker process."""
    from app.diarization import load_speaker_model

    load_speaker_model()
    logger.info(f"Worker {os.getpid()} initialized with speaker model")


def get_vad_executor():
    """Get or create the VAD ProcessPoolExecutor."""
    global _vad_executor
    if _vad_executor is None:
        _vad_executor = ProcessPoolExecutor(
            max_workers=VAD_WORKERS, initializer=init_vad_worker
        )
        logger.info(f"VAD executor initialized with {VAD_WORKERS} workers")
    return _vad_executor


def get_diarization_executor():
    global _diarization_executor
    if _diarization_executor is None:
        # Use spawn context for CUDA compatibility
        spawn_context = multiprocessing.get_context("spawn")
        _diarization_executor = ProcessPoolExecutor(
            max_workers=1, initializer=init_diarization_worker, mp_context=spawn_context
        )
        logger.info("Diarization executor initialized with 1 worker (spawn context)")
    return _diarization_executor


def initialize_models():
    """Initialize all models at startup."""
    get_vad_executor()

    if ENABLE_ONLINE_DIARIZATION:
        try:
            get_diarization_executor()
        except Exception as e:
            logger.warning(
                f"Failed to load speaker model: {e}. Online diarization disabled."
            )


initialize_models()


def process_audio_segment(args: Tuple) -> List[Tuple]:
    """
    Process a segment of audio in a worker process.

    Args:
        args: Tuple of (audio_segment, start_offset, sample_rate, frame_size,
                       maxlen, minimum_silent_ms, minimum_trigger_vad_ms)

    Returns:
        List of (wav_data, start_ts, end_ts, silence_ratio) tuples
    """
    (
        audio_segment,
        start_offset,
        worker_sample_rate,
        worker_frame_size,
        worker_maxlen,
        worker_minimum_silent_ms,
        worker_minimum_trigger_vad_ms,
    ) = args

    global _worker_silero
    if _worker_silero is None:
        from silero_vad import load_silero_vad

        _worker_silero = load_silero_vad(onnx=True)

    _worker_silero.reset_states()

    chunks = []
    wav_data = np.array([], dtype=np.float32)
    last_timestamp = start_offset
    total_silent = 0
    total_silent_frames = 0
    total_frames = 0

    num_frames = len(audio_segment) // worker_frame_size

    for i in range(num_frames):
        start_idx = i * worker_frame_size
        end_idx = start_idx + worker_frame_size
        frame = audio_segment[start_idx:end_idx]

        if len(frame) < worker_frame_size:
            frame = np.pad(frame, (0, worker_frame_size - len(frame)), mode="constant")

        total_frames += 1

        frame_pt = torch.from_numpy(frame).unsqueeze(0)
        vad_score = _worker_silero(frame_pt, sr=worker_sample_rate).numpy()[0][0]
        vad = vad_score > 0.5

        if vad:
            total_silent = 0
        else:
            total_silent += len(frame)
            total_silent_frames += 1

        wav_data = np.concatenate([wav_data, frame])
        audio_len = len(wav_data) / worker_sample_rate
        audio_len_ms = audio_len * 1000
        silent_len = (total_silent / worker_sample_rate) * 1000
        silence_ratio = total_silent_frames / total_frames if total_frames > 0 else 0

        vad_trigger = (
            audio_len_ms >= worker_minimum_trigger_vad_ms
            and silent_len >= worker_minimum_silent_ms
        )

        if vad_trigger or audio_len >= worker_maxlen:
            start_ts = last_timestamp
            end_ts = last_timestamp + audio_len
            chunks.append((wav_data.copy(), start_ts, end_ts, silence_ratio))

            last_timestamp = end_ts
            total_silent = 0
            total_silent_frames = 0
            total_frames = 0
            wav_data = np.array([], dtype=np.float32)

    # Handle remaining samples
    remaining_samples = len(audio_segment) % worker_frame_size
    if remaining_samples > 0:
        remaining_frame = audio_segment[-remaining_samples:]
        wav_data = np.concatenate([wav_data, remaining_frame])

    if len(wav_data) > 0:
        audio_len = len(wav_data) / worker_sample_rate
        silence_ratio = total_silent_frames / total_frames if total_frames > 0 else 0
        start_ts = last_timestamp
        end_ts = last_timestamp + audio_len
        chunks.append((wav_data.copy(), start_ts, end_ts, silence_ratio))

    return chunks


app = FastAPI(title="STT API", description="Long audio transcription API with VAD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def transcribe_chunk(
    wav_data: np.ndarray,
    language: str,
    timestamp_granularities: str,
    last_timestamp: float,
    response_format: str = "verbose_json",
    stream: str = "false",
) -> Tuple[str, Optional[str]]:
    """
    Transcribe a single audio chunk using upstream STT API.

    Args:
        wav_data: Audio data as numpy float32 array
        language: Language hint for transcription
        timestamp_granularities: 'segment' or 'word'
        last_timestamp: Timestamp offset to add to chunk timestamps

    Returns:
        Tuple of (transcribed text with adjusted timestamps, detected language)
    """
    url = urllib.parse.urljoin(STT_API_URL, "/v1/audio/transcriptions")

    timeout = aiohttp.ClientTimeout(total=600, connect=120, sock_read=300)

    if len(wav_data) > 0:
        max_val = np.max(np.abs(wav_data))
        if max_val > 0:
            wav_data = wav_data / max_val

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, wav_data, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    wav_bytes = wav_buffer.read()

    form_data = FormData()
    form_data.add_field(
        "file", wav_bytes, filename="audio.wav", content_type="audio/wav"
    )

    if language and language != "null":
        form_data.add_field("language", language)

    if timestamp_granularities:
        form_data.add_field("timestamp_granularities[]", timestamp_granularities)

    form_data.add_field("response_format", response_format)
    form_data.add_field("stream", stream)

    texts = ""
    detected_language = None

    try:
        async with upstream_semaphore:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=form_data) as r:
                    if r.status == 200:
                        response_data = await r.json()
                        logger.debug(
                            f"Upstream API response keys: {list(response_data.keys())}"
                        )

                        if "language" in response_data:
                            lang_value = response_data["language"]
                            if lang_value is not None:
                                detected_language = lang_value

                        if "segments" in response_data:
                            segments = response_data.get("segments", [])
                            if segments:
                                text_parts = []
                                for seg in segments:
                                    start_ts = seg.get("start", 0.0)
                                    end_ts = seg.get("end", 0.0)
                                    text = seg.get("text", "").strip()
                                    if text:
                                        adjusted_start = round(
                                            start_ts + last_timestamp, 2
                                        )
                                        adjusted_end = round(end_ts + last_timestamp, 2)
                                        text_parts.append(
                                            f"<|{adjusted_start}|>{text}<|{adjusted_end}|>"
                                        )
                                texts = "".join(text_parts)
                            else:
                                logger.warning("Upstream API returned empty segments")
                                texts = ""
                        elif "text" in response_data:
                            texts = response_data["text"]

                            for replace_token in replaces:
                                texts = texts.replace(replace_token, "")

                            matches = re.findall(pattern, texts)
                            for match in matches:
                                timestamp = float(match.split("|")[1])
                                timestamp += last_timestamp
                                timestamp = round(timestamp, 2)
                                timestamp_str = f"<|{timestamp}|>"
                                texts = texts.replace(match, timestamp_str)
                        else:
                            logger.warning(
                                f"Unexpected response format from upstream API: {list(response_data.keys())}"
                            )
                            texts = ""

                        if texts and not re.search(pattern_unified, texts):
                            logger.warning(
                                f"No timestamp pairs found in transcription text. Text length: {len(texts)}"
                            )
                    else:
                        error_text = await r.text()
                        try:
                            error_json = json.loads(error_text)
                            error_detail = error_json.get(
                                "detail",
                                error_json.get("error", {}).get("message", error_text),
                            )
                        except json.JSONDecodeError:
                            error_detail = error_text
                        raise HTTPException(
                            status_code=r.status,
                            detail=f"Upstream STT API error: {error_detail}",
                        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calling upstream STT API: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error transcribing chunk: {str(e)}"
        )

    return texts, detected_language


def parse_segments(text: str) -> List[dict]:
    """
    Parse text with timestamp markers into structured segments.

    Args:
        text: Text with markers like <|speaker:N|><|start|>text<|end|>

    Returns:
        List of segment dictionaries with id, start, end, text, and optionally speaker
    """
    matches = re.findall(pattern_unified, text)
    segments = []

    for idx, match in enumerate(matches):
        speaker_id_str = match[0]
        start_time = float(match[1])
        segment_text = match[2].strip()
        end_time = float(match[3])

        segment_dict = {
            "id": idx,
            "start": start_time,
            "end": end_time,
            "text": segment_text,
        }

        if speaker_id_str:
            try:
                segment_dict["speaker"] = int(speaker_id_str)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Segment {idx}: Invalid speaker_id '{speaker_id_str}': {e}"
                )

        segments.append(segment_dict)

    return segments


async def offline_diarize(
    audio_bytes: bytes, filename: str = "audio.wav"
) -> List[dict]:
    """
    Call external OSD service for precise speaker diarization.

    The OSD service uses pyannote/speaker-diarization-3.1 which is more accurate
    but slower than online diarization.

    Args:
        audio_bytes: Raw audio file bytes
        filename: Original filename for format detection

    Returns:
        List of diarization segments: [{"start": float, "end": float, "speaker": str}, ...]
    """
    url = urllib.parse.urljoin(OSD_API_URL, "/diarize")

    timeout = aiohttp.ClientTimeout(total=600, connect=120, sock_read=300)

    form_data = FormData()
    form_data.add_field(
        "file", audio_bytes, filename=filename, content_type="application/octet-stream"
    )

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=form_data) as r:
                if r.status == 200:
                    result = await r.json()
                    logger.info(f"OSD returned {len(result)} diarization segments")
                    return result
                else:
                    error_text = await r.text()
                    logger.error(f"OSD API error {r.status}: {error_text}")
                    raise HTTPException(
                        status_code=r.status,
                        detail=f"OSD diarization service error: {error_text}",
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to OSD service at {OSD_API_URL}: {e}")
        raise HTTPException(
            status_code=503, detail=f"OSD diarization service unavailable: {str(e)}"
        )


def find_speaker_for_timestamp(
    start: float, end: float, diarization_segments: List[dict]
) -> Optional[int]:
    """
    Find the speaker for a given timestamp range using overlap.

    Args:
        start: Segment start time
        end: Segment end time
        diarization_segments: List of {"start": float, "end": float, "speaker": str}

    Returns:
        Speaker ID (int) or None if no overlap found
    """
    best_overlap = 0
    best_speaker = None

    for dia_seg in diarization_segments:
        dia_start = dia_seg.get("start", 0)
        dia_end = dia_seg.get("end", 0)
        speaker = dia_seg.get("speaker", "SPEAKER_00")

        overlap_start = max(start, dia_start)
        overlap_end = min(end, dia_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            try:
                best_speaker = int(speaker.replace("SPEAKER_", ""))
            except (ValueError, AttributeError):
                best_speaker = 0

    return best_speaker


def find_speaker_for_chunk_timestamp(
    start: float,
    end: float,
    chunk_ranges: List[Tuple[float, float, int]],
    speaker_assignments: Dict[int, int],
) -> Optional[int]:
    """
    Find the speaker for a given timestamp range using overlap with chunk boundaries.
    Used for online diarization mode.

    Args:
        start: Segment start time
        end: Segment end time
        chunk_ranges: List of (chunk_start, chunk_end, chunk_idx) tuples
        speaker_assignments: Dictionary mapping chunk_idx -> speaker_id

    Returns:
        Speaker ID (int) or None if no overlap found
    """
    best_overlap = 0
    best_speaker = None

    for chunk_start, chunk_end, chunk_idx in chunk_ranges:
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker_assignments.get(chunk_idx, 0)

    return best_speaker


def merge_speakers_with_segments(
    segments: List[dict],
    speaker_data: dict,
    mode: str,
    chunks_to_transcribe: List[tuple] = None,
) -> List[dict]:
    """
    Add speaker field to each transcription segment.

    Args:
        segments: Transcription segments [{"id": 0, "start": 0.0, "end": 3.5, "text": "..."}, ...]
        speaker_data: Either:
            - online mode: {chunk_idx: speaker_id} from run_online_diarization()
            - offline mode: [{"start": float, "end": float, "speaker": str}, ...] from OSD
        mode: "online" or "offline"
        chunks_to_transcribe: For online mode, list of (wav_data, start_ts, end_ts)

    Returns:
        Segments with added "speaker" field
    """
    if mode == "online" and chunks_to_transcribe:
        chunk_ranges = []
        for idx, (_, start_ts, end_ts) in enumerate(chunks_to_transcribe):
            chunk_ranges.append((start_ts, end_ts, idx))
        chunk_ranges.sort(key=lambda x: x[0])

        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", seg_start)

            speaker = find_speaker_for_chunk_timestamp(
                seg_start, seg_end, chunk_ranges, speaker_data
            )
            seg["speaker"] = int(speaker) if speaker is not None else 0

    elif mode == "offline":
        for seg in segments:
            speaker = find_speaker_for_timestamp(
                seg.get("start", 0), seg.get("end", 0), speaker_data
            )
            seg["speaker"] = int(speaker) if speaker is not None else 0

    return segments


def vad_parallel(
    audio_data: np.ndarray,
    sample_rate: int,
    frame_size: int,
    maxlen: float,
    minimum_silent_ms: int,
    minimum_trigger_vad_ms: int,
) -> List[Tuple]:
    """
    Process audio with VAD using multiple processes.

    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of audio
        frame_size: Frame size for VAD processing
        maxlen: Maximum chunk length in seconds
        minimum_silent_ms: Minimum silence duration for VAD trigger (ms)
        minimum_trigger_vad_ms: Minimum audio length to trigger VAD (ms)

    Returns:
        List of (wav_data, start_ts, end_ts, silence_ratio) tuples
    """
    audio_duration = len(audio_data) / sample_rate
    segment_duration = audio_duration / VAD_WORKERS
    segment_samples = int(segment_duration * sample_rate)

    segments = []
    for i in range(VAD_WORKERS):
        start_sample = i * segment_samples
        end_sample = (
            start_sample + segment_samples if i < VAD_WORKERS - 1 else len(audio_data)
        )
        start_offset = start_sample / sample_rate
        segments.append(
            (
                audio_data[start_sample:end_sample],
                start_offset,
                sample_rate,
                frame_size,
                maxlen,
                minimum_silent_ms,
                minimum_trigger_vad_ms,
            )
        )

    executor = get_vad_executor()
    results = list(executor.map(process_audio_segment, segments))

    all_chunks = []
    for segment_chunks in results:
        all_chunks.extend(segment_chunks)

    return all_chunks


@app.get("/")
async def read_root():
    return {"message": "STT API", "version": "1.0"}


@app.post("/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    file: bytes = File(),
    language: str = Form(None),
    response_format: str = Form("json"),
    minimum_silent_ms: int = Form(MINIMUM_SILENT_MS),
    minimum_trigger_vad_ms: int = Form(MINIMUM_TRIGGER_VAD_MS),
    reject_segment_vad_ratio: float = Form(REJECT_SEGMENT_VAD_RATIO),
    diarization: str = Form("none"),  # none | online | offline
    speaker_similarity: float = Form(0.5),
    speaker_max_n: int = Form(5),
):
    """
    Long audio transcription API with VAD chunking.

    Parameters:
    - file: Audio file (multipart/form-data)
    - language: Language hint (e.g., 'en', 'ms', 'zh', 'ta') or None for auto-detect
    - response_format: 'text', 'json', or 'verbose_json' (default: 'json')
    - minimum_silent_ms: Minimum silence duration for VAD trigger (ms)
    - minimum_trigger_vad_ms: Minimum audio length to trigger VAD (ms)
    - reject_segment_vad_ratio: Reject segments with this ratio of silence (0.0-1.0)
    - diarization: 'none', 'online', or 'offline'
    - speaker_similarity: Cosine similarity threshold for online diarization
    - speaker_max_n: Maximum speakers for online diarization
    """
    async with request_semaphore:
        return await _process_transcription(
            request=request,
            file=file,
            language=language,
            response_format=response_format,
            minimum_silent_ms=minimum_silent_ms,
            minimum_trigger_vad_ms=minimum_trigger_vad_ms,
            reject_segment_vad_ratio=reject_segment_vad_ratio,
            diarization=diarization,
            speaker_similarity=speaker_similarity,
            speaker_max_n=speaker_max_n,
        )


async def _process_transcription(
    request: Request,
    file: bytes,
    language: str,
    response_format: str,
    minimum_silent_ms: int,
    minimum_trigger_vad_ms: int,
    reject_segment_vad_ratio: float,
    diarization: str = "none",
    speaker_similarity: float = 0.3,
    speaker_max_n: int = 10,
):
    """Internal transcription processing (wrapped by request semaphore)."""
    t_total_start = time.time()
    logger.info(f"Request: language={language}, diarization={diarization}")

    if language is None:
        language = "null"
    else:
        language = language.lower().strip()

    if language not in {"none", "null", "en", "ms", "zh", "ta"}:
        raise HTTPException(
            status_code=400, detail="language only supports: none, null, en, ms, zh, ta"
        )

    response_format = response_format.lower().strip()
    if response_format not in {"text", "json", "verbose_json"}:
        raise HTTPException(
            status_code=400,
            detail="response_format only supports: text, json, verbose_json",
        )

    diarization = diarization.lower().strip()
    if diarization not in {"none", "online", "offline"}:
        raise HTTPException(
            status_code=400, detail="diarization only supports: none, online, offline"
        )

    if diarization == "online":
        try:
            get_diarization_executor()
        except Exception as e:
            raise HTTPException(
                status_code=503, detail=f"Online diarization unavailable: {str(e)}"
            )

    # Phase 1: Load audio
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
            tmp_file.write(file)
            tmp_path = tmp_file.name

        logger.info("Loading audio file...")
        audio_data, _ = librosa.load(tmp_path, sr=sample_rate, mono=True)
        logger.info(f"Audio loaded: {len(audio_data)} samples at {sample_rate}Hz")

        os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error loading audio file: {str(e)}"
        )

    try:
        # Phase 2: VAD chunking
        t_vad_start = time.time()
        logger.info(f"Phase 1: VAD chunking (parallel, {VAD_WORKERS} workers)...")
        chunks = vad_parallel(
            audio_data=audio_data,
            sample_rate=sample_rate,
            frame_size=frame_size,
            maxlen=maxlen,
            minimum_silent_ms=minimum_silent_ms,
            minimum_trigger_vad_ms=minimum_trigger_vad_ms,
        )
        logger.info(
            f"⏱️ Phase 1 complete: {len(chunks)} chunks created in {time.time() - t_vad_start:.2f}s"
        )

        audio_data = None  # Free memory

        # Filter chunks by silence ratio
        chunks_to_transcribe = []
        for chunk_idx, (wav_chunk, start_ts, end_ts, silence_ratio) in enumerate(
            chunks
        ):
            if silence_ratio <= reject_segment_vad_ratio:
                chunks_to_transcribe.append((wav_chunk, start_ts, end_ts))
            else:
                logger.debug(
                    f"Chunk {chunk_idx + 1}/{len(chunks)}: Skipped (too silent: {silence_ratio:.1%})"
                )

        chunks = None  # Free memory
        total_chunks = len(chunks_to_transcribe)
        logger.info(f"Chunks to transcribe: {total_chunks}")

        # Phase 3: Start diarization task in parallel (if enabled)
        diarize_task = None
        audio_chunks_for_diarization = None

        if diarization == "offline":
            logger.info("Starting offline diarization in background...")
            diarize_task = asyncio.create_task(offline_diarize(file))

        elif diarization == "online":
            logger.info("Starting online diarization in background...")
            # Extract audio data for diarization
            audio_chunks_for_diarization = [
                wav_chunk for wav_chunk, _, _ in chunks_to_transcribe
            ]

            loop = asyncio.get_event_loop()
            diarize_task = loop.run_in_executor(
                get_diarization_executor(),
                run_online_diarization,
                audio_chunks_for_diarization,
                speaker_similarity,
                speaker_max_n,
            )

        # Phase 4: Transcription (runs in parallel with diarization)
        t_transcribe_start = time.time()
        logger.info("Phase 2: Transcribing chunks concurrently...")

        all_transcriptions = []
        detected_language = None

        async def _disconnect_monitor(tasks_to_cancel: List[asyncio.Task]):
            """Poll for client disconnect every 0.5s, cancel tasks if disconnected."""
            while True:
                await asyncio.sleep(0.5)
                if await request.is_disconnected():
                    logger.info("Client disconnected, cancelling all pending transcription tasks")
                    for task in tasks_to_cancel:
                        if not task.done():
                            task.cancel()
                    return

        for batch_start in range(0, total_chunks, CHUNK_BATCH_SIZE):
            batch_end = min(batch_start + CHUNK_BATCH_SIZE, total_chunks)
            batch = chunks_to_transcribe[batch_start:batch_end]

            batch_num = batch_start // CHUNK_BATCH_SIZE + 1
            total_batches = (total_chunks + CHUNK_BATCH_SIZE - 1) // CHUNK_BATCH_SIZE

            logger.info(
                f"Processing batch {batch_num}/{total_batches} "
                f"(chunks {batch_start + 1}-{batch_end})"
            )

            t_batch_start = time.time()

            batch_tasks = [
                asyncio.create_task(transcribe_chunk(
                    wav_data=wav_chunk,
                    language=language,
                    timestamp_granularities="segment",
                    last_timestamp=start_ts,
                ))
                for wav_chunk, start_ts, _ in batch
            ]

            monitor_task = asyncio.create_task(_disconnect_monitor(batch_tasks))

            try:
                batch_results = await asyncio.gather(*batch_tasks)
            except asyncio.CancelledError:
                logger.info(f"Batch {batch_num} cancelled due to client disconnect")
                monitor_task.cancel()
                raise
            finally:
                monitor_task.cancel()

            logger.info(
                f"⏱️ Batch {batch_num} STT took: {time.time() - t_batch_start:.2f}s"
            )

            for transcription, chunk_lang in batch_results:
                all_transcriptions.append(transcription)
                if detected_language is None and chunk_lang is not None:
                    detected_language = chunk_lang

        logger.info(
            f"⏱️ Phase 2 complete: {len(all_transcriptions)} chunks transcribed in {time.time() - t_transcribe_start:.2f}s"
        )

        # Phase 5: Wait for diarization to complete
        speaker_assignments = None
        if diarize_task is not None:
            t_diar_wait = time.time()
            try:
                speaker_assignments = await diarize_task
                if diarization == "offline":
                    logger.info(
                        f"⏱️ Offline diarization complete: {len(speaker_assignments)} segments (waited {time.time() - t_diar_wait:.2f}s)"
                    )
                else:
                    logger.info(
                        f"⏱️ Online diarization complete: {len(speaker_assignments)} assignments (waited {time.time() - t_diar_wait:.2f}s)"
                    )
            except Exception as e:
                logger.error(f"Diarization failed: {e}")
                speaker_assignments = None

        combined_text = "".join(all_transcriptions)

    except asyncio.CancelledError:
        if diarize_task is not None and not diarize_task.done():
            diarize_task.cancel()
        logger.info("Request cancelled (client disconnected)")
        return {"text": ""}
    except Exception as e:
        if diarize_task is not None and not diarize_task.done():
            diarize_task.cancel()
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

    # Phase 6: Parse and merge results
    segments = parse_segments(combined_text)

    if speaker_assignments is not None and segments:
        segments = merge_speakers_with_segments(
            segments=segments,
            speaker_data=speaker_assignments,
            mode=diarization,
            chunks_to_transcribe=chunks_to_transcribe
            if diarization == "online"
            else None,
        )

    final_text = " ".join(seg["text"] for seg in segments) if segments else ""

    if not final_text:
        final_text = re.sub(pattern, "", combined_text).strip()

    logger.info(f"⏱️ Total processing time: {time.time() - t_total_start:.2f}s")

    if response_format == "verbose_json":
        duration = segments[-1]["end"] if segments else 0.0
        return {
            "language": detected_language or "unknown",
            "duration": duration,
            "text": final_text,
            "segments": segments,
        }
    elif response_format == "json":
        return {"text": final_text}
    else:
        return final_text

def process_vad_frames(args: Tuple) -> List[Tuple[bool, float]]:
    """
    Process a batch of audio frames through VAD in a worker process.

    Args:
        args: Tuple of (frames_list, worker_sample_rate)
              frames_list is a list of np.float32 arrays, each of frame_size length

    Returns:
        List of (is_speech, vad_score) for each frame
    """
    frames_list, worker_sample_rate = args

    global _worker_silero
    if _worker_silero is None:
        from silero_vad import load_silero_vad
        _worker_silero = load_silero_vad(onnx=True)

    results = []
    for frame in frames_list:
        frame_pt = torch.from_numpy(frame).unsqueeze(0)
        vad_score = _worker_silero(frame_pt, sr=worker_sample_rate).numpy()[0][0]
        results.append((bool(vad_score > 0.5), float(vad_score)))

    return results


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.wav_data: Dict[str, np.ndarray] = {}
        self.wav_queue: Dict[str, list] = {}
        self.total_silent: Dict[str, int] = {}
        self.total_silent_frames: Dict[str, int] = {}
        self.total_frames: Dict[str, int] = {}
        self.last_timestamp: Dict[str, float] = {}
        self.pending_tasks: Dict[str, List[asyncio.Task]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.wav_data[client_id] = np.array([], dtype=np.float32)
        self.wav_queue[client_id] = []
        self.total_silent[client_id] = 0
        self.total_silent_frames[client_id] = 0
        self.total_frames[client_id] = 0
        self.last_timestamp[client_id] = 0.0
        self.pending_tasks[client_id] = []

    def disconnect(self, client_id: str):
        # Cancel all pending transcription tasks for this client
        for task in self.pending_tasks.get(client_id, []):
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled pending task for client {client_id}")
        self.active_connections.pop(client_id, None)
        self.wav_data.pop(client_id, None)
        self.wav_queue.pop(client_id, None)
        self.total_silent.pop(client_id, None)
        self.total_silent_frames.pop(client_id, None)
        self.total_frames.pop(client_id, None)
        self.last_timestamp.pop(client_id, None)
        self.pending_tasks.pop(client_id, None)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)


manager = ConnectionManager()


@app.get("/streaming")
async def get():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_stt(
    websocket: WebSocket,
    language: str = Query("null"),
    minimum_silent_ms: int = Query(MINIMUM_SILENT_MS),
    minimum_trigger_vad_ms: int = Query(MINIMUM_TRIGGER_VAD_MS),
    reject_segment_vad_ratio: float = Query(REJECT_SEGMENT_VAD_RATIO),
):
    client_id = str(id(websocket))
    await manager.connect(websocket, client_id=client_id)
    logger.debug(f"WebSocket client {client_id} connected")

    loop = asyncio.get_event_loop()
    executor = get_vad_executor()
    disconnected = asyncio.Event()

    async def _transcribe_and_send(wav_data, last_ts):
        """Run transcription and send result back. Runs as a background task."""
        try:
            texts, detected_language = await transcribe_chunk(
                wav_data=wav_data,
                language=language,
                timestamp_granularities="segment",
                last_timestamp=last_ts,
                response_format="text",
            )
        except asyncio.CancelledError:
            logger.info(f"Transcription cancelled for client {client_id}")
            return
        except HTTPException as e:
            if not disconnected.is_set():
                try:
                    error_msg = json.dumps({"error": e.detail})
                    await manager.send_personal_message(error_msg, websocket)
                except Exception:
                    pass
            return

        if disconnected.is_set():
            return

        try:
            if texts:
                segments = parse_segments(texts)
                if segments:
                    result = json.dumps({
                        "type": "transcription",
                        "language": detected_language,
                        "segments": segments,
                    })
                else:
                    clean_text = re.sub(pattern, "", texts).strip()
                    result = json.dumps({
                        "type": "transcription",
                        "language": detected_language,
                        "text": clean_text,
                    })
                await manager.send_personal_message(result, websocket)
        except Exception:
            pass

    async def _process_audio_data(audio_bytes: bytes):
        """Process incoming binary audio data through VAD pipeline."""
        try:
            array = np.frombuffer(audio_bytes, dtype=np.float32)
        except Exception:
            error = json.dumps({"error": "input must be float32 audio bytes"})
            await manager.send_personal_message(error, websocket)
            return

        manager.wav_data[client_id] = np.concatenate(
            [manager.wav_data[client_id], array]
        )

        frames = []
        while True:
            buf = manager.wav_data[client_id][:frame_size]
            if len(buf) == frame_size:
                manager.wav_data[client_id] = manager.wav_data[client_id][frame_size:]
                frames.append(buf)
            else:
                break

        if not frames:
            return

        vad_results = await loop.run_in_executor(
            executor,
            process_vad_frames,
            (frames, sample_rate),
        )

        for i, (vad, vad_score) in enumerate(vad_results):
            manager.total_frames[client_id] += 1

            if vad:
                manager.total_silent[client_id] = 0
            else:
                manager.total_silent[client_id] += len(frames[i])
                manager.total_silent_frames[client_id] += 1

            manager.wav_queue[client_id].append(frames[i])
            audio_len = (len(manager.wav_queue[client_id]) * frame_size) / sample_rate
            audio_len_ms = audio_len * 1000
            silent_len = (manager.total_silent[client_id] / sample_rate) * 1000
            negative_ratio = (
                manager.total_silent_frames[client_id] / manager.total_frames[client_id]
                if manager.total_frames[client_id] > 0
                else 0
            )

            vad_trigger = (
                audio_len_ms >= minimum_trigger_vad_ms
                and silent_len >= minimum_silent_ms
            )

            if vad_trigger or audio_len >= maxlen:
                if negative_ratio <= reject_segment_vad_ratio:
                    wav_data = np.concatenate(manager.wav_queue[client_id])
                    last_ts = manager.last_timestamp[client_id]

                    task = asyncio.create_task(
                        _transcribe_and_send(wav_data, last_ts)
                    )
                    manager.pending_tasks[client_id].append(task)
                    # Clean up finished tasks
                    manager.pending_tasks[client_id] = [
                        t for t in manager.pending_tasks[client_id] if not t.done()
                    ]
                else:
                    silent_msg = json.dumps({"type": "silent"})
                    await manager.send_personal_message(silent_msg, websocket)

                manager.last_timestamp[client_id] += audio_len
                manager.total_silent[client_id] = 0
                manager.total_silent_frames[client_id] = 0
                manager.total_frames[client_id] = 0
                manager.wav_queue[client_id] = []

        await asyncio.sleep(0)

    async def _flush_remaining():
        """Process any remaining audio in the queue after client signals done."""
        if client_id not in manager.wav_queue:
            return
        remaining = manager.wav_queue[client_id]
        if remaining:
            wav_data = np.concatenate(remaining)
            if len(wav_data) > 0:
                negative_ratio = (
                    manager.total_silent_frames[client_id] / manager.total_frames[client_id]
                    if manager.total_frames[client_id] > 0
                    else 0
                )
                if negative_ratio <= reject_segment_vad_ratio:
                    last_ts = manager.last_timestamp[client_id]
                    task = asyncio.create_task(
                        _transcribe_and_send(wav_data, last_ts)
                    )
                    manager.pending_tasks[client_id].append(task)
                else:
                    silent_msg = json.dumps({"type": "silent"})
                    await manager.send_personal_message(silent_msg, websocket)

        # Wait for all pending transcription tasks to finish
        pending = [t for t in manager.pending_tasks.get(client_id, []) if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.receive":
                if "bytes" in msg and msg["bytes"]:
                    await _process_audio_data(msg["bytes"])
                elif "text" in msg and msg["text"]:
                    try:
                        text_data = json.loads(msg["text"])
                        if text_data.get("type") == "end":
                            logger.info(f"WebSocket client {client_id} signalled end of audio")
                            await _flush_remaining()
                            await websocket.close()
                            break
                    except json.JSONDecodeError:
                        pass
            elif msg["type"] == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        disconnected.set()
        manager.disconnect(client_id)

if ENABLE_FORCE_ALIGNMENT:
    from app.force_alignment.model import queue_force_align, load_global_alignment_model, step

    load_global_alignment_model()

    @app.post("/force_align")
    async def force_align(
        request: Request,
        file: bytes = File(..., description="Audio 30 seconds chunk (WAV, mp3)"),
        language: str = Form(..., description="Language code (e.g., 'en', 'es')"),
        transcript: str = Form(..., description="Transcript text"),
    ):
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await queue_force_align(fut, file, transcript, language)

        async def _monitor_disconnect():
            while not fut.done():
                await asyncio.sleep(0.5)
                if await request.is_disconnected():
                    logger.info("Force align client disconnected, cancelling request")
                    fut.cancel()
                    return

        monitor = asyncio.create_task(_monitor_disconnect())
        try:
            result = await fut
            return result
        except asyncio.CancelledError:
            logger.info("Force align request cancelled (client disconnected)")
            return {"words_alignment": [], "length": 0}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            monitor.cancel()

    _step_task: asyncio.Task = None

    @app.on_event("startup")
    async def _start_force_alignment_step():
        global _step_task
        _step_task = asyncio.create_task(step())

    @app.on_event("shutdown")
    async def _stop_force_alignment_step():
        if _step_task is not None:
            _step_task.cancel()
            try:
                await _step_task
            except asyncio.CancelledError:
                pass