import os
import re
import json
import logging
import io
import tempfile
import asyncio
from typing import Optional, List, Tuple
import numpy as np
import torch
import aiohttp
from aiohttp import FormData
import urllib.parse
from fastapi import FastAPI, File, Form, HTTPException
import librosa
import soundfile as sf
from silero_vad import load_silero_vad

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STT_API_URL = os.environ.get("STT_API_URL", "https://stt-engine-rtx.aies.scicom.dev")
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
MAX_CHUNK_LENGTH = float(os.environ.get("MAX_CHUNK_LENGTH", "25"))
MINIMUM_SILENT_MS = int(os.environ.get("MINIMUM_SILENT_MS", "200"))
MINIMUM_TRIGGER_VAD_MS = int(os.environ.get("MINIMUM_TRIGGER_VAD_MS", "1500"))
REJECT_SEGMENT_VAD_RATIO = float(os.environ.get("REJECT_SEGMENT_VAD_RATIO", "0.9"))

buffer_size = 4096
sample_rate = SAMPLE_RATE
maxlen = MAX_CHUNK_LENGTH
frame_size = 512  # silero frame size
replaces = [
    "<|startoftranscript|>",
    "<|endoftext|>",
    "<|transcribe|>",
    "<|transcribeprecise|>",
]
pattern = r"<\|\-?\d+\.?\d*\|>"
# Handles both with and without speaker labels
pattern_unified = r"(?:<\|speaker:(\d+)\|>)?<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>"

logger.info("Loading silero VAD model...")
silero = load_silero_vad(onnx=True)
logger.info("Silero VAD model loaded")

app = FastAPI(title="STT API", description="Long audio transcription API with VAD")


async def transcribe_chunk(
    wav_data: np.ndarray,
    language: str,
    timestamp_granularities: str,
    last_timestamp: float,
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

    # Normalize to prevent clipping
    if len(wav_data) > 0:
        max_val = np.max(np.abs(wav_data))
        if max_val > 0:
            wav_data = wav_data / max_val

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, wav_data, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    wav_bytes = wav_buffer.read()

    # OpenAI-compatible multipart form data
    form_data = FormData()
    form_data.add_field(
        "file", wav_bytes, filename="audio.wav", content_type="audio/wav"
    )

    if language and language != "null":
        form_data.add_field("language", language)

    if timestamp_granularities:
        form_data.add_field("timestamp_granularities[]", timestamp_granularities)

    form_data.add_field("response_format", "verbose_json")
    form_data.add_field("stream", "false")

    texts = ""
    detected_language = None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form_data) as r:
                if r.status == 200:
                    response_data = await r.json()
                    logger.debug(
                        f"Upstream API response keys: {list(response_data.keys())}"
                    )

                    if "language" in response_data:
                        detected_language = response_data["language"]

                    # Handle both segments format and text with embedded timestamps
                    if "segments" in response_data:
                        segments = response_data.get("segments", [])
                        if segments:
                            text_parts = []
                            for seg in segments:
                                start_ts = seg.get("start", 0.0)
                                end_ts = seg.get("end", 0.0)
                                text = seg.get("text", "").strip()
                                if text:
                                    adjusted_start = round(start_ts + last_timestamp, 2)
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

                        # Remove special tokens before timestamp adjustment
                        for replace_token in replaces:
                            texts = texts.replace(replace_token, "")

                        # Adjust timestamps by adding last_timestamp offset
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
                        error_detail = json.loads(error_text).get("detail", error_text)
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


@app.get("/")
async def read_root():
    return {"message": "STT API", "version": "1.0"}


@app.post("/audio/transcriptions")
async def audio_transcriptions(
    file: bytes = File(),
    language: str = Form(None),
    response_format: str = Form("json"),
    minimum_silent_ms: int = Form(MINIMUM_SILENT_MS),
    minimum_trigger_vad_ms: int = Form(MINIMUM_TRIGGER_VAD_MS),
    reject_segment_vad_ratio: float = Form(REJECT_SEGMENT_VAD_RATIO),
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
    """
    logger.info(f"Request: language={language}")

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

    # Phase 1: VAD chunking
    chunks = []  # (wav_data, start_timestamp, end_timestamp, silence_ratio)
    wav_data = np.array([], dtype=np.float32)
    last_timestamp = 0.0
    total_silent = 0
    total_silent_frames = 0
    total_frames = 0
    frames_per_chunk = frame_size

    try:
        logger.info("Phase 1: VAD chunking...")
        num_frames = len(audio_data) // frames_per_chunk

        for i in range(num_frames):
            start_idx = i * frames_per_chunk
            end_idx = start_idx + frames_per_chunk
            frame = audio_data[start_idx:end_idx]

            if len(frame) < frames_per_chunk:
                frame = np.pad(
                    frame, (0, frames_per_chunk - len(frame)), mode="constant"
                )

            total_frames += 1

            frame_pt = torch.from_numpy(frame).unsqueeze(0)
            vad_score = silero(frame_pt, sr=sample_rate).numpy()[0][0]
            vad = vad_score > 0.5

            if vad:
                total_silent = 0
            else:
                total_silent += len(frame)
                total_silent_frames += 1

            wav_data = np.concatenate([wav_data, frame])
            audio_len = len(wav_data) / sample_rate
            audio_len_ms = audio_len * 1000
            silent_len = (total_silent / sample_rate) * 1000
            silence_ratio = (
                total_silent_frames / total_frames if total_frames > 0 else 0
            )

            vad_trigger = (
                audio_len_ms >= minimum_trigger_vad_ms
                and silent_len >= minimum_silent_ms
            )

            if vad_trigger or audio_len >= maxlen:
                start_ts = last_timestamp
                end_ts = last_timestamp + audio_len
                chunks.append((wav_data.copy(), start_ts, end_ts, silence_ratio))

                last_timestamp = end_ts
                total_silent = 0
                total_silent_frames = 0
                total_frames = 0
                wav_data = np.array([], dtype=np.float32)

        remaining_samples = len(audio_data) % frames_per_chunk
        if remaining_samples > 0:
            remaining_frame = audio_data[-remaining_samples:]
            wav_data = np.concatenate([wav_data, remaining_frame])

        if len(wav_data) > 0:
            audio_len = len(wav_data) / sample_rate
            silence_ratio = (
                total_silent_frames / total_frames if total_frames > 0 else 0
            )
            start_ts = last_timestamp
            end_ts = last_timestamp + audio_len
            chunks.append((wav_data.copy(), start_ts, end_ts, silence_ratio))

        logger.info(f"Phase 1 complete: {len(chunks)} chunks created")

        # Phase 2: Concurrent transcription
        logger.info("Phase 2: Transcribing chunks concurrently...")

        transcription_tasks = []

        for chunk_idx, (wav_chunk, start_ts, end_ts, silence_ratio) in enumerate(
            chunks
        ):
            if silence_ratio <= reject_segment_vad_ratio:
                logger.info(
                    f"Chunk {chunk_idx + 1}/{len(chunks)}: "
                    f"{start_ts:.2f}s-{end_ts:.2f}s "
                    f"(silence: {silence_ratio:.1%})"
                )
                transcription_tasks.append(
                    transcribe_chunk(
                        wav_data=wav_chunk,
                        language=language,
                        timestamp_granularities="segment",
                        last_timestamp=start_ts,
                    )
                )
            else:
                logger.info(
                    f"Chunk {chunk_idx + 1}/{len(chunks)}: Skipped (too silent: {silence_ratio:.1%})"
                )

        results = await asyncio.gather(*transcription_tasks)

        all_transcriptions = []
        detected_language = None

        for transcription, chunk_lang in results:
            all_transcriptions.append(transcription)
            if detected_language is None and chunk_lang is not None:
                detected_language = chunk_lang

        logger.info(f"Phase 2 complete: {len(all_transcriptions)} chunks transcribed")

        combined_text = "".join(all_transcriptions)

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

    segments = parse_segments(combined_text)

    final_text = " ".join(seg["text"] for seg in segments) if segments else ""

    if not final_text:
        final_text = re.sub(pattern, "", combined_text).strip()

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
    else:  # 'text'
        return final_text
