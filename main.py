import os
import re
import json
import logging
import io
import tempfile
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
import malaya_speech
from malaya_speech.model.clustering import StreamingKMeansMaxCluster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
STT_API_URL = os.environ.get("STT_API_URL", "https://stt-engine-rtx.aies.scicom.dev")
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
MAX_CHUNK_LENGTH = float(os.environ.get("MAX_CHUNK_LENGTH", "25"))
MINIMUM_SILENT_MS = int(os.environ.get("MINIMUM_SILENT_MS", "200"))
MINIMUM_TRIGGER_VAD_MS = int(os.environ.get("MINIMUM_TRIGGER_VAD_MS", "1500"))
REJECT_SEGMENT_VAD_RATIO = float(os.environ.get("REJECT_SEGMENT_VAD_RATIO", "0.9"))

# Audio processing constants
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
# Unified pattern that handles both with and without speaker labels
pattern_unified = r"(?:<\|speaker:(\d+)\|>)?<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>"

# Initialize VAD model
logger.info("Loading silero VAD model...")
silero = load_silero_vad(onnx=True)
logger.info("Silero VAD model loaded")

# Speaker vector for diarization (lazy loaded)
speaker_v = None

app = FastAPI(title="STT API", description="Long audio transcription API with VAD")


def load_speaker_v():
    """Lazy load speaker vector model for diarization"""
    global speaker_v
    if speaker_v is None:
        logger.info("Loading speaker vector model...")
        speaker_v = malaya_speech.speaker_vector.nemo(
            model="huseinzol05/nemo-titanet_large"
        )
        _ = speaker_v.eval()
        logger.info("Speaker vector model loaded")
    return speaker_v


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

    # Normalize audio to prevent clipping
    if len(wav_data) > 0:
        max_val = np.max(np.abs(wav_data))
        if max_val > 0:
            wav_data = wav_data / max_val

    # Create proper WAV file in memory using soundfile
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, wav_data, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    wav_bytes = wav_buffer.read()

    # Use FormData to properly send multipart form data (OpenAI-compatible format)
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

                    # Extract language if available
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

                        # Remove special tokens first
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


def apply_diarization(
    combined_text: str, full_audio: np.ndarray, diarization: StreamingKMeansMaxCluster
) -> str:
    """
    Apply speaker diarization to transcribed text with timestamps.

    This processes ALL segments from the combined transcription and adds speaker
    labels by analyzing the corresponding audio from the full recording.

    Args:
        combined_text: Transcription with timestamp markers
        full_audio: Complete audio array (float32, normalized)
        diarization: Streaming K-Means clusterer for speaker identification

    Returns:
        Text with speaker labels added: <|speaker:N|><|start|>text<|end|>
    """
    load_speaker_v()

    # Extract all segments with timestamps
    matches = re.findall(pattern_unified, combined_text)

    if not matches:
        logger.warning("No segments found for diarization")
        return combined_text

    logger.info(f"Applying diarization to {len(matches)} segments")

    text_parts_with_speakers = []

    for idx, match in enumerate(matches):
        # Pattern: (?:<\|speaker:(\d+)\|>)?<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>
        # Groups: (speaker_id_or_empty, start, text, end)
        # match[0] is existing speaker (ignored - we re-diarize all segments)
        start_time = float(match[1])
        segment_text = match[2]
        end_time = float(match[3])

        # Skip segments that are too short for meaningful diarization
        if len(segment_text.strip()) <= 2:
            logger.debug(f"Segment {idx}: Too short, skipping diarization")
            text_parts_with_speakers.append(
                f"<|{start_time}|>{segment_text}<|{end_time}|>"
            )
            continue

        # Convert timestamps to sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Validate indices are within bounds
        start_sample = max(0, min(start_sample, len(full_audio)))
        end_sample = max(0, min(end_sample, len(full_audio)))

        if end_sample <= start_sample:
            logger.warning(
                f"Segment {idx}: Invalid time range ({start_time}s-{end_time}s)"
            )
            text_parts_with_speakers.append(
                f"<|{start_time}|>{segment_text}<|{end_time}|>"
            )
            continue

        # Extract audio segment
        audio_segment = full_audio[start_sample:end_sample]

        if len(audio_segment) == 0:
            logger.warning(f"Segment {idx}: No audio data")
            text_parts_with_speakers.append(
                f"<|{start_time}|>{segment_text}<|{end_time}|>"
            )
            continue

        try:
            # Normalize segment (full_audio should already be normalized, but ensure consistency)
            max_val = np.max(np.abs(audio_segment))
            if max_val > 0:
                audio_segment = audio_segment / max_val

            # Convert to int16 for speaker vector model
            audio_int16 = np.int16(audio_segment * 32767)

            # Extract speaker embedding
            embedding = speaker_v([audio_int16])[0]

            # Cluster to get speaker ID (streaming clustering maintains state)
            speaker_id = malaya_speech.diarization.streaming(embedding, diarization)

            # Extract numeric ID (handle "speaker N" or just "N" format)
            if isinstance(speaker_id, str):
                num_match = re.search(r"\d+", str(speaker_id))
                speaker_id_num = num_match.group(0) if num_match else "0"
            else:
                speaker_id_num = str(speaker_id)

            logger.debug(
                f"Segment {idx}: speaker={speaker_id_num}, "
                f"time={start_time:.2f}-{end_time:.2f}s, "
                f"text='{segment_text[:50]}...'"
            )

            # Add speaker label
            text_parts_with_speakers.append(
                f"<|speaker:{speaker_id_num}|><|{start_time}|>{segment_text}<|{end_time}|>"
            )

        except Exception as e:
            logger.error(f"Segment {idx}: Diarization failed: {e}")
            # Fallback: keep segment without speaker label
            text_parts_with_speakers.append(
                f"<|{start_time}|>{segment_text}<|{end_time}|>"
            )

    result = "".join(text_parts_with_speakers)
    speaker_count = len(re.findall(r"<\|speaker:\d+\|>", result))
    logger.info(f"Diarization complete: {speaker_count} speaker labels added")

    return result


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
        speaker_id_str = match[0]  # Empty string if no speaker
        start_time = float(match[1])
        segment_text = match[2].strip()
        end_time = float(match[3])

        segment_dict = {
            "id": idx,
            "start": start_time,
            "end": end_time,
            "text": segment_text,
        }

        # Add speaker field only if present
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
    enable_diarization: bool = Form(False),
    speaker_similarity: float = Form(0.5),
    speaker_max_n: int = Form(5),
    minimum_silent_ms: int = Form(MINIMUM_SILENT_MS),
    minimum_trigger_vad_ms: int = Form(MINIMUM_TRIGGER_VAD_MS),
    reject_segment_vad_ratio: float = Form(REJECT_SEGMENT_VAD_RATIO),
):
    """
    Long audio transcription API with VAD chunking and optional speaker diarization.

    Parameters:
    - file: Audio file (multipart/form-data)
    - language: Language hint (e.g., 'en', 'ms', 'zh', 'ta') or None for auto-detect
    - response_format: 'text', 'json', or 'verbose_json' (default: 'json')
    - enable_diarization: Enable speaker diarization (maintains speaker identity across full audio)
    - speaker_similarity: Diarization threshold (0.0-1.0, lower = more speakers)
    - speaker_max_n: Maximum number of speakers (1-100)
    - minimum_silent_ms: Minimum silence duration for VAD trigger (ms)
    - minimum_trigger_vad_ms: Minimum audio length to trigger VAD (ms)
    - reject_segment_vad_ratio: Reject segments with this ratio of silence (0.0-1.0)
    """
    logger.info(f"Request: diarization={enable_diarization}, language={language}")

    # Validate parameters
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

    if enable_diarization:
        if not (0.0 < speaker_similarity < 1.0):
            raise HTTPException(
                status_code=400, detail="speaker_similarity must be between 0.0 and 1.0"
            )
        if not (1 < speaker_max_n < 100):
            raise HTTPException(
                status_code=400, detail="speaker_max_n must be between 1 and 100"
            )

    # Load audio file
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

    # Keep a copy of full audio for diarization (if enabled)
    full_audio = audio_data.copy() if enable_diarization else None

    # Initialize diarization clusterer (maintains state across all chunks)
    diarization = None
    if enable_diarization:
        logger.info(
            f"Initializing diarization: similarity={speaker_similarity}, "
            f"max_speakers={speaker_max_n}"
        )
        diarization = StreamingKMeansMaxCluster(
            threshold=speaker_similarity, max_clusters=speaker_max_n
        )

    # PHASE 1: Chunk audio using VAD
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

            # Run VAD
            frame_pt = torch.from_numpy(frame).unsqueeze(0)
            vad_score = silero(frame_pt, sr=sample_rate).numpy()[0][0]
            vad = vad_score > 0.5

            if vad:
                total_silent = 0
            else:
                total_silent += len(frame)
                total_silent_frames += 1

            # Accumulate audio
            wav_data = np.concatenate([wav_data, frame])
            audio_len = len(wav_data) / sample_rate
            audio_len_ms = audio_len * 1000
            silent_len = (total_silent / sample_rate) * 1000
            silence_ratio = (
                total_silent_frames / total_frames if total_frames > 0 else 0
            )

            # Check if chunk is ready
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

        # Handle remaining samples
        remaining_samples = len(audio_data) % frames_per_chunk
        if remaining_samples > 0:
            remaining_frame = audio_data[-remaining_samples:]
            wav_data = np.concatenate([wav_data, remaining_frame])

        # Process remaining accumulated audio
        if len(wav_data) > 0:
            audio_len = len(wav_data) / sample_rate
            silence_ratio = (
                total_silent_frames / total_frames if total_frames > 0 else 0
            )
            start_ts = last_timestamp
            end_ts = last_timestamp + audio_len
            chunks.append((wav_data.copy(), start_ts, end_ts, silence_ratio))

        logger.info(f"Phase 1 complete: {len(chunks)} chunks created")

        # PHASE 2: Transcribe all chunks
        logger.info("Phase 2: Transcribing chunks...")
        all_transcriptions = []
        detected_language = None

        for chunk_idx, (wav_chunk, start_ts, end_ts, silence_ratio) in enumerate(
            chunks
        ):
            # Skip chunks that are mostly silent
            if silence_ratio <= reject_segment_vad_ratio:
                logger.info(
                    f"Chunk {chunk_idx + 1}/{len(chunks)}: "
                    f"{start_ts:.2f}s-{end_ts:.2f}s "
                    f"(silence: {silence_ratio:.1%})"
                )
                transcription, chunk_lang = await transcribe_chunk(
                    wav_data=wav_chunk,
                    language=language,
                    timestamp_granularities="segment",
                    last_timestamp=start_ts,
                )
                all_transcriptions.append(transcription)

                # Capture language from first chunk
                if detected_language is None and chunk_lang is not None:
                    detected_language = chunk_lang
            else:
                logger.info(
                    f"Chunk {chunk_idx + 1}/{len(chunks)}: Skipped (too silent: {silence_ratio:.1%})"
                )

        logger.info(f"Phase 2 complete: {len(all_transcriptions)} chunks transcribed")

        # Combine transcriptions
        combined_text = "".join(all_transcriptions)

        # PHASE 3: Apply diarization if enabled
        if enable_diarization and full_audio is not None:
            logger.info("Phase 3: Applying diarization to full transcription...")
            combined_text = apply_diarization(combined_text, full_audio, diarization)
            logger.info("Phase 3 complete")

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

    # Parse segments from combined text
    segments = parse_segments(combined_text)

    # Extract plain text from segments (join with spaces)
    final_text = " ".join(seg["text"] for seg in segments) if segments else ""

    # If no segments parsed, fallback to cleaning combined text
    if not final_text:
        final_text = re.sub(pattern, "", combined_text).strip()

    # Return based on response_format
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
