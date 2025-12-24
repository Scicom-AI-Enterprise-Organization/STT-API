import os
import re
import json
import logging
import io
import tempfile
from typing import Optional
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
pattern_pair = r"<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>"

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
    diarization: Optional[StreamingKMeansMaxCluster] = None,
) -> tuple[str, Optional[str]]:
    """
    Transcribe a single audio chunk using upstream STT API.

    Args:
        wav_data: Audio data as numpy float32 array
        language: Language hint for transcription
        timestamp_granularities: 'segment' or 'word'
        last_timestamp: Timestamp offset to add to chunk timestamps
        diarization: Optional diarization clusterer

    Returns:
        Tuple of (transcribed text with adjusted timestamps, detected language)
    """
    if diarization is not None:
        load_speaker_v()

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

    # OpenAI uses timestamp_granularities as an array
    if timestamp_granularities:
        form_data.add_field("timestamp_granularities[]", timestamp_granularities)

    # Request verbose_json format to get timestamps in response
    form_data.add_field("response_format", "verbose_json")

    # Request non-streaming response for synchronous processing
    form_data.add_field("stream", "false")

    texts = ""
    detected_language = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form_data) as r:
                if r.status == 200:
                    # Handle non-streaming JSON response only
                    response_data = await r.json()
                    logger.debug(
                        f"Upstream API response keys: {list(response_data.keys())}"
                    )

                    # Extract language if available
                    if "language" in response_data:
                        detected_language = response_data["language"]

                    # Extract text and timestamps from response
                    # Handle both segments format and text with embedded timestamps
                    if "segments" in response_data:
                        # verbose_json format with segments - reconstruct text with timestamp markers
                        segments = response_data.get("segments", [])
                        if segments:
                            # Build text with timestamp markers: <|start|>text<|end|>
                            text_parts = []
                            for seg in segments:
                                start_ts = seg.get("start", 0.0)
                                end_ts = seg.get("end", 0.0)
                                text = seg.get("text", "").strip()
                                if text:
                                    # Add timestamp offset to relative timestamps
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
                        # Text format - may have embedded timestamps
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

                    # Log if no timestamps found
                    if texts and not re.search(pattern_pair, texts):
                        logger.warning(
                            f"No timestamp pairs found in transcription text. Text length: {len(texts)}"
                        )

                    # Handle diarization if enabled
                    if diarization is not None and len(texts) > 0:
                        matches = re.findall(pattern_pair, texts)
                        if len(matches):
                            match = matches[0]
                            if len(match[1]) > 2:
                                # Convert absolute timestamps back to relative for audio extraction
                                start_abs = float(match[0])
                                end_abs = float(match[-1])
                                start_rel = int(
                                    (start_abs - last_timestamp) * sample_rate
                                )
                                end_rel = int((end_abs - last_timestamp) * sample_rate)
                                # Convert float32 to int16 for diarization
                                sample_wav_float = wav_data[start_rel:end_rel]
                                if len(sample_wav_float) > 0:
                                    max_val = np.max(np.abs(sample_wav_float))
                                    if max_val > 0:
                                        sample_wav_float = sample_wav_float / max_val
                                    sample_wav = np.int16(sample_wav_float * 32768)
                                    v = speaker_v([sample_wav])[0]
                                    speaker = malaya_speech.diarization.streaming(
                                        v, diarization
                                    )
                                    speaker = f"{speaker}|>"
                                    splitted = texts.split("<|")
                                    texts = "<|".join(
                                        splitted[:1] + [speaker] + splitted[1:]
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
    Long audio transcription API with VAD chunking.

    Accepts audio file uploads, chunks them using VAD (silero-vad) with 25-second max chunks,
    maintains timestamps across chunks, and optionally supports speaker diarization.

    Parameters:
    - file: Audio file (multipart/form-data)
    - language: Language hint (e.g., 'en', 'ms', 'zh', 'ta') or None for auto-detect
    - response_format: Response format - 'text', 'json', or 'verbose_json' (default: 'json')
    - enable_diarization: Enable speaker diarization
    - speaker_similarity: Diarization threshold (0.0-1.0)
    - speaker_max_n: Maximum number of speakers (1-100)
    - minimum_silent_ms: Minimum silence duration for VAD trigger (ms)
    - minimum_trigger_vad_ms: Minimum audio length to trigger VAD (ms)
    - reject_segment_vad_ratio: Reject segments with this ratio of silence (0.0-1.0)

    Returns:
    - 'text': Plain text string
    - 'json': JSON with 'text' field containing full transcription
    - 'verbose_json': JSON with 'text', 'segments' (with timestamps), 'language', 'duration'
    """
    if language is None:
        language = "null"
    else:
        language = language.lower().strip()

    # Validate language
    if language not in {"none", "null", "en", "ms", "zh", "ta"}:
        raise HTTPException(
            status_code=400, detail="language only supports: none, null, en, ms, zh, ta"
        )

    # Validate response_format
    response_format = response_format.lower().strip()
    if response_format not in {"text", "json", "verbose_json"}:
        raise HTTPException(
            status_code=400,
            detail="response_format only supports: text, json, verbose_json",
        )

    # Validate diarization parameters
    if enable_diarization:
        if not (0.0 < speaker_similarity < 1.0):
            raise HTTPException(
                status_code=400, detail="speaker_similarity must be between 0.0 and 1.0"
            )
        if not (1 < speaker_max_n < 100):
            raise HTTPException(
                status_code=400, detail="speaker_max_n must be between 1 and 100"
            )

    # Load audio file using librosa (supports various formats)
    # Save bytes to temporary file for librosa to read
    frames_per_chunk = frame_size  # silero frame size

    try:
        # Create a temporary file to store the audio bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
            tmp_file.write(file)
            tmp_path = tmp_file.name

        # Load audio with librosa (automatically resamples to target sample_rate)
        logger.info("Loading audio file...")
        audio_data, original_sr = librosa.load(tmp_path, sr=sample_rate, mono=True)
        logger.info(f"Audio loaded: {len(audio_data)} samples at {sample_rate}Hz")

        # Clean up temporary file
        os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error loading audio file: {str(e)}"
        )

    # Initialize diarization if enabled
    diarization = None
    if enable_diarization:
        diarization = StreamingKMeansMaxCluster(
            threshold=speaker_similarity, max_clusters=speaker_max_n
        )

    # Phase 1: Chunk entire audio first - collect all chunks with their timestamps
    # This allows us to know all chunk boundaries before sending to API
    chunks = []  # List of (wav_data, start_timestamp, end_timestamp, negative_ratio)
    wav_data = np.array([], dtype=np.float32)
    last_timestamp = 0.0
    total_silent = 0
    total_silent_frames = 0
    total_frames = 0

    try:
        logger.info("Phase 1: Chunking audio with VAD...")
        # Process audio in chunks of frame_size
        num_frames = len(audio_data) // frames_per_chunk

        for i in range(num_frames):
            start_idx = i * frames_per_chunk
            end_idx = start_idx + frames_per_chunk
            frame = audio_data[start_idx:end_idx]

            # Ensure frame is exactly frames_per_chunk (pad if necessary)
            if len(frame) < frames_per_chunk:
                frame = np.pad(
                    frame, (0, frames_per_chunk - len(frame)), mode="constant"
                )

            total_frames += 1

            # Convert to PyTorch tensor for VAD
            frame_pt = torch.from_numpy(frame).unsqueeze(0)

            # Run silero VAD
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
            negative_ratio = (
                total_silent_frames / total_frames if total_frames > 0 else 0
            )

            # Check if chunk is ready
            vad_trigger = (
                audio_len_ms >= minimum_trigger_vad_ms
                and silent_len >= minimum_silent_ms
            )

            if vad_trigger or audio_len >= maxlen:
                # Store chunk with its metadata
                start_timestamp = last_timestamp
                end_timestamp = last_timestamp + audio_len
                chunks.append(
                    (wav_data.copy(), start_timestamp, end_timestamp, negative_ratio)
                )

                # Update timestamp and reset accumulators
                last_timestamp = end_timestamp
                total_silent = 0
                total_silent_frames = 0
                total_frames = 0
                wav_data = np.array([], dtype=np.float32)

        # Process remaining audio samples that don't make a complete frame
        remaining_samples = len(audio_data) % frames_per_chunk
        if remaining_samples > 0:
            remaining_frame = audio_data[-remaining_samples:]
            # Don't pad - just add the remaining samples to wav_data
            wav_data = np.concatenate([wav_data, remaining_frame])

        # Process remaining accumulated audio if any
        if len(wav_data) > 0:
            audio_len = len(wav_data) / sample_rate
            negative_ratio = (
                total_silent_frames / total_frames if total_frames > 0 else 0
            )
            start_timestamp = last_timestamp
            end_timestamp = last_timestamp + audio_len
            chunks.append(
                (wav_data.copy(), start_timestamp, end_timestamp, negative_ratio)
            )

        logger.info(f"Phase 1 complete: Found {len(chunks)} chunks")

        # Phase 2: Send all chunks to upstream API
        logger.info("Phase 2: Sending chunks to upstream API...")
        all_transcriptions = []
        detected_language = None

        for chunk_idx, (
            wav_data,
            start_timestamp,
            end_timestamp,
            negative_ratio,
        ) in enumerate(chunks):
            # Only transcribe if not mostly silent
            if negative_ratio <= reject_segment_vad_ratio:
                logger.info(
                    f"Transcribing chunk {chunk_idx + 1}/{len(chunks)} (start: {start_timestamp:.2f}s, end: {end_timestamp:.2f}s)"
                )
                transcription, chunk_language = await transcribe_chunk(
                    wav_data=wav_data,
                    language=language,
                    timestamp_granularities="segment",
                    last_timestamp=start_timestamp,  # Use chunk's start timestamp as offset
                    diarization=diarization,
                )
                all_transcriptions.append(transcription)
                # Capture language from first chunk
                if detected_language is None and chunk_language is not None:
                    detected_language = chunk_language
            else:
                logger.info(
                    f"Skipping chunk {chunk_idx + 1}/{len(chunks)} (too silent: {negative_ratio:.2%})"
                )

        logger.info(f"Phase 2 complete: Transcribed {len(all_transcriptions)} chunks")

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

    # Combine all transcriptions
    combined_text = "".join(all_transcriptions)

    # Parse timestamp pairs to extract segments
    matches = re.findall(pattern_pair, combined_text)
    segments = []
    all_texts = []

    for no, (start, text, end) in enumerate(matches):
        start_timestamp = float(start)
        end_timestamp = float(end)
        segments.append(
            {
                "id": no,
                "start": start_timestamp,
                "end": end_timestamp,
                "text": text.strip(),
            }
        )
        all_texts.append(text.strip())

    # If no timestamp pairs found, use the raw text (might not have timestamps)
    if not segments:
        # Remove timestamp markers but keep text
        cleaned_text = re.sub(pattern, "", combined_text)
        cleaned_text = cleaned_text.strip()
        if cleaned_text:
            all_texts = [cleaned_text]

    # Join without spaces to match reference behavior
    final_text = "".join(all_texts) if all_texts else combined_text.strip()

    # Return based on response_format
    if response_format == "verbose_json":
        duration = segments[-1]["end"] if segments else 0.0
        return {
            "language": detected_language or "unknown",
            "duration": duration,
            "text": final_text,
            "segments": segments if segments else [],
        }
    elif response_format == "json":
        return {"text": final_text}
    else:  # response_format == 'text'
        return final_text
