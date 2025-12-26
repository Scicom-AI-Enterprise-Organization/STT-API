## stt-api

Long-form speech-to-text API that:

- **Chunks long audio** using VAD into manageable pieces.
- **Keeps global timestamps** across all chunks.
- **Transcribes chunks concurrently** for improved performance.
- **Proxies to an upstream STT engine** via an OpenAI-compatible `/v1/audio/transcriptions` endpoint.

The main FastAPI app is in `app/main.py` and exposes:

- **`GET /`** – simple health/version check.
- **`POST /audio/transcriptions`** – long audio transcription with VAD chunking.

---

## Features

- **Long audio support**: VAD-driven chunking so very long recordings can be processed reliably.
- **Concurrent processing**: Multiple chunks are transcribed in parallel for faster results.
- **Language hints**: `language` supports `en`, `ms`, `zh`, `ta`, or `none/null` for auto-detect.
- **Multiple response formats**:
  - `text`: plain text.
  - `json`: `{ "text": "..." }`.
  - `verbose_json`: includes segments with timestamps and language.
- **Configurable VAD parameters**: Tune silence detection and chunking behavior.

---

## Architecture (high level)

- **1. Ingest**:
  - Client uploads an audio file to `POST /audio/transcriptions`.
- **2. VAD + chunking**:
  - Audio is streamed through Silero VAD and split into chunks based on silence and a max chunk length.
  - Each chunk is normalised and converted to a proper WAV buffer.
- **3. Concurrent transcription**:
  - All chunks are sent concurrently to the upstream STT at `STT_API_URL/v1/audio/transcriptions` using multipart form data.
  - The upstream returns timestamps (segments or embedded markers) and optional language info.
- **4. Merge & respond**:
  - All chunk transcriptions are merged, timestamps are rebased to global time, and the final response is returned in the requested `response_format`.

---

## How to run

### Using Docker Compose (Recommended)

1. **Create a `.env` file** (optional) with environment variables:
   ```bash
   STT_API_URL=https://stt-engine-rtx.aies.scicom.dev
   SAMPLE_RATE=16000
   MAX_CHUNK_LENGTH=25
   MINIMUM_SILENT_MS=200
   MINIMUM_TRIGGER_VAD_MS=1500
   REJECT_SEGMENT_VAD_RATIO=0.9
   ```

2. **Build and start the service**:
   ```bash
   docker-compose up --build
   ```

   The API will be available at `http://localhost:9090`.

3. **Run in detached mode**:
   ```bash
   docker-compose up -d
   ```

4. **View logs**:
   ```bash
   docker-compose logs -f stt-api
   ```

5. **Stop the service**:
   ```bash
   docker-compose down
   ```

### Using uvicorn directly

From the project root:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Note**: Make sure you have all dependencies installed. The project uses `uv` for dependency management:

```bash
uv sync
```

---

## Testing

### Run tests in Docker

1. **Make sure the container is running**:
   ```bash
   docker-compose up -d
   ```

2. **Install dev dependencies** (first time only):
   ```bash
   docker-compose exec stt-api uv sync --extra dev
   ```

3. **Run tests**:
   ```bash
   docker-compose exec stt-api uv run pytest tests/ -v
   ```

### Test the API with curl

**Health check**:
```bash
curl http://localhost:9090/
```

**Transcription**:
```bash
curl -X POST "http://localhost:9090/audio/transcriptions" \
  -F "file=@test_audio/pas.mp3" \
  -F "language=en" \
  -F "response_format=json"
```

---

## Basic usage

```bash
curl -X POST "http://localhost:9090/audio/transcriptions" \
  -F "file=@test_audio/sample.wav" \
  -F "language=en" \
  -F "response_format=json"
```

Key form fields:

- **`file`**: audio file to transcribe (required).
- **`language`**: one of `none`, `null`, `en`, `ms`, `zh`, `ta` (optional).
- **`response_format`**: `text`, `json`, or `verbose_json` (optional, defaults to `json`).
- **`minimum_silent_ms`**: minimum silence duration for VAD trigger in milliseconds (optional, defaults to 200).
- **`minimum_trigger_vad_ms`**: minimum audio length to trigger VAD in milliseconds (optional, defaults to 1500).
- **`reject_segment_vad_ratio`**: reject segments with this ratio of silence (0.0-1.0, optional, defaults to 0.9).

### Example with verbose JSON

```bash
curl -X POST "http://localhost:9090/audio/transcriptions" \
  -F "file=@test_audio/sample.wav" \
  -F "language=ms" \
  -F "response_format=verbose_json"
```

Response includes segments with timestamps:

```json
{
  "language": "ms",
  "duration": 12.26,
  "text": "Pas rela dituduh demikian...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.68,
      "text": "Pas rela dituduh demikian daripada menjadi pelacur politik."
    }
    // ... more segments
  ]
}
```

---

## Environment Variables

- **`STT_API_URL`**: Upstream STT API endpoint (default: `https://stt-engine-rtx.aies.scicom.dev`).
- **`SAMPLE_RATE`**: Audio sample rate in Hz (default: `16000`).
- **`MAX_CHUNK_LENGTH`**: Maximum chunk length in seconds (default: `25`).
- **`MINIMUM_SILENT_MS`**: Minimum silence duration for VAD trigger in milliseconds (default: `200`).
- **`MINIMUM_TRIGGER_VAD_MS`**: Minimum audio length to trigger VAD in milliseconds (default: `1500`).
- **`REJECT_SEGMENT_VAD_RATIO`**: Reject segments with this ratio of silence (default: `0.9`).
