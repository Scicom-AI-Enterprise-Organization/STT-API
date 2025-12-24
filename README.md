## stt-api

Long-form speech-to-text API that:

- **Chunks long audio** using VAD into manageable pieces.
- **Keeps global timestamps** across all chunks.
- **Optionally performs speaker diarization** before returning results.
- **Proxies to an upstream STT engine** via an OpenAI-compatible `/v1/audio/transcriptions` endpoint.

The main FastAPI app is in `main.py` and exposes:

- **`GET /`** – simple health/version check.
- **`POST /audio/transcriptions`** – long audio transcription with optional diarization.

---

## Features

- **Long audio support**: VAD-driven chunking so very long recordings can be processed reliably.
- **Language hints**: `language` supports `en`, `ms`, `zh`, `ta`, or `none/null` for auto-detect.
- **Multiple response formats**:
  - `text`: plain text.
  - `json`: `{ "text": "..." }`.
  - `verbose_json`: includes segments with timestamps and language.
- **Optional diarization**:
  - Toggle via `enable_diarization`.
  - Tunable similarity threshold and max speakers.

---

## Architecture (high level)

- **1. Ingest**:
  - Client uploads an audio file to `POST /audio/transcriptions`.
- **2. VAD + chunking**:
  - Audio is streamed through Silero VAD and split into chunks based on silence and a max chunk length.
  - Each chunk is normalised and converted to a proper WAV buffer.
- **3. Upstream STT**:
  - Each chunk is sent to the upstream STT at `STT_API_URL/v1/audio/transcriptions` using multipart form data.
  - The upstream returns timestamps (segments or embedded markers) and optional language info.
- **4. Optional diarization**:
  - When enabled, a speaker embedding model plus clustering assigns speaker labels to segments.
- **5. Merge & respond**:
  - All chunk transcriptions are merged, timestamps are rebased to global time, and the final response is returned in the requested `response_format`.

---

## How to run

From the project root:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Basic usage

```bash
curl -X POST "http://localhost:8000/audio/transcriptions" \
  -F "file=@test_audio/sample.wav" \
  -F "language=en" \
  -F "response_format=json"
```

Key form fields:

- **`file`**: audio file to transcribe.
- **`language`**: one of `none`, `null`, `en`, `ms`, `zh`, `ta` (optional).
- **`response_format`**: `text`, `json`, or `verbose_json` (optional, defaults to `json`).
- **`enable_diarization`**: `true`/`false` to turn speaker diarization on or off (optional).
