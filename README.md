# STT-API

Long-form speech-to-text API that:

- **Chunks long audio** using VAD (Silero or FireRed) into manageable pieces
- **Keeps global timestamps** across all chunks
- **Transcribes chunks concurrently** for improved performance
- **Proxies to an upstream STT engine** via an OpenAI-compatible `/v1/audio/transcriptions` endpoint
- **Real-time WebSocket streaming** with per-client VAD and live transcription
- **Force alignment** for word-level timestamps using CTC alignment (MMS-300M) with dynamic batching
- **Speaker diarization** with online clustering (TitaNet + StreamingKMeans or BIRCH) or offline (pyannote) modes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Client Request                              │
│                         (audio file upload)                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Endpoint                                 │
│                    POST /audio/transcriptions                            │
│              (request_semaphore: max 20 concurrent)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌──────────────────────────────┐    ┌──────────────────────────────────────┐
│   PHASE 1: VAD Chunking      │    │         Audio Loading                │
│   (Parallel Processing)      │    │   librosa → 16kHz mono numpy         │
│                              │    └──────────────────────────────────────┘
│  ┌─────────────────────────┐ │
│  │  ProcessPoolExecutor    │ │
│  │  (VAD_WORKERS=8)        │ │
│  │                         │ │
│  │  Worker 1 ─► Silero VAD │ │
│  │  Worker 2 ─► Silero VAD │ │
│  │  ...                    │ │
│  │  Worker N ─► Silero VAD │ │
│  └─────────────────────────┘ │
│                              │
│  Output: List of chunks with │
│  timestamps & silence ratio  │
└──────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: Transcription                                │
│                                                                          │
│   Filter chunks (skip if silence_ratio > reject_segment_vad_ratio)      │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Batch Processing (CHUNK_BATCH_SIZE=8)                          │   │
│   │                                                                  │   │
│   │  Batch 1: [chunk1, chunk2, ... chunk8] ──► asyncio.gather()     │   │
│   │  Batch 2: [chunk9, chunk10, ...]        ──► asyncio.gather()    │   │
│   │  ...                                                             │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                    ┌─────────┴─────────┐                                 │
│                    ▼                   ▼                                 │
│   ┌──────────────────────────┐  ┌─────────────────────────────────────┐ │
│   │  Upstream STT API Calls  │  │  Online Diarization (if enabled)    │ │
│   │  (upstream_semaphore:    │  │  (incremental, during transcription)│ │
│   │   max 100 concurrent)    │  │                                     │ │
│   │                          │  │  • Extract embeddings (batched)     │ │
│   │  transcribe_chunk() ──►  │  │  • Assign speakers incrementally   │ │
│   │  POST to STT_API_URL     │  │  • StreamingKMeans or BIRCH cluster  │ │
│   │  (with timestamp adj.)   │  │                                     │ │
│   └──────────────────────────┘  └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Response Assembly                                   │
│                                                                          │
│   1. Combine all transcription texts                                     │
│   2. Parse timestamps into structured segments                           │
│   3. Return in requested format (text/json/verbose_json)                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Processing Flow

1. **Ingest**: Client uploads audio to `POST /audio/transcriptions`
2. **VAD + Chunking**: Audio is processed through Silero VAD in parallel workers, split into chunks based on silence detection and max chunk length (25s)
3. **Concurrent Transcription**: Chunks are sent concurrently to upstream STT API with timestamp adjustment
4. **Online Diarization** (if enabled): Processes chunks incrementally during transcription:
   - Extracts speaker embeddings in small batches (default: 4 chunks)
   - Assigns speakers incrementally using StreamingKMeans or BIRCH clustering
   - Maintains GPU batching efficiency while enabling true incremental processing
5. **Merge & Respond**: All transcriptions are merged with global timestamps and speaker assignments (if diarization enabled), then returned

### Concurrency Model

| Semaphore | Default | Purpose |
|-----------|---------|---------|
| `MAX_CONCURRENT_REQUESTS` | 20 | Limits full request processing (memory-heavy) |
| `MAX_CONCURRENT_UPSTREAM` | 100 | Limits concurrent upstream API calls (I/O-bound) |
| `VAD_WORKERS` | 8 | Process pool workers for VAD (CPU-bound) |
| `CHUNK_BATCH_SIZE` | 8 | Chunks per async transcription batch |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health/version check |
| `/audio/transcriptions` | POST | Long audio transcription with VAD chunking |
| `/transcribe` | GET | Browser UI for POST transcription |
| `/streaming` | GET | Browser UI for WebSocket streaming |
| `/ws` | WebSocket | Real-time streaming transcription with VAD |
| `/force_align` | POST | Force alignment (word-level timestamps from audio + transcript) |

---

## LiveKit Turn Detector Plugin

A custom fork of `livekit-plugins-turn-detector` with vLLM backend support for end-of-turn detection.

### Installation

```bash
# Core STT-API (includes turn detector base deps)
pip install .

# With full LiveKit agent stack
pip install ".[livekit]"
```

### Usage

```python
from stt_api.livekit_plugin.turn_detector import MultilingualModel

session = AgentSession(
    stt=groq.STT(model="whisper-large-v3-turbo", language="en"),
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=openai.TTS(model="gpt-4o-mini-tts", voice="ash"),
    turn_detection=MultilingualModel(),
)
```

### Configuration

Set the vLLM endpoint via environment variable:

```bash
export LIVEKIT_REMOTE_EOT_URL="https://your-vllm-endpoint.example.com"
```

When `LIVEKIT_REMOTE_EOT_URL` is unset, the plugin falls back to local inference using the HuggingFace model `livekit/turn-detector`.

---

## Prerequisites

- Docker and Docker Compose
- External Docker network `stt-network`

---

## Quick Start

### 1. Create External Network

```bash
docker network create stt-network
```

### 2. Run vLLM

```bash
docker compose -f vllm.yaml up --build --detach
```

Or with a private model (create `.env_vllm` with `HUGGING_FACE_HUB_TOKEN=`):

```bash
STT_MODEL=openai/whisper-large-v3-turbo GPU_MEM_UTIL=0.7 \
docker compose -f vllm.yaml up --build --detach
```

### 3. Configure Environment (Optional)

Create a `.env` file:

```bash
STT_API_URL=http://stt-engine:9089
SAMPLE_RATE=16000
MAX_CHUNK_LENGTH=25
MINIMUM_SILENT_MS=400
MINIMUM_TRIGGER_VAD_MS=1500
REJECT_SEGMENT_VAD_RATIO=0.7
VAD_THRESHOLD=0.5
MINIMUM_SPEECH_MS=250
MAX_CONCURRENT_REQUESTS=20
VAD_WORKERS=8
```

### 4. Build and Run

```bash
docker compose up --build
```

The API will be available at `http://localhost:9091`.

### Running Without Docker

```bash
uv sync
uv run uvicorn stt_api.main:app --host 0.0.0.0 --port 9091
```

---

## Usage

### Basic Transcription

```bash
curl -X POST "http://localhost:9091/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "language=en" \
  -F "response_format=json"
```

Or use the browser UI at `http://localhost:9091/transcribe`.

<img src="transcribe.png" width="50%">

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (multipart/form-data) |
| `language` | string | null | Language hint: `en`, `ms`, `zh`, `ta`, or `null` for auto-detect |
| `response_format` | string | json | Response format: `text`, `json`, or `verbose_json` |
| `vad` | string | silero | VAD backend: `silero` or `firered` |
| `vad_threshold` | float | 0.5 | Speech probability threshold (0.0–1.0) |
| `minimum_silent_ms` | int | 400 | Minimum silence duration to trigger a segment cut (ms) |
| `minimum_speech_ms` | int | 250 | Minimum speech detected before triggering transcription (ms) |
| `minimum_trigger_vad_ms` | int | 1500 | Minimum audio length before VAD can trigger (ms) |
| `reject_segment_vad_ratio` | float | 0.7 | Discard chunks where silence exceeds this ratio (0.0–1.0) |
| `diarization` | string | none | Diarization mode: `none`, `kmeans`, `birch`, or `pyannote` |
| `speaker_similarity` | float | 0.5 | Online mode: speaker clustering threshold (0.0–1.0) |
| `speaker_max_n` | int | 5 | Online mode: maximum number of speakers |

### Response Formats

**`json`** (default):
```json
{"text": "Transcribed text here..."}
```

**`verbose_json`**:
```json
{
  "language": "en",
  "duration": 144.94,
  "text": "Transcribed text here...",
  "segments": [
    {"id": 0, "start": 0.0,  "end": 3.68, "text": "First segment text."},
    {"id": 1, "start": 3.68, "end": 7.42, "text": "Second segment text."}
  ]
}
```

**`verbose_json` with diarization**:
```json
{
  "language": "en",
  "duration": 144.94,
  "text": "Hello there. Hi, how are you?",
  "segments": [
    {"id": 0, "start": 0.0,  "end": 3.68, "text": "Hello there.",    "speaker": 0},
    {"id": 1, "start": 3.68, "end": 7.42, "text": "Hi, how are you?", "speaker": 1}
  ]
}
```

**`text`**: Plain text string

---

## WebSocket Streaming

The `/ws` endpoint provides real-time streaming transcription. The client streams microphone audio over a WebSocket; the server runs VAD on incoming frames and transcribes speech segments as they are detected.

### How It Works

```
┌──────────────┐     float32 bytes      ┌───────────────────────────────────────┐
│   Browser     │ ─────────────────────► │  WebSocket /ws                        │
│   (mic @16k)  │                        │                                       │
│               │ ◄───────────────────── │  1. Buffer into numpy array           │
│  JSON results │     JSON messages      │  2. Split into 512-sample frames      │
└──────────────┘                        │  3. Run VAD (Silero or FireRed)        │
                                         │  4. Track silence / speech state       │
                                         │  5. On VAD trigger → transcribe_chunk  │
                                         │     (POST to upstream STT API)         │
                                         │  6. Send result back over WebSocket    │
                                         └───────────────────────────────────────┘
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | string | null | Language hint: `en`, `ms`, `zh`, `ta`, or `null` for auto-detect |
| `vad` | string | silero | VAD backend: `silero` or `firered` |
| `vad_threshold` | float | 0.5 | Speech probability threshold (0.0–1.0) |
| `minimum_silent_ms` | int | 400 | Minimum silence duration to trigger a segment cut (ms) |
| `minimum_speech_ms` | int | 250 | Minimum speech detected before triggering transcription (ms) |
| `minimum_trigger_vad_ms` | int | 1500 | Minimum audio length before VAD can trigger (ms) |
| `reject_segment_vad_ratio` | float | 0.7 | Discard chunks where silence exceeds this ratio (0.0–1.0) |

### Client Protocol

1. Connect to `ws://<host>/ws?language=en`
2. Send raw `Float32Array` audio bytes (16kHz, mono)
3. Receive JSON messages:

```json
// Transcription result
{"type": "transcription", "language": "en", "segments": [{"id": 0, "start": 0.0, "end": 3.5, "text": "Hello world."}]}

// Silent segment (skipped)
{"type": "silent"}

// Error
{"error": "error details"}
```

### Demo Page

Visit `http://localhost:9091/streaming` for a browser-based demo with microphone capture, audio visualizer, and live transcription display.

<img src="streaming.png" width="50%">

### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:9091/ws?language=en');
ws.binaryType = 'arraybuffer';

const audioContext = new AudioContext({ sampleRate: 16000 });
const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
const source = audioContext.createMediaStreamSource(stream);
const processor = audioContext.createScriptProcessor(4096, 1, 1);

processor.onaudioprocess = (e) => {
    if (ws.readyState === WebSocket.OPEN)
        ws.send(new Float32Array(e.inputBuffer.getChannelData(0)).buffer);
};

source.connect(processor);
processor.connect(audioContext.destination);

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'transcription') console.log(data.segments);
};
```

---

## Speaker Diarization

Optional speaker diarization to identify who is speaking in each segment.

### Modes

| Mode | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| `none` | No speaker labels (default) | Fastest | N/A |
| `kmeans` | TitaNet + StreamingKMeans (incremental, centroid-based) | Fast | Good |
| `birch` | TitaNet + StreamingBIRCH (incremental, tree-based, better for many speakers) | Fast | Good+ |
| `pyannote` | External OSD service (pyannote/speaker-diarization-3.1) | Slow | Best |

### Online Diarization

Uses TitaNet Large for speaker embeddings with batched GPU inference and an incremental clustering algorithm. Processes chunks during transcription (not after) for lower latency.

Two clustering methods are available:

- **`kmeans`**: StreamingKMeansMaxCluster — centroid-based, fast, works well for a known small number of speakers.
- **`birch`**: StreamingBIRCH — tree-based online clustering, handles more speakers and uneven distributions better.

**Parameters:**
- `speaker_similarity`: Cosine similarity threshold (0.0–1.0). Higher = stricter matching, fewer speakers. Default: `0.5`
- `speaker_max_n`: Maximum speakers to detect. Default: `5`

### Offline Diarization

Calls an external OSD service running pyannote/speaker-diarization-3.1. More accurate but requires the OSD service to be running.

### Examples

```bash
# KMeans online diarization
curl -X POST "http://localhost:9091/audio/transcriptions" \
  -F "file=@meeting.mp3" \
  -F "language=en" \
  -F "response_format=verbose_json" \
  -F "diarization=kmeans" \
  -F "speaker_similarity=0.7" \
  -F "speaker_max_n=5"

# BIRCH online diarization (better for many speakers)
curl -X POST "http://localhost:9091/audio/transcriptions" \
  -F "file=@meeting.mp3" \
  -F "language=en" \
  -F "response_format=verbose_json" \
  -F "diarization=birch" \
  -F "speaker_similarity=0.5" \
  -F "speaker_max_n=10"
```

---

## Force Alignment

The `/force_align` endpoint produces word-level timestamps by aligning a known transcript to audio using CTC forced alignment (MMS-300M model).

### How It Works

```
┌──────────────────────────────────────────────────────────────────────┐
│                         POST /force_align                            │
│                  (audio file + transcript + language)                 │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Dynamic Batching Queue                          │
│  step() loop:                                                        │
│    1. await first request                                            │
│    2. collect more within DYNAMIC_BATCHING_MICROSLEEP window         │
│    3. process batch (DYNAMIC_BATCHING_BATCH_SIZE, default: 8)       │
└──────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
┌────────────────────────────┐  ┌──────────────────────────────────────┐
│  GPU: Batch Emission       │  │  CPU: Postprocessing (parallel)      │
│  (ThreadPoolExecutor)      │  │  (ProcessPoolExecutor)               │
│                            │  │                                      │
│  1. librosa.load → 16kHz   │  │  1. Text normalization + romanization│
│  2. Window + pad audio     │  │  2. Viterbi trellis alignment        │
│  3. Model forward pass     │  │  3. Backtrack + merge segments       │
│  4. log_softmax            │  │  4. Word-level timestamps            │
└────────────────────────────┘  └──────────────────────────────────────┘
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (WAV, mp3, etc.) — ideally 30s chunks |
| `language` | string | required | Language code: `eng`, `ms`, `chi`, `ta` |
| `transcript` | string | required | Known transcript text to align |

### Example

```bash
curl -X POST "http://localhost:9091/force_align" \
  -F "file=@audio.mp3" \
  -F "language=eng" \
  -F "transcript=Yes sir, what can I help you?"
```

### Response

```json
{
  "words_alignment": [
    {"text": "Yes",  "start": 0.12, "end": 0.38, "score": 0.95},
    {"text": "sir,", "start": 0.38, "end": 0.62, "score": 0.91},
    {"text": "what", "start": 0.70, "end": 0.88, "score": 0.88},
    {"text": "can",  "start": 0.88, "end": 1.06, "score": 0.92},
    {"text": "I",    "start": 1.06, "end": 1.14, "score": 0.97},
    {"text": "help", "start": 1.14, "end": 1.36, "score": 0.94},
    {"text": "you?", "start": 1.36, "end": 1.62, "score": 0.89}
  ],
  "length": 2.78
}
```

If a client disconnects while waiting, the server cancels the pending future so the batching loop skips it.

---

## Environment Variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_API_URL` | https://stt-engine-rtx.aies.scicom.dev | Upstream STT API endpoint |
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `MAX_CHUNK_LENGTH` | 25 | Maximum chunk length (seconds) |

### VAD

| Variable | Default | Description |
|----------|---------|-------------|
| `VAD_THRESHOLD` | 0.5 | Speech probability threshold |
| `MINIMUM_SILENT_MS` | 400 | Minimum silence to cut a segment (ms) |
| `MINIMUM_SPEECH_MS` | 250 | Minimum speech before triggering transcription (ms) |
| `MINIMUM_TRIGGER_VAD_MS` | 1500 | Minimum audio length before VAD can trigger (ms) |
| `REJECT_SEGMENT_VAD_RATIO` | 0.7 | Discard chunks where silence exceeds this ratio |

### Concurrency

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_REQUESTS` | 20 | Max concurrent full requests (memory limit) |
| `MAX_CONCURRENT_UPSTREAM` | 100 | Max concurrent upstream API calls |
| `VAD_WORKERS` | 8 | Number of VAD process pool workers |
| `CHUNK_BATCH_SIZE` | 8 | Chunks per async transcription batch |

### Diarization

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ONLINE_DIARIZATION` | true | Load TitaNet model at startup |
| `OSD_API_URL` | http://osd:8000 | Offline diarization service URL |
| `SPEAKER_EMBEDDING_BATCH_SIZE` | 16 | Batch size for speaker embedding GPU inference |

### Force Alignment

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_FORCE_ALIGNMENT` | true | Load MMS alignment model at startup |
| `DYNAMIC_BATCHING_BATCH_SIZE` | 8 | Max requests batched per GPU forward pass |
| `DYNAMIC_BATCHING_MICROSLEEP` | 1e-4 | Collection window for batching (seconds) |

### Thread Limiting (VAD workers)

Set these to avoid CPU oversubscription across worker processes:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
```

---

## Testing

### Quick Reference

| Test Type | Command |
|-----------|---------|
| Unit tests | `uv run pytest tests/test_main.py tests/test_diarization.py -v` |
| Integration tests (container) | `docker compose --profile test run --rm stress-test uv run pytest tests/ -v` |
| Direct API test | `curl -X POST http://localhost:9091/audio/transcriptions -F "file=@audio.mp3"` |

### Unit Tests

```bash
uv sync --extra dev
uv run pytest tests/test_main.py tests/test_diarization.py -v

# Specific test class
uv run pytest tests/test_diarization.py::TestOnlineDiarization -v
```

### Integration Tests

```bash
# Start the API
docker compose up -d stt-api

# Run from container (container-to-container)
docker compose --profile test run --rm stress-test \
  uv run pytest tests/test_integration.py tests/test_diarization_integration.py -v

# Run from host (localhost)
STT_API_URL=http://localhost:9091 uv run pytest tests/test_integration.py -v
```

---

## Stress Testing

### Running Stress Tests

```bash
# POST transcription (50 concurrent)
docker compose -f stress-test.yaml run --rm stress-test

# POST transcription (100 concurrent)
docker compose -f stress-test.yaml run --rm -e CONCURRENCY=100 stress-test

# WebSocket streaming (50 concurrent)
docker compose -f stress-test-ws.yaml run --rm stress-test-ws

# WebSocket streaming (100 concurrent)
docker compose -f stress-test-ws.yaml run --rm -e CONCURRENCY=100 stress-test-ws

# Force alignment (100 concurrent)
docker compose -f stress-test-force-alignment.yaml run --rm -e CONCURRENCY=100 stress-test-force-alignment

# Cancellation tests
docker compose -f stress-test-cancel.yaml run --rm stress-test-cancel
docker compose -f stress-test-ws-cancel.yaml run --rm stress-test-ws-cancel

# With diarization (kmeans)
docker compose -f stress-test.yaml run --rm \
  -e CONCURRENCY=100 \
  -e DIARIZATION_MODE=kmeans \
  -e SPEAKER_SIMILARITY=0.7 \
  -e SPEAKER_MAX_N=5 \
  stress-test

# With diarization (birch)
docker compose -f stress-test.yaml run --rm \
  -e CONCURRENCY=100 \
  -e DIARIZATION_MODE=birch \
  -e SPEAKER_SIMILARITY=0.5 \
  -e SPEAKER_MAX_N=10 \
  stress-test
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CONCURRENCY` | 50 | Number of concurrent requests/clients |
| `WARMUP_COUNT` | 3 | Warmup requests before test |
| `STT_API_URL` | http://stt-api:9091 | API URL |
| `AUDIO_FILE` | /app/test_audio/masak.mp3 | Audio file to use |
| `DIARIZATION_MODE` | none | `none`, `kmeans`, `birch`, or `pyannote` |
| `SPEAKER_SIMILARITY` | 0.5 | Speaker clustering threshold (online) |
| `SPEAKER_MAX_N` | 5 | Max speakers (online) |

### Sample Output

Based on single RTX 3090 Ti,

```
Loading audio file: /app/test_audio/masak.mp3
Audio duration: 144.94s
API URL: http://stt-api:9091
Diarization mode: none

--- Warmup (3 requests) ---
  Warmup 1: 1.276s ✓
  Warmup 2: 1.245s ✓
  Warmup 3: 1.325s ✓

--- Running Stress Test (100 concurrent requests) ---
Completed in 96.898s

==================================================
STT-API STRESS TEST REPORT
==================================================

--- Test Configuration ---
Concurrency: 100
Audio Duration: 144.94s
Diarization: none
Total Requests: 100
Successful: 100
Failed: 0
Success Rate: 100.0%

--- Latency Report ---
Min Time: 11.254s
Max Time: 96.897s
Avg Time: 54.397s
P50 (Median): 54.369s
P90: 91.091s
P95: 95.709s
P99: 96.852s

--- Real-Time Factor (RTF) Report ---
(RTF < 1.0 means faster than real-time)
Min RTF: 0.078
Max RTF: 0.669
Avg RTF: 0.375
P50 RTF: 0.375
P90 RTF: 0.628
P95 RTF: 0.660
P99 RTF: 0.668

--- Throughput ---
Total Wall Time: 96.897s
Requests/second: 1.03
Audio seconds processed/second: 149.58

==================================================
```

### Sample Output WS

Based on single RTX 3090 Ti,

```
Loading audio file: /app/test_audio/masak.mp3
Audio duration: 144.94s
Audio samples: 2318976
API URL: http://stt-api:9091
WebSocket URL: ws://stt-api:9091/ws?language=ms

--- Warmup (3 clients) ---
  Warmup 1: 1.312s, 9 segments, TTFT: 0.167s [ok]
  Warmup 2: 1.273s, 9 segments, TTFT: 0.153s [ok]
  Warmup 3: 1.289s, 9 segments, TTFT: 0.153s [ok]

--- Running Stress Test (100 concurrent clients) ---
Completed in 66.398s

============================================================
STT-API WEBSOCKET STRESS TEST REPORT
============================================================

--- Test Configuration ---
Concurrency: 100
Audio Duration: 144.94s
Language: ms
Chunk Size: 100ms
Total Clients: 100
Successful: 100
Failed: 0
Success Rate: 100.0%

--- Total Session Time ---
Min: 49.259s
Max: 66.308s
Avg: 61.027s
P50: 62.617s
P90: 66.037s
P95: 66.240s
P99: 66.251s

--- Time to First Transcription (TTFT) ---
Min: 8.576s
Max: 27.832s
Avg: 12.514s
P50: 10.997s
P90: 17.927s

--- Segments ---
Total Transcription Segments: 1183
Total Silent Segments: 802
Avg Segments/Client: 11.8

--- Real-Time Factor (RTF) ---
(RTF < 1.0 means faster than real-time)
Min RTF: 0.340
Max RTF: 0.457
Avg RTF: 0.421
P50 RTF: 0.432

--- Throughput ---
Total Wall Time: 66.308s
Clients/second: 1.51
Audio seconds processed/second: 218.59

============================================================
```

### Sample Output Force Alignment

Based on single RTX 3090 Ti,

```
API URL: http://stt-api:9091
Loading audio files...
Loaded 4 audio-transcript pairs:
  husein-chinese.mp3: 2.68s [chi] "是的先生，我能帮您什么吗?"
  husein-english.mp3: 2.78s [eng] "Yes sir, what can I help you?"
  husein-tamil.mp3: 2.80s [ta] "ஆமா ஐயா, நான் உங்களுக்கு என்ன உதவி செய்ய வேண்டும்?"
  husein-malay.mp3: 2.24s [ms] "Ya encik, apa yang saya boleh tolong?"

--- Warmup (3 requests) ---
  Warmup 1 (husein-chinese.mp3): 0.132s, 12 words [ok]
  Warmup 2 (husein-english.mp3): 0.051s, 7 words [ok]
  Warmup 3 (husein-tamil.mp3): 0.052s, 8 words [ok]

--- Running Stress Test (100 concurrent requests) ---
Completed in 4.041s

============================================================
FORCE ALIGNMENT STRESS TEST REPORT
============================================================

--- Test Configuration ---
Concurrency: 100
Audio Files: 4
  husein-chinese.mp3: 2.68s (chi)
  husein-english.mp3: 2.78s (eng)
  husein-tamil.mp3: 2.80s (ta)
  husein-malay.mp3: 2.24s (ms)
Avg Audio Duration: 2.62s
Total Requests: 100
Successful: 100
Failed: 0
Success Rate: 100.0%

--- Latency Report ---
Min: 0.162s
Max: 4.038s
Avg: 2.178s
P50: 2.220s
P90: 3.719s
P95: 4.031s
P99: 4.036s

--- Real-Time Factor (RTF) ---
(RTF < 1.0 means faster than real-time)
Min RTF: 0.062
Max RTF: 1.538
Avg RTF: 0.830
P50 RTF: 0.846
P90 RTF: 1.417
P95 RTF: 1.536
P99 RTF: 1.537

--- Alignment Stats ---
Total Words Aligned: 850
Total Audio Aligned: 262.50s
Avg Words/Request: 8.5

--- Throughput ---
Total Wall Time: 4.038s
Requests/second: 24.76
Audio seconds aligned/second: 65.00

============================================================
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| **RTF** | Real-Time Factor - < 1.0 means faster than real-time playback |
| **P50/P90/P95/P99** | Latency percentiles |
| **Throughput** | Audio seconds processed per wall-clock second |

---

## VAD Benchmarking

The `benchmark_vad.py` script compares sequential vs parallel VAD processing.

### Running VAD Benchmark

```bash
# Run with default settings
python benchmark_vad.py

# Run with specific audio file
python benchmark_vad.py --audio test_audio/masak.mp3

# Run with specific number of workers
python benchmark_vad.py --workers 8

# Run with more iterations
python benchmark_vad.py --runs 5
```

### Benchmark Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--audio` | test_audio/masak.mp3 | Path to audio file |
| `--workers` | 4 | Number of worker processes for parallel mode |
| `--runs` | 3 | Number of benchmark iterations |

### Environment Variables (for parallel mode)

Set these to limit per-process threading and avoid CPU oversubscription:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmark_vad.py
```

### Sample Output

```
============================================================
VAD BENCHMARK: Sequential vs Parallel
============================================================

Loading audio: test_audio/masak.mp3
Audio duration: 144.94s (2319040 samples)
Number of workers for parallel: 4

--- Sequential VAD ---
  Run 1: 2.345s (42 chunks)
  Run 2: 2.312s (42 chunks)
  Run 3: 2.298s (42 chunks)
  Average: 2.318s, Min: 2.298s

--- Parallel VAD (4 workers) ---
  Run 1: 0.892s (45 chunks)
  Run 2: 0.876s (45 chunks)
  Run 3: 0.881s (45 chunks)
  Average: 0.883s, Min: 0.876s

============================================================
RESULTS SUMMARY
============================================================

| Method     | Avg Time | Min Time | Chunks | Speedup |
|------------|----------|----------|--------|---------|
| Sequential |   2.318s |   2.298s |     42 | 1.00x   |
| Parallel   |   0.883s |   0.876s |     45 | 2.62x   |

✅ Parallel is 2.62x FASTER than sequential

VAD RTF (lower is better):
  Sequential: 0.0160 (62.5x faster than real-time)
  Parallel:   0.0061 (164.1x faster than real-time)
============================================================
```

### Understanding Results

- **Chunks difference**: Parallel may produce slightly more chunks due to VAD state not being shared across segment boundaries
- **Speedup**: Parallel speedup depends on CPU cores and audio length
- **RTF**: Both are much faster than real-time; the bottleneck is upstream transcription, not VAD

---

## Tuning for Production

### Memory

Each concurrent request loads audio into memory:
- 1 min audio @ 16kHz mono ≈ 1.9 MB
- 20 concurrent requests × 2.5 min ≈ ~95 MB audio buffers

### Recommended Configurations

```bash
# Conservative (8GB RAM)
MAX_CONCURRENT_REQUESTS=10  VAD_WORKERS=4

# Standard (16GB RAM)
MAX_CONCURRENT_REQUESTS=20  VAD_WORKERS=8

# High-capacity (32GB+ RAM)
MAX_CONCURRENT_REQUESTS=50  VAD_WORKERS=16
```

### Monitoring

Watch these metrics under load:
- Memory usage (keep below 80%)
- P95 latency (should stay below audio duration for real-time processing)
- Success rate (should be 100%)

---
 
## Diarization Benchmark
 
### Overview
 
Benchmarks the diarization component of STT-API using the AMI dataset from HuggingFace. Evaluation measures how accurately the system identifies **who spoke when** in a multi-speaker recording.
 
Two online diarization algorithms are benchmarked:
- **kmeans** — TitaNet Large + StreamingKMeans clustering
- **birch** — TitaNet Large + StreamingBIRCH clustering
 
Each algorithm is tested across 7 `speaker_similarity` thresholds (0.2 → 0.8).
 
### What is DER?
 
**DER (Diarization Error Rate)** is the standard metric for evaluating speaker diarization. It measures the fraction of time incorrectly attributed to the wrong speaker or incorrectly labeled as speech/non-speech.
 
```
DER = (False Alarm + Missed Detection + Speaker Confusion) / Total Reference Speech Duration
```
 
| Component | Description |
|---|---|
| **False Alarm** | System labels non-speech regions as speech |
| **Missed Detection** | System fails to detect actual speech |
| **Speaker Confusion** | Speech is attributed to the wrong speaker |
 
A lower DER means better performance. A DER of 0% means perfect diarization.
 
### Datasets
 
| Dataset | Description |
|---|---|
| [diarizers-community/ami](https://huggingface.co/datasets/diarizers-community/ami) | Meeting recordings with multiple speakers, challenging overlapping speech |
| [diarizers-community/voxconverse](https://huggingface.co/datasets/diarizers-community/voxconverse) | Multispeaker audio dataset derived from YouTube videos |
 
### Baseline Results (diarizers Test class)
 
Evaluated using the `Test` class from [huggingface/diarizers](https://github.com/huggingface/diarizers/blob/main/src/diarizers/test.py) directly against the pyannote segmentation model.
 
| Dataset | DER | False Alarm | Missed Detection | Confusion |
|---|---|---|---|---|
| AMI | 17.93% | 4.03% | 10.04% | 3.86% |
| VoxConverse | 11.20% | 4.32% | 3.52% | 3.36% |
 
### STT-API Diarization Results (AMI)
 
Benchmarked kmeans and birch across `speaker_similarity` thresholds 0.2 → 0.8 on the AMI test set.
 
Total benchmark time: **16.35 minutes**
 
| Algorithm | Similarity | AMI DER (%) |
|---|---|---|
| kmeans | 0.2 | 87.77 |
| kmeans | 0.3 | 86.20 |
| kmeans | 0.4 | 82.71 |
| kmeans | 0.5 | 80.17 |
| kmeans | 0.6 | 78.19 |
| kmeans | 0.7 | 76.64 |
| kmeans | 0.8 | 75.41 |
| birch | 0.2 | 83.98 |
| birch | 0.3 | 85.49 |
| birch | 0.4 | 82.22 |
| birch | 0.5 | 79.52 |
| birch | 0.6 | 77.68 |
| birch | 0.7 | 77.70 |
| **birch** | **0.8** | **75.38** ✅ best |
 
### Key Findings
 
- **Best: birch with `speaker_similarity=0.8`**, achieving DER of **75.38%** on AMI.
- Both algorithms show a clear trend: higher similarity threshold → lower DER.
- birch slightly outperforms kmeans across most similarity thresholds.
- Lower similarity thresholds (0.2–0.3) produce higher DER (84–88%) due to over-segmentation.
 
### How to Reproduce
 
```bash
# 1. Install dependencies
pip install pyannote.metrics soundfile "datasets==2.21.0" aiohttp onnxruntime
 
# 2. Start the local STT-API server (in a separate terminal)
STT_API_URL=https://stt-engine-tm-l40.aies.scicom.dev uvicorn stt_api.main:app --host 0.0.0.0 --port 9091
 
# 3. Run the benchmark
python3.10 benchmark_diarization.py
```
 
 
### Environment
 
| Component | Details |
|---|---|
| Dataset | [diarizers-community/ami](https://huggingface.co/datasets/diarizers-community/ami) IHM, test split |
| Baseline evaluation | [huggingface/diarizers test.py](https://github.com/huggingface/diarizers/blob/main/src/diarizers/test.py) |
| Speaker embedding model | TitaNet Large (`huseinzol05/nemo-titanet_large`) |
| Online clustering | StreamingKMeans / StreamingBIRCH |
| STT engine | `https://stt-engine-tm-l40.aies.scicom.dev` |
| Evaluation metric | DER via `pyannote.metrics` |
| Benchmark script | `benchmark_diarization.py` |