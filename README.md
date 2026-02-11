# stt-api

Long-form speech-to-text API that:

- **Chunks long audio** using Silero VAD into manageable pieces
- **Keeps global timestamps** across all chunks
- **Transcribes chunks concurrently** for improved performance
- **Proxies to an upstream STT engine** via an OpenAI-compatible `/v1/audio/transcriptions` endpoint
- **Speaker diarization** with online (TitaNet + StreamingKMeans + CUDA Streams) or offline (pyannote) modes

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
│   Filter chunks (skip if silence_ratio > 0.9)                           │
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
│   │  POST to STT_API_URL     │  │  • Reuse StreamingKMeansMaxCluster  │ │
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
   - Assigns speakers incrementally using StreamingKMeansMaxCluster
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

### 2. Run vLLM,

```bash
docker compose -f vllm.yaml up --detach
```

Or if you have private model,

Make sure you create [.env_vllm](.env_vllm),

```
HUGGING_FACE_HUB_TOKEN=
```

```bash
STT_MODEL=openai/whisper-medium GPU_MEM_UTIL=0.7 \
docker compose -f vllm.yaml up --detach
```

### 3. Configure Environment (Optional)

Create a `.env` file:

```bash
STT_API_URL=http://stt-engine:9089
SAMPLE_RATE=16000
MAX_CHUNK_LENGTH=25
MINIMUM_SILENT_MS=200
MINIMUM_TRIGGER_VAD_MS=1500
REJECT_SEGMENT_VAD_RATIO=0.9
MAX_CONCURRENT_REQUESTS=20
VAD_WORKERS=8
```

### 4. Build and Run

```bash
# Start the service
docker-compose up --build

# Or run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f stt-api

# Stop the service
docker-compose down
```

The API will be available at `http://localhost:9090`.

### Running Without Docker

```bash
# Install dependencies
uv sync

# Run the server
uv run uvicorn app.main:app --host 0.0.0.0 --port 9090
```

---

## Usage

### Basic Transcription

```bash
curl -X POST "http://localhost:9090/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "language=en" \
  -F "response_format=json"
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (multipart/form-data) |
| `language` | string | null | Language hint: `en`, `ms`, `zh`, `ta`, or `null` for auto-detect |
| `response_format` | string | json | Response format: `text`, `json`, or `verbose_json` |
| `minimum_silent_ms` | int | 200 | Minimum silence duration for VAD trigger (ms) |
| `minimum_trigger_vad_ms` | int | 1500 | Minimum audio length to trigger VAD (ms) |
| `reject_segment_vad_ratio` | float | 0.9 | Reject segments with this ratio of silence (0.0-1.0) |
| `diarization` | string | none | Diarization mode: `none`, `online`, or `offline` |
| `speaker_similarity` | float | 0.3 | Online mode: speaker clustering threshold (0.0-1.0) |
| `speaker_max_n` | int | 10 | Online mode: maximum number of speakers |

### Response Formats

**`json`** (default):
```json
{
  "text": "Transcribed text here..."
}
```

**`verbose_json`**:
```json
{
  "language": "en",
  "duration": 144.94,
  "text": "Transcribed text here...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.68,
      "text": "First segment text."
    },
    {
      "id": 1,
      "start": 3.68,
      "end": 7.42,
      "text": "Second segment text."
    }
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
    {
      "id": 0,
      "start": 0.0,
      "end": 3.68,
      "text": "Hello there.",
      "speaker": 0
    },
    {
      "id": 1,
      "start": 3.68,
      "end": 7.42,
      "text": "Hi, how are you?",
      "speaker": 1
    }
  ]
}
```

**`text`**: Plain text string

---

## Speaker Diarization

The API supports optional speaker diarization to identify who is speaking in each segment.

### Diarization Modes

| Mode | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| `none` | No speaker labels (default) | Fastest | N/A |
| `online` | TitaNet + StreamingKMeans (incremental during transcription) | Fast | Good |
| `offline` | External OSD service (pyannote) | Slow | Best |

### Online Diarization

Uses TitaNet Large for speaker embeddings with batched GPU inference, combined with StreamingKMeansMaxCluster for incremental speaker assignment. **Processes chunks incrementally during transcription** (not after all chunks complete), providing lower latency and better performance than batch processing.

**How it works:**
- Creates a single `StreamingKMeansMaxCluster` instance at the start
- As chunks are transcribed, extracts embeddings in small batches (default: 4 chunks)
- Assigns speakers incrementally using the shared cluster instance
- Maintains GPU batching efficiency while enabling true incremental processing

**Parameters:**
- `speaker_similarity`: Cosine similarity threshold (0.0-1.0). Higher = stricter matching, fewer speakers. Default: 0.3
- `speaker_max_n`: Maximum speakers to detect. Default: 10

### Offline Diarization

Calls an external OSD (Offline Speaker Diarization) service running pyannote/speaker-diarization-3.1. More accurate but requires the OSD service to be running and takes longer.

### Example with Diarization

```bash
curl -X POST "http://localhost:9090/audio/transcriptions" \
  -F "file=@meeting.mp3" \
  -F "language=en" \
  -F "response_format=verbose_json" \
  -F "diarization=online" \
  -F "speaker_similarity=0.7" \
  -F "speaker_max_n=5"
```

---

## Testing

### Quick Reference

| Test Type | Command |
|-----------|---------|
| Direct API test | `curl -X POST http://localhost:9090/audio/transcriptions -F "file=@audio.mp3"` |
| Unit tests | `uv run pytest tests/test_main.py tests/test_diarization.py -v` |
| Integration tests (container) | `docker compose --profile test run --rm stress-test uv run pytest tests/ -v` |
| Stress test | `docker compose run --rm stress-test` |
| Stress test with diarization | `docker compose run --rm -e DIARIZATION_MODE=online stress-test` |

### Direct API Testing

Test the API directly using curl:

```bash
# Basic transcription
curl -X POST "http://localhost:9090/audio/transcriptions" \
  -F "file=@test_audio/masak.mp3" \
  -F "language=ms" \
  -F "response_format=json"

# With online diarization (speaker_similarity defaults to 0.3)
curl -X POST "http://localhost:9090/audio/transcriptions" \
  -F "file=@test_audio/masak.mp3" \
  -F "response_format=verbose_json" \
  -F "diarization=online" \
  -F "speaker_max_n=5"

# With offline diarization (requires OSD service)
curl -X POST "http://localhost:9090/audio/transcriptions" \
  -F "file=@test_audio/masak.mp3" \
  -F "response_format=verbose_json" \
  -F "diarization=offline"

# Health check
curl http://localhost:9090/
```

### Unit Tests

Run unit tests locally (no running API needed):

```bash
# Install dev dependencies
uv sync --extra dev

# Run all unit tests
uv run pytest tests/test_main.py tests/test_diarization.py -v

# Run specific test
uv run pytest tests/test_diarization.py::TestOnlineDiarization -v
```

### Integration Tests (Container-to-Container)

Run integration tests from within the Docker network:

```bash
# Start the API first
docker compose up -d stt-api

# Run integration tests from stress-test container
docker compose --profile test run --rm stress-test \
  uv run pytest tests/test_integration.py tests/test_diarization_integration.py -v

# Run only diarization tests
docker compose --profile test run --rm stress-test \
  uv run pytest tests/test_diarization_integration.py -v

# Run parameterized tests for a specific diarization mode
docker compose --profile test run --rm stress-test \
  uv run pytest tests/test_diarization_integration.py -k "online" -v
```

### Integration Tests (Local to Container)

Run integration tests from your local machine against the containerized API:

```bash
# Start the API
docker compose up -d stt-api

# Install dev dependencies locally
uv sync --extra dev

# Run tests (pointing to localhost)
STT_API_URL=http://localhost:9090 uv run pytest tests/test_integration.py -v
```

---

## Stress Testing

The `stress_test.py` script benchmarks API performance under concurrent load.

### Running Stress Tests

```bash
# Run with default settings (50 concurrent requests, no diarization)
docker compose run --rm stress-test

# Run with online diarization
docker compose run --rm -e DIARIZATION_MODE=online stress-test

# Run with offline diarization
docker compose run --rm -e DIARIZATION_MODE=offline stress-test

# Run with custom concurrency and diarization
docker compose run --rm \
  -e CONCURRENCY=100 \
  -e DIARIZATION_MODE=online \
  -e SPEAKER_SIMILARITY=0.8 \
  -e SPEAKER_MAX_N=5 \
  stress-test

# Run with custom audio file
docker compose run --rm -e AUDIO_FILE=/app/test_audio/custom.mp3 stress-test
```

### Stress Test Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `CONCURRENCY` | 50 | Number of concurrent requests |
| `WARMUP_COUNT` | 3 | Number of warmup requests before test |
| `STT_API_URL` | http://stt-api:9090 | API URL to test |
| `AUDIO_FILE` | /app/test_audio/masak.mp3 | Audio file for testing |
| `DIARIZATION_MODE` | none | Diarization mode: `none`, `online`, `offline` |
| `SPEAKER_SIMILARITY` | 0.3 | Speaker clustering threshold (online mode) |
| `SPEAKER_MAX_N` | 10 | Maximum speakers to detect (online mode) |

### Sample Output

```
==================================================
STT-API STRESS TEST REPORT
==================================================

--- Test Configuration ---
Concurrency: 100
Audio Duration: 144.94s
Diarization: online
  Speaker Similarity: 0.75
  Max Speakers: 10
Total Requests: 100
Successful: 100
Failed: 0
Success Rate: 100.0%

--- Latency Report ---
Min Time: 7.320s
Max Time: 192.346s
Avg Time: 94.513s
P50 (Median): 97.060s
P90: 158.926s
P95: 167.154s
P99: 171.979s

--- Real-Time Factor (RTF) Report ---
(RTF < 1.0 means faster than real-time)
Min RTF: 0.051
Max RTF: 1.327
Avg RTF: 0.652
P50 RTF: 0.670
P90 RTF: 1.096

--- Throughput ---
Total Wall Time: 192.346s
Requests/second: 0.52
Audio seconds processed/second: 75.36
==================================================
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

## Environment Variables Reference

### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_API_URL` | https://stt-engine-rtx.aies.scicom.dev | Upstream STT API endpoint |
| `SAMPLE_RATE` | 16000 | Audio sample rate in Hz |
| `MAX_CHUNK_LENGTH` | 25 | Maximum chunk length in seconds |

### VAD Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIMUM_SILENT_MS` | 200 | Minimum silence duration for VAD trigger (ms) |
| `MINIMUM_TRIGGER_VAD_MS` | 1500 | Minimum audio length to trigger VAD (ms) |
| `REJECT_SEGMENT_VAD_RATIO` | 0.9 | Reject segments with this ratio of silence |

### Concurrency Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_REQUESTS` | 20 | Max concurrent full requests (memory limit) |
| `MAX_CONCURRENT_UPSTREAM` | 100 | Max concurrent upstream API calls |
| `VAD_WORKERS` | 8 | Number of VAD process pool workers |
| `CHUNK_BATCH_SIZE` | 8 | Chunks per async transcription batch |

### Thread Limiting (for VAD workers)

| Variable | Recommended | Description |
|----------|-------------|-------------|
| `OMP_NUM_THREADS` | 2 | OpenMP threads per process |
| `OPENBLAS_NUM_THREADS` | 2 | OpenBLAS threads per process |
| `MKL_NUM_THREADS` | 2 | Intel MKL threads per process |

### Diarization Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OSD_API_URL` | http://osd:8000 | Offline diarization service URL |
| `ENABLE_ONLINE_DIARIZATION` | true | Load TitaNet model at startup |
| `SPEAKER_EMBEDDING_BATCH_SIZE` | 16 | Batch size for speaker embedding GPU inference |

---

## Tuning for Production

### Memory Considerations

Each concurrent request loads audio into memory:
- 1 minute audio @ 16kHz mono ≈ 1.9 MB numpy array
- 20 concurrent requests with 2.5 min audio each ≈ 95 MB just for audio buffers

### Recommended Starting Configuration

```bash
# Conservative (8GB RAM)
MAX_CONCURRENT_REQUESTS=10
VAD_WORKERS=4

# Standard (16GB RAM)
MAX_CONCURRENT_REQUESTS=20
VAD_WORKERS=8

# High-capacity (32GB+ RAM)
MAX_CONCURRENT_REQUESTS=50
VAD_WORKERS=16
```

### Monitoring

Watch these metrics under load:
- Memory usage (should stay below 80%)
- P95 latency (should stay below audio duration for real-time processing)
- Success rate (should be 100%)
