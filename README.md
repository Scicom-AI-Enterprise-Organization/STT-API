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
| `/audio/speaker_vector` | POST | Speaker embeddings (TitaNet Large) for one or more files |
| `/transcribe` | GET | Browser UI for POST transcription |
| `/streaming` | GET | Browser UI for WebSocket streaming |
| `/ws` | WebSocket | Real-time streaming transcription with VAD |
| `/force_align` | POST | Force alignment (word-level timestamps from audio + transcript) |

---

## LiveKit Plugins

Plugins for [LiveKit Agents](https://docs.livekit.io/agents/), shipped from this
repo and installable **without the STT server stack**:

| Plugin | What it does |
|---|---|
| `WhisperSTT` `DropBlankSTT` | Whisper STT that will not let silence interrupt the agent — [whisper_stt](stt_api/livekit_plugin/whisper_stt/README.md) |
| `PulseVAD` | Frame-level voice activity, 2,118 parameters — [pulse_vad](stt_api/livekit_plugin/pulse_vad/README.md) |
| `SemanticVAD` | End of turn decided from audio, ahead of the transcript — [semantic_vad](stt_api/livekit_plugin/semantic_vad/README.md) |
| `MultilingualModel` | End of turn from text — a fork of `livekit-plugins-turn-detector` with a vLLM backend |
| `GTCRN` | Self-hosted noise cancellation, in-process, no LiveKit Cloud licence |
| `DummySTT` `DummyLLM` `DummyTTS` | Canned responses for load tests, zero API cost |

Index and install notes: [`stt_api/livekit_plugin/`](stt_api/livekit_plugin/README.md).

### Install into an agent

```bash
uv pip install "stt-api[scicom-livekit-plugin] @ git+https://github.com/Scicom-AI-Enterprise-Organization/STT-API.git"
```

or as a dependency of the agent project:

```toml
[project]
dependencies = [
    "stt-api[scicom-livekit-plugin] @ git+https://github.com/Scicom-AI-Enterprise-Organization/STT-API.git",
]
```

Append `@<sha-or-tag>` to the URL to pin. The repo is private, so the installing
machine needs a Git credential (`gh auth setup-git`, or a token in the URL for CI).

One extra covers every plugin above: `livekit-agents[turn-detector]`, `aiohttp`,
`transformers`, `huggingface_hub`, `onnxruntime`, `numpy` and `soundfile` — and
nothing from `[server]`: no torch, no fastapi, no diarization stack.
`tests/test_packaging.py` enforces that.

### Install for the server or for development

```bash
pip install ".[server]"                          # the STT API server stack
pip install ".[server,scicom-livekit-plugin]"    # both
pip install ".[dev]"                             # test tooling
pip install ".[benchmark]"                       # the noise-cancellation and VAD shootouts
pip install .                                    # the package alone: zero dependencies
```

`livekit` is an alias for `scicom-livekit-plugin`; both install the same set.
WER/CER scoring (`stt_api.evaluation`) works with the core install — `[evaluation]`
is only needed by `load_canonical`, which reads dataset variant maps from
HuggingFace.

### Usage

One import line, whichever plugins the agent uses:

```python
from stt_api.livekit_plugin import GTCRN, PulseVAD, SemanticVAD, WhisperSTT

session = AgentSession(
    stt=WhisperSTT(),                                 # reads STT_URL / STT_API / STT_MODEL
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=openai.TTS(model="gpt-4o-mini-tts", voice="ash"),
    vad=ctx.proc.userdata["vad"],                     # PulseVAD.load() in prewarm
    turn_detection=SemanticVAD(backend=ScicomEoT()),
)
```

Names resolve on first use, so importing one plugin does not load the others'
dependencies. Each plugin's README has the real usage.

### Configuration

Set the vLLM endpoint for the turn detector via environment variable:

```bash
export LIVEKIT_REMOTE_EOT_URL="https://your-vllm-endpoint.example.com"
```

When `LIVEKIT_REMOTE_EOT_URL` is unset, the plugin falls back to local inference using the HuggingFace model `livekit/turn-detector`.

---

## LiveKit Semantic VAD Plugin

`stt_api/livekit_plugin/semantic_vad/` — end-of-turn decided from the **waveform**
rather than the transcript.

The turn detector above is a text model: it cannot answer until the STT has
produced words, and on the production stack that transcript arrives a median
1.5 s after the speaker stops. An audio-native model answers in tens of
milliseconds from audio the agent already has. They are complementary, not
alternatives — text sees semantics ("...and then he said" is clearly unfinished),
audio sees prosody a transcript throws away.

```python
from stt_api.livekit_plugin.semantic_vad import SemanticVAD, ScicomEoT

session = AgentSession(
    stt=..., llm=..., tts=...,
    vad=ctx.proc.userdata["vad"],          # still required — it decides *when* to ask
    turn_detection=SemanticVAD(backend=ScicomEoT("base")),
)
```

| backend | model | languages | ms/call |
|---|---|---|---|
| `ScicomEoT("tiny"\|"base"\|"small")` | `Scicom-intl/semantic-vad-eot-whisper-*` | **ms**, en | 24 / 43 / 145 |
| `SmartTurnV3()` | `pipecat-ai/smart-turn-v3` | 23 (no `ms`) | 36 |
| `RemoteEoT(url)` | your own GPU, plain JSON POST | — | network |

`ScicomEoT` is trained on real Malaysian call-centre telephony, which is why it is
the default choice here; `base` is the sensible size (`small` is 4x the CPU for
+0.02 AUC, `tiny` is for when CPU binds).

Verified end to end against a real `AgentSession`: p(eot) 0.9 against a 0.2
threshold commits the turn at +0.51 s, and 0.05 against 0.5 waits to +2.93 s — the
probability moves the boundary by 2.4 s. Run it with
`RUN_LIVEKIT_INTEGRATION=1 pytest tests/test_semantic_vad.py`.

Self-hosting needs **no protobuf websocket server**, which is the usual assumption
about LiveKit's audio EoT path — see
[`stt_api/livekit_plugin/semantic_vad/README.md`](stt_api/livekit_plugin/semantic_vad/README.md).

---

## LiveKit PulseVAD Plugin

`stt_api/livekit_plugin/pulse_vad/` — frame-level voice activity detection from a
**2,118-parameter** CNN ([PulseVAD](https://github.com/AydinAdnan/PulseVAD), MIT).
A drop-in for `silero.VAD`: 76 KB of weights and filterbank against silero's
2.22 MB.

This answers a different question from the two plugins above. They decide *has
the speaker finished*; this decides *is someone speaking right now*. An
`AgentSession` wants both — the VAD decides when to ask, the turn detector
answers.

```python
from stt_api.livekit_plugin.pulse_vad import PulseVAD

def prewarm(proc):
    proc.userdata["vad"] = PulseVAD.load()          # 2.1k fp32; blocking

session = AgentSession(
    vad=ctx.proc.userdata["vad"],
    turn_detection=SemanticVAD(backend=ScicomEoT("base")),
    stt=..., llm=..., tts=...,
)
```

Two things will bite you, and both are silent:

- **p(speech) saturates at 0.711**, not 1.0 (0.895 for the `81k` teacher).
  Carrying silero's `activation_threshold=0.8` across gives a VAD that fires on
  **0 %** of windows — an agent that never hears anyone, with no error. `load()`
  refuses a threshold at or above the ceiling. The default is `0.35`, the
  midpoint of the measured range.
- **`precision="fp32"` is the default**, unlike upstream. int8 measured *no
  faster* off-microcontroller (0.019 ms vs 0.018 ms) in a file twice the size.

The model's 200 ms window **slides** over a 32 ms hop, so it reports at silero's
cadence rather than breaking LiveKit's duration defaults — about 0.3 % of one
core.

### Measured against silero on labelled production audio

`benchmark/` runs both through the real `livekit.agents.vad.VAD` interface on
turns from `stt-dev/drift/`, where an **empty Whisper transcript is a genuine
no-speech label**.

```bash
pip install ".[benchmark]"
python -m stt_api.livekit_plugin.pulse_vad.benchmark \
    --s3-prefix stt-dev/drift/proxy-.../2026-09-22/ --limit 1500
```

Each model is scored at **its own** best threshold — scoring both at 0.5 would
compare silero's mid-range against PulseVAD's 70th percentile and call the
difference accuracy.

On 1,500 production turns (4.4 h) the two are **statistically indistinguishable**
on both detection (99.02 % vs 99.17 %, p = 0.84) and false fires (4.44 % vs
2.78 %, p = 0.57). What differs is cost: PulseVAD is **1.33x more expensive per
window** and commits a turn **96 ms later**. Its 2,118 parameters buy footprint
(76 KB vs 2.22 MB), not CPU — the numpy log-mel front end costs 4x the model, so
the *model* is 4.5x cheaper than silero's while the *package* is 1.33x dearer.

**The recommendation is to keep silero** unless model footprint is the binding
constraint. Full numbers and method in
[`PULSE_VAD_BENCH.md`](PULSE_VAD_BENCH.md).

---

## LiveKit Whisper STT Plugin

`stt_api/livekit_plugin/whisper_stt/` — an OpenAI-compatible Whisper client that
**refuses to turn silence into an interruption**.

```python
from livekit.plugins import silero
from stt_api.livekit_plugin.whisper_stt import WhisperSTT

session = AgentSession(
    stt=WhisperSTT(),          # reads STT_URL / STT_API / STT_MODEL
    vad=silero.VAD.load(),     # required — this STT is not streaming
    llm=..., tts=...,
)
```

**Whisper answers silence with a single space, not an empty string** — measured,
`' '` comes back with HTTP 200 for digital silence, room tone, and even 0.10 s of
real speech. Across 1,503 production turns, **180 (12 %)** returned exactly `' '`.

A space is truthy and LiveKit only tests emptiness, so it passes
`audio_recognition.py:1215` and reaches `on_final_transcript`, which calls
`_interrupt_by_audio_activity()` with no text check. The agent gets cut off by
silence with nothing logged:

| STT returns | interrupts the agent |
|---|---|
| `''` | no |
| `' '`, `'  \n\t'`, `'.'` | **yes** |

Two guards, both measured:

- **Blank transcripts return `text=""`** — never `alternatives=[]`, which would
  swap the interruption for an `IndexError` at `audio_recognition.py:1207`.
  Filtering is `.strip()` length, not a character class: four production turns
  are real Tamil and Chinese speech with no ASCII alphanumerics.
- **Silent segments never reach the network.** Speech turns sit at −18 dBFS
  median, non-speech at −240 (169 of 180 are literally all zeros, each exactly
  0.30 s). An RMS floor of −50 dBFS skips **94 %** of the useless calls and
  loses **0** of 1,323 real turns.

`DropBlankSTT(openai.STT())` applies the blank guard to any other provider —
`livekit-plugins-openai` has the same defect.

**It does not stop the interruption, though** — measured in a real
`AgentSession`, a 0.35 s VAD false positive cuts the agent's TTS *even when the
STT returns `''`*, because `speech_duration` accumulates through silero's 0.55 s
hangover and a 0.15 s burst already reports 0.70 s. What the guard buys is no
chat bubble and no LLM context from silence, plus ~94 % fewer STT calls.

The lever that gates the VAD path too is `min_words`, inert at its default of 0:

```python
session = AgentSession(
    stt=WhisperSTT(),
    vad=silero.VAD.load(),
    llm=..., tts=...,
    min_interruption_words=1,                            # livekit-agents 1.3.x-1.7.x
    # turn_handling={"interruption": {"min_words": 1}},  # 1.8+ spelling
)
```

`turn_handling` does not exist before 1.8 and raises `TypeError` there;
[`example_agent.py`](stt_api/livekit_plugin/whisper_stt/example_agent.py) picks
the right spelling at runtime and is runnable as-is.

Measured end to end on 1.3.11 with the agent mid-TTS: an unfiltered STT fires a
chat bubble carrying `' '` **and** cuts the TTS; `WhisperSTT` fires no bubble;
`min_words=1` stops the cut — and real speech still barges in. The plugin
removes the bubble, `min_words` removes the interruption, and you need both.
Verified against **1.3.11 and 1.8.2**.

`''`, `' '` and `'.'` all split to 0 words; `' ஹலோ மாயா!'` splits to 2. See
[`whisper_stt/README.md`](stt_api/livekit_plugin/whisper_stt/README.md) for the
measurements and the one control still unverified.

---

## Transcription Scoring (`stt_api.evaluation`)

WER/CER for ASR output, reported twice: as an ordinary scorer charges it, and again with
spelling conventions normalized away — so you can see how much of a WER is the model and
how much is the reference and the hypothesis disagreeing about how to *write* the same
words (`dua puluh tiga` vs `23`, `okay lah` vs `ok la`, `[laugh]` vs nothing).

Standard library only; the default mode does no network I/O.

```python
from stt_api.evaluation import score

r = score("saya bayar 23 ringgit", "saya bayar dua puluh tiga ringgit")
r.wer                     # [0.5]   as scored
r.normalized_wer          # [0.0]   the whole error was how a number was written
r.normalized_hypothesis   # ['saya bayar 23 ringgit']
r.normalized_reference    # ['saya bayar 23 ringgit']
```

Argument order is `(hypothesis, reference)` — ASR output first. Pass one string pair or
two equal-length lists; the result is always lists, plus pooled `corpus_wer` /
`corpus_cer`.

```bash
python -m stt_api.evaluation --input pairs.csv      # ref,hyp columns
python -m stt_api.evaluation --self-test            # offline checks, no data needed
```

An optional LLM pass handles the residue the rules cannot settle. Export three variables and
it needs nothing else — no file, no client object:

```bash
export OPENAI_BASE_URL=https://your-endpoint.example.com/v1
export OPENAI_API_KEY=sk-...
export MODEL_NAME=your-model-id
```

```python
r = score(hyps, refs, mode="llm")     # reads the environment on its own
```

Every edit the model proposes is validated as convention-only and reverted otherwise.
(A `.env` works too — see [`stt_api/evaluation/.env.example`](stt_api/evaluation/.env.example).)

See [`stt_api/evaluation/README.md`](stt_api/evaluation/README.md) for the full API,
`.env` setup, dataset variant maps, and how to read the numbers honestly.

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
uv sync --extra server
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

### Speaker Vectors

Speaker embeddings from TitaNet Large — the same model and the same batched GPU
path the online diarization uses.

```bash
# one file -> one vector
curl -X POST "http://localhost:9091/audio/speaker_vector" -F "file=@speaker.wav"
```

```json
{"vectors": [[0.0123, -0.0456, ...]], "dim": 192, "count": 1, "normalized": false}
```

**Send several files in one request.** They are embedded as a single GPU batch,
which is the reason to batch rather than loop over calls — one round trip and one
kernel launch instead of N:

```bash
curl -X POST "http://localhost:9091/audio/speaker_vector" \
  -F "file=@a.wav" -F "file=@b.wav" -F "file=@c.wav"
```

Vectors come back in request order, so `vectors[i]` belongs to the i-th file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file; repeat the field to batch several |
| `normalize` | bool | `false` | L2-normalise each vector so a dot product *is* cosine similarity |

`normalize` is off by default because raw is what the model emits and what the
diarization path consumes — `clustering_torch` computes `cosine_similarity`
explicitly on unnormalised vectors. Turn it on only if you are storing vectors to
compare by dot product later, and be consistent: comparing a normalised vector
against a raw one silently gives the wrong distance.

Comparing two speakers:

```python
import numpy as np, requests

r = requests.post("http://localhost:9091/audio/speaker_vector",
                  files=[("file", open("a.wav","rb")), ("file", open("b.wav","rb"))])
a, b = (np.array(v) for v in r.json()["vectors"])
similarity = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
```

**Minimum length is 0.5 s** (`MIN_CHUNK_SAMPLES_FOR_EMBEDDING`, 8000 samples at
16 kHz). Shorter clips are rejected with a 400 rather than embedded: TitaNet
needs enough frames for its mel spectrogram and below that returns a plausible-
looking vector that means nothing.

Requires the speaker model, so it shares `ENABLE_ONLINE_DIARIZATION`'s worker; if
that failed to load the endpoint returns 503.

#### Measured speed

One idle H20, via the same `extract_embeddings_batched` path the endpoint calls.
Full tables, the 6x optimisation and caveats:
[`SPEAKER_VECTOR_BENCH.md`](SPEAKER_VECTOR_BENCH.md).

| | |
|---|---|
| single 3 s file, end to end | **11.5 ms** (4.7 ms GPU, 0.4 ms decode) |
| 16 files in one request | 25.3 ms total, **1.58 ms per vector** |
| sustained throughput | **~1760 vectors/s** |
| RTF | ~0.0001, i.e. ~9000x faster than real time |

- **Batching is worth ~8x and saturates at 16**, which is why
  `SPEAKER_EMBEDDING_BATCH_SIZE` defaults to 16. Larger requests still work (they
  chunk internally); past 16 you trade latency for throughput already collected.
- **Cost tracks audio duration, not file count.** 16 half-second clips cost 6 ms;
  one 30-second clip costs 51 ms. Decoding is negligible (0.4 ms p50).
- **Ragged batches shift vectors slightly** (cosine 0.96-1.0 vs computing a clip
  alone) because the model sees padding. Enroll and query the same way, or batch
  clips of similar length.

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
