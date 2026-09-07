# CLAUDE.md — STT-API

Guidance for working in this repo. `README.md` is the usage doc; this file is the
list of things that will bite you or silently corrupt a measurement.
Module-scoped notes live beside their code — see
[`stt_api/evaluation/CLAUDE.md`](stt_api/evaluation/CLAUDE.md).

## Layout

| path | role |
|---|---|
| `stt_api/main.py` | FastAPI app: transcription, streaming WS, force alignment, speaker vectors |
| `stt_api/diarization.py` | TitaNet Large speaker embeddings + online diarization |
| `stt_api/clustering_torch.py` | streaming k-means / BIRCH over speaker vectors |
| `stt_api/evaluation/` | convention-normalized WER/CER scoring (**has its own CLAUDE.md**) |
| `stt_api/livekit_plugin/turn_detector/` | **text** end-of-turn detector, vLLM-backed |
| `stt_api/livekit_plugin/semantic_vad/` | **audio** end-of-turn detector (semantic VAD) |
| `stt_api/livekit_plugin/noise_cancellation/` | GTCRN filter + `benchmark/` harness |
| `stt_api/livekit_plugin/dummy/` | no-op STT/LLM/TTS, useful for agent tests |

## Repo-wide traps

### The package is imported as `app`, not `stt_api`

`docker-compose.yaml` mounts `./stt_api:/app/app`, so every absolute import
inside the package says `from app.diarization import ...`. On a laptop that name
does not exist, which is why `tests/test_diarization.py` and `tests/test_main.py`
fail with `ModuleNotFoundError: No module named 'app'` / `'fastapi'` — those
failures are **pre-existing environment gaps, not regressions**. Check before
blaming a change. To import the package locally:

```python
import stt_api, sys
sys.modules.setdefault("app", stt_api)
```

### `stt_api/diarization.py` needs CUDA at *import* time

It builds `cuda.Stream()` at module scope, so the module cannot be imported at
all on a machine without CUDA. `stt_api/main.py` additionally runs
`initialize_models()` at import, spawning process pools. Both have to be stubbed
to test anything off a GPU box — see `tests/test_speaker_vector_api.py` for the
pattern.

### The speaker model lives in a separate spawn process

`get_diarization_executor()` is a 1-worker `ProcessPoolExecutor` with a spawn
context (CUDA needs it). Anything returned from that worker must be **picklable**
— CUDA tensors are not. That is the whole reason `embed_chunks_for_api` exists
next to `extract_embeddings_batched`: it moves to CPU and flattens to lists on
the worker side.

Speaker vectors are **unnormalized** by design; `clustering_torch` calls
`cosine_similarity` explicitly. If you add a consumer, either normalize both
sides or neither — mixing them silently yields the wrong distance.

## LiveKit plugins

There are **two unrelated end-of-turn interfaces** and picking the wrong one is
the main way this goes wrong:

| | `turn_detector/` | `semantic_vad/` |
|---|---|---|
| input | transcript text | streaming 16 kHz audio |
| LiveKit API | `livekit-plugins-turn-detector` | `livekit.agents.inference.eot` |
| answers | only after the STT emits words | from audio already buffered |

On the production stack the transcript arrives a median **1.5 s** after the
speaker stops, so no text detector can beat that. That is the reason
`semantic_vad/` exists, not a preference.

### `turn_detector/`

- **`--served-model-name livekit/turn-detector` is mandatory.** The plugin
  hardcodes that string in the request body; serving under the real model name
  404s every request.
- **`_normalize_text` strips punctuation**, and the `Semantic-VAD` pipeline
  benchmark shows this drives `p(<|im_end|>)` to ≈0 on every turn — "the shipped
  normalization turns the detector into a timer". Treat it as a known defect,
  not a neutral preprocessing step.
- **It fails open**: `_extract_eot_probability` returns `1.0` when it cannot
  parse a response, i.e. "turn is over". A broken endpoint therefore looks like
  an agent that interrupts constantly, not like an outage.
- `--api-key` on the vLLM server breaks it silently — the plugin sends no
  `Authorization` header, and vLLM exempts `/health`, so the job stays green.

### `semantic_vad/`

- Imports **private** LiveKit internals (`livekit.agents.inference.eot.base`),
  verified against 1.7.x. An upgrade can move them.
- **It fails closed** (`p = 0.0`, hold) — deliberately the opposite of
  `turn_detector/`. Failing toward silence is safer for a voice agent.
- Self-hosting needs **no protobuf websocket server**: the transport is a
  seven-method Protocol implemented in-process.

## Measurement invariants

These are the ways a benchmark in this repo can look fine and be wrong.

### Pooled WER is unbounded under insertions

One ASR repetition loop — 407 words against a 20-word reference — contributed
~73 % of all errors across a 100-item corpus and turned a model whose *median*
per-item WER was 0 % into an apparent 70 % catastrophe. `asr.is_degenerate`
excludes those and counts them separately. **Always read pooled WER next to the
median**; a large gap means a few items are driving it.

### Cost numbers need a quiet machine

Quality metrics are deterministic and reproduce under any load. Per-frame cost
does not: GTCRN measured p99 1.09 ms idle and 9.00 ms on the same audio while
another benchmark used the other cores. Run cost comparisons pinned and alone;
relative ordering within one run survives contention, absolute budget does not.

### Waveform metrics do not apply to generative models

PESQ/STOI/SI-SDR compare waveforms sample by sample. A vocoder emits a *different*
waveform that sounds like the same speech, so they rank it last however good it
sounds. Judge that class on DNSMOS **and WER** — and never on MOS alone: DNSMOS
and WER were *inversely* correlated across those models (Spearman ρ = +0.90),
so picking on perceptual quality selects almost exactly the wrong model.

### Alignment before any reference metric

Every enhancer imposes a different algorithmic delay, and a 32 ms misalignment
moves wideband PESQ by about a point — more than the gap between any two models.
`benchmark/audio.py:estimate_delay` handles it. A wrong delay does not error, it
just reranks the table.

## Conventions

- Ruff (`v0.1.8`, pre-commit) — `ruff check` and `ruff format`.
- Optional extras are real boundaries: `server`, `livekit`, `evaluation`,
  `benchmark`, `dev`. The core install is `aiohttp` + `transformers` only, and
  `stt_api.evaluation` is standard-library apart from one lazy import. Do not
  add a top-level dependency to a module that is meant to run inside an agent.
- Tests that need weights or a GPU should skip, not fail — see the
  `importorskip` / `RUN_LIVEKIT_INTEGRATION` patterns in `tests/`.
