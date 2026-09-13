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
| `stt_api/nemo_speaker_vector.py` | the TitaNet model itself — **read the perf note below before touching `prep_batch`** |
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

### Two speaker-vector performance traps, both already paid for

`nemo_speaker_vector.py` was 6x slower than it needed to be, and in both cases the
GPU was innocent — it was doing 8 ms of work inside a 57 ms batch. Do not
reintroduce either:

- **`sequence_1d` padded through Python objects.** `.tolist()` on every input
  array, list concatenation, then `np.array` to re-parse — ~768k float objects
  per batch of 16, **50 ms**. There is now a vectorised fast path for ndarray
  input; the generic path is kept for anything else.
- **Pinned staging cost 40x the transfer it optimised.** H2D for a 16x48000 batch
  is 0.07 ms; copying *into* the page-locked buffer was 10.4 ms on this host.
  Pageable is 41x faster end to end. `SPEAKER_PIN_MEMORY=1` restores pinning —
  measure before assuming your host is the other kind.

Numbers and method: [`SPEAKER_VECTOR_BENCH.md`](SPEAKER_VECTOR_BENCH.md).

**Ragged batches shift vectors.** Clips of differing length are padded to the
longest, and the model sees the padding: cosine 0.963-1.0 against computing a
clip alone, versus 3e-05 for uniform lengths. This predates the optimisation
(padding output is bit-identical before and after) but it matters for an
embedding API — enroll and query the same way, or batch similar lengths.

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
- **`do_normalize` is per-checkpoint, not per-family.** smart-turn-v3 wants it on;
  the Scicom `semantic-vad-eot-whisper-*` models ship `do_normalize: false` in
  `eot_window.json`. On one tone the flag moves p(eot) from 0.39 to 0.53 — both
  plausible, nothing errors. Backends read the checkpoint's own config; keep it
  that way rather than hardcoding a default.
- Audio is **left**-padded into the 8 s window so the decision point stays at the
  end. Right-padding (the feature extractor's default) is out of distribution and
  scores confidently wrong.
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

### Testing a turn detector: assert the effect, not the call

Two ways to write a passing test for a LiveKit turn detector that proves nothing:

- **Asserting the detector was consulted.** `run_inference` being reached and a
  turn committing are both true while LiveKit ignores the verdict entirely.
- **Watching `user_input_transcribed`.** That fires when the STT returns, not
  when the turn commits, so a fast path and a slow path both read as +0.50 s.

Assert on `on_user_turn_completed`, and drive the session twice with a fixed
probability either side of the threshold. A working integration commits on
`min_delay` in one case and waits for `max_delay` in the other — currently a 2.4 s
difference. `tests/test_semantic_vad.py` does exactly this.

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
