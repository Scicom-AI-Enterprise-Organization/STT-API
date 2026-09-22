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
| `stt_api/livekit_plugin/pulse_vad/` | **frame** VAD (is anyone speaking) — silero drop-in + `benchmark/` |
| `stt_api/livekit_plugin/whisper_stt/` | Whisper STT client that stops silence interrupting the agent |
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

There are **three unrelated interfaces** here and picking the wrong one is the
main way this goes wrong. The first two answer *has the speaker finished*; the
third answers *is anyone speaking right now*, and a session needs one of each —
the VAD decides when to ask, the turn detector answers.

| | `turn_detector/` | `semantic_vad/` | `pulse_vad/` | `whisper_stt/` |
|---|---|---|---|---|
| input | transcript text | streaming 16 kHz audio | streaming 16 kHz audio | one VAD segment |
| LiveKit API | `livekit-plugins-turn-detector` | `livekit.agents.inference.eot` | `livekit.agents.vad.VAD` | `livekit.agents.stt.STT` |
| passed as | `turn_detection=` | `turn_detection=` | `vad=` | `stt=` |
| answers | only after the STT emits words | from audio already buffered | per 32 ms hop | what was said |

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

### `pulse_vad/`

- **p(speech) saturates at 0.711**, not 1.0 — 0.708 int8, 0.895 for the `81k`
  teacher. A threshold at or above the ceiling fires on **0 %** of windows: an
  agent that never hears anyone, with no exception and no log line. Carrying
  silero's `activation_threshold=0.8` across is the obvious way to hit it, so
  `load()` raises. `model.CEILINGS` is the source of truth and is asserted in
  `tests/test_pulse_vad.py`; re-measure it if a checkpoint is ever swapped.
- **The graph emits `[non_speech, speech]` logits, not a probability.** The exact
  mirror of the `semantic_vad` trap, where the sigmoid is already in the graph
  and applying a second one pinned every score to ~0.73.
- **int8 is not faster off-microcontroller** (0.019 ms vs fp32's 0.018 ms) and
  its QDQ file is twice the size. `precision="fp32"` is the default here even
  though upstream's `load_pulsevad` defaults to quantised.
- **The 200 ms window is not the update rate.** It slides over a 32 ms hop, so
  LiveKit's duration defaults still mean what they mean. At a 200 ms interval
  `min_speech_duration=0.05` would be satisfied by a single window and one
  spurious inference would open a turn.
- **The front end, not the model, is the cost — and cutting its arithmetic makes
  it slower.** 67 us of numpy against 17 us of ONNX, so silero's fully-fused
  graph beats this 2,118-parameter model per window. Only the FFT (16 us) is real
  compute; the rest is nine numpy calls whose cost is dispatch, not data —
  `std()` over 1,344 floats measures 4.94 us. The 200 ms window slides by 32 ms,
  so 85 % of the STFT is recomputed each hop, and removing that redundancy is the
  obvious fix — **it was tried and it is 15 % slower**, because caching the
  spectra takes more numpy calls than it saves FFTs. At these sizes the call
  count is the cost function. The real fix is folding the front end into the ONNX
  graph; with it free, PulseVAD costs 17 us against silero's 78 us.
- **Upstream leaves the first sample of every window un-pre-emphasised**
  (`pre[0] = x[0]`), and that sample feeds the window's own mean and variance. It
  is a spike of median 2x and up to 53x the rest of the window's std, so a
  streaming implementation must re-derive index 0 per window rather than
  pre-emphasise continuously — doing the obvious continuous thing moves p(speech)
  by up to 0.28.
- Weights are **vendored**, not taken from the `pulsevad` PyPI package — it pulls
  scipy and soundfile for a file reader this plugin never calls, and needs
  Python 3.11 while this repo supports 3.10. `frontend.py` is a bit-exact
  reimplementation, asserted against a transcription of upstream.

Numbers, method and the silero head-to-head:
[`PULSE_VAD_BENCH.md`](PULSE_VAD_BENCH.md). **The conclusion is "keep silero"** —
PulseVAD matches it on accuracy over 1,500 real turns but costs 1.5x the CPU and
commits 96 ms later. It wins only on footprint, and by less than it looks — 76 KB
against 2.22 MB, because the 64.4 KB mel filterbank sits outside the graph. Note
the VAD is 0.42 % of a core per stream either way, so this is never a lever on
CPU load.

### `whisper_stt/`

- **Whisper answers silence with `' '`, not `''`, and a space is truthy.**
  LiveKit only tests emptiness, so the space passes
  `stream_adapter.py:149` and `audio_recognition.py:1215`, reaches
  `on_final_transcript` (`:1217`), and that calls
  `_interrupt_by_audio_activity()` (`agent_activity.py:2531`) **with no text
  check of its own**. The agent gets cut off by silence and nothing logs an
  error. 12 % of production turns (180 of 1,503) return exactly `' '`. Verified
  by driving `AudioRecognition._on_stt_event`: `''` does not fire the hook,
  `' '` / `'  \n\t'` / `'.'` all do.
- **Return `text=""`, never `alternatives=[]`.** `audio_recognition.py:1207`
  indexes `alternatives[0]` with no length check, so an empty list swaps the
  interruption for an `IndexError`.
- **Filter on `.strip()` length, never on a character class.** Four turns in the
  production sample are real speech with no ASCII alphanumerics at all
  (`' ஹலோ மாயா!'`, `' 你好,我想现在付款…'`); an `[A-Za-z0-9]` test would silently
  discard Tamil and Chinese callers.
- **94 % of the useless STT calls can be skipped locally.** Speech turns sit at
  −18 dBFS median, non-speech at **−240 dBFS** — 169 of 180 are literally all
  zero samples, each exactly 0.30 s. An RMS floor of −50 dBFS skips 170/180 and
  loses 0/1,323 real turns. Also: 0.10 s of *real* speech transcribes to `' '`,
  so segments that short need never be sent.
- **Those all-zero segments are not silero's.** Fed 0.30 s of digital silence it
  peaks at p(speech) 0.0089 against a 0.5 threshold. Something upstream emits
  fixed 0.30 s zero buffers — a warmup or health probe fits the signature. Trace
  it rather than blaming the VAD.
- **The blank guard does NOT stop the interruption — the VAD does it first.**
  Measured in a real in-process `AgentSession` with the agent mid-TTS: a 0.35 s
  false positive cut the agent's speech **even when the STT returned `''`**,
  `_interrupt_source == "audio_activity"` in both arms. What the guard does buy
  is no chat bubble and no LLM context from silence (a blank never reaches
  `on_final_transcript`), plus ~94 % fewer STT calls.
- **`VADEvent.speech_duration` keeps accumulating through silero's 0.55 s
  silence hangover**, so a burst far shorter than LiveKit's 0.5 s interrupt
  threshold still clears it — 0.15 s and 0.35 s bursts both peak at **0.70 s**.
  Every VAD segment, however short, can interrupt by itself via
  `agent_activity.py:2447`. Do not assume a short segment is harmless; an
  earlier version of this file said sub-0.5 s segments could only interrupt
  through the STT path, and that was wrong.
- **The lever that gates *both* paths is `min_words`**, at
  `agent_activity.py:2325` — and it is inert at its default of **0**:

      AgentSession(..., min_interruption_words=1)                      # 1.3.x-1.7.x
      AgentSession(..., turn_handling={"interruption": {"min_words": 1}})  # 1.8+

  **`turn_handling` does not exist before 1.8 and raises `TypeError` there** —
  `ucc_tm-voice-assist` pins `livekit-agents~=1.3` and resolves to **1.3.11**,
  so it needs the first form. Every mechanism above is present in both versions;
  only line numbers and this spelling differ (1.3.11: the blank guard is
  `audio_recognition.py:355`, the `min_words` gate `agent_activity.py:1174`, the
  VAD-alone interrupt `agent_activity.py:1243`). It is Unicode-safe: `''`, `' '` and `'.'` all split to 0 words,
  while `' ஹலோ மாயா!'` splits to 2 and `' 你好,我想现在付款'` to 8. Measured end to
  end on 1.3.11 with the agent mid-TTS: unfiltered STT fires a bubble carrying
  `' '` and cuts the TTS; `WhisperSTT` fires **no** bubble; `min_words=1` stops
  the cut — and **real speech still barges in**, which is the control that makes
  the setting usable rather than just interruption switched off. The plugin
  removes the bubble, `min_words` removes the interruption, and you need both.
  `whisper_stt/example_agent.py` wires it and picks the right spelling for the
  installed version at runtime.

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

### A VAD benchmark measures the corpus unless you stop it

Three corrections, each of which changed a headline number in
`pulse_vad/benchmark/`:

- **Score each model at its own threshold.** PulseVAD saturates at 0.711 and
  silero at ~1.0; scoring both at 0.5 compares silero's mid-range against
  PulseVAD's 70th percentile and reports the gap as accuracy.
- **Exclude the hangover from the false-positive region.**
  `min_silence_duration` keeps a VAD speaking for 0.55 s after the last word *by
  design*. Counting it made both detectors look ~20 % false-positive — a
  measurement of the setting, not the model.
- **Pair timing within an item.** Production turns are cut by the upstream stack,
  not trimmed to the first phoneme, so both detectors wait through the same
  lead-in: over 250 real turns both reported a median onset of *exactly* 644 ms.
  Differencing within an item cancels it.

Plain F1 is nearly flat across thresholds here because the speech-bearing body
dominates it, so it picks an operating point almost at random. Optimise
`detect_rate - false_fire_rate` instead — what the agent actually experiences.

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
