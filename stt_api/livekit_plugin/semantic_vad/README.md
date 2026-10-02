# Semantic VAD for LiveKit Agents

End-of-turn decided from the **waveform** — prosody, final-syllable lengthening,
intonation — rather than from a transcript.

## Why not the text turn detector

[`../turn_detector/`](../turn_detector/) runs an LLM over the transcript. It
works, but it cannot answer until the STT has produced words, and on the
production stack that transcript arrives a **median 1.5 s (p90 4.5 s)** after the
speaker stops. No text detector can beat that, because the text does not exist
yet. Measured on Malay dialect speech, neither LiveKit's multilingual ONNX
detector nor the Qwen3 text detector improved on plain VAD endpointing as
shipped.

An audio-native model answers from audio the agent already has, in tens of
milliseconds.

They are complementary, not exclusive: text sees semantics an acoustic model
cannot ("...and then he said" is clearly unfinished), audio sees prosody a
transcript throws away. Running both and taking the earlier confident answer is a
reasonable design; this plugin gives you the audio half.

## How it plugs into LiveKit

LiveKit ships **two unrelated EoT interfaces**, and picking the wrong one is the
main way this goes wrong:

| | `livekit-plugins-turn-detector` | `livekit.agents.inference.eot` |
|---|---|---|
| input | transcript text | **streaming 16 kHz PCM** |
| entry | `EOUModelBase.predict_end_of_turn(chat_ctx)` | `TurnDetector(...).stream()` |
| transports | in-process ONNX | `v1` cloud websocket / `v1-mini` local ctypes |

Only the second can carry an audio model. Its `v1` transport streams audio to
LiveKit's gateway as **protobuf over a websocket**, which is why self-hosting is
usually described as "implement their websocket server".

**You don't have to.** The seam between the stream engine and its transport is
`_StreamingTurnDetectionTransport`, a seven-method Protocol:

```python
session_id, run(), run_inference(request_id), push_frame(frame), flush(), attach(stream), detach()
```

Implement that and everything above it stays stock — audio ingress, resampling to
16 kHz, request bookkeeping, metrics. `detector.py` implements it in-process, so
the model can be a local ONNX session or a plain HTTP call to your own GPU. No
protobuf, no websocket server.

Two numbers inherited from LiveKit that constrain any backend:

- **`DEFAULT_PREDICTION_TIMEOUT = 1.0 s`.** Miss it and the request is abandoned;
  with `local_fallback=True` the session *stickily* degrades to LiveKit's mini
  model and never comes back. This plugin defaults `local_fallback=False` so a
  slow backend shows up as a slow backend, not as a silent model swap.
- **The stock local buffer is 1.2 s** (`_CLIENT_BUFFER_SECONDS`) — right for the
  mini model, far too short for a model reasoning over an utterance. Here the
  buffer is sized from the backend's own `window_seconds` instead.

## Usage

```python
from stt_api.livekit_plugin.semantic_vad import SemanticVAD, ScicomEoT

session = AgentSession(
    stt=..., llm=..., tts=...,
    vad=ctx.proc.userdata["vad"],          # still required
    turn_detection=SemanticVAD(backend=ScicomEoT("base")),
)
```

`ScicomEoT("tiny" | "base" | "small")` is the in-house family, trained on
Malaysian call-centre telephony. `SmartTurnV3()` is the open pipecat model —
same architecture, 23 languages, no Malay.

The VAD is not optional and not redundant: LiveKit's VAD decides *when* to ask
(it only fires `inference_start` after ~200 ms of silence), and this model
answers. Nothing here polls mid-word, which matters because a semantic model
asked mid-syllable has nothing useful to say.

Point it at your own GPU instead:

```python
from stt_api.livekit_plugin.semantic_vad import SemanticVAD, RemoteEoT

SemanticVAD(backend=RemoteEoT("http://gpu-host:8080/eot", window_seconds=8.0))
```

`RemoteEoT` posts `{"audio": "<base64 PCM s16le>", "sample_rate": 16000}` and
reads `{"probability": ...}`. Deliberately boring — once you own the transport
you own the wire, so there is no reason to reimplement LiveKit's protobuf.

## Verified end to end

A real `AgentSession` (silero VAD + STT + this detector, audio paced at 1x wall
clock), driven twice with a fixed probability either side of the threshold:

| p(eot) | threshold | turn commits | path |
|---:|---:|---:|---|
| 0.9 | 0.2 | **+0.51 s** after speech | fast (`min_delay` 0.5 s) |
| 0.05 | 0.5 | **+2.93 s** after speech | slow (`max_delay` 3.0 s) |

The probability moves the turn boundary by 2.4 s, which is the only thing that
proves the plugin does anything. `RUN_LIVEKIT_INTEGRATION=1 pytest
tests/test_semantic_vad.py` runs it.

That assertion is deliberately about the *effect*, not the call. An earlier
version of the test checked only that `run_inference` was reached and a turn
committed — it passed while the verdict was being ignored. It also watched
`user_input_transcribed`, which fires when the STT returns rather than when the
turn commits, so both the fast and slow cases looked identical at +0.50 s. If you
extend these tests, assert on `on_user_turn_completed`.

## Which model

Everything below is audio-native unless marked otherwise.

| model | params | licence | languages | streaming | notes |
|---|---|---|---|---|---|
| **`Scicom-intl/semantic-vad-eot-whisper-{tiny,base,small}`** | 8 M / 20 M / 88 M | Apache-2.0 | **ms**, en | windowed 8 s | **implemented** as `ScicomEoT`. Trained on real Malaysian call-centre telephony. Re-released 2026-10-03 |
| **Scicom VAD**, **Scicom VAD Telephony** | 635 M | private | **ms**, en | windowed 8 s | GPU only, call it with `RemoteEoT`. Re-released 2026-10-02 |
| **`pipecat-ai/smart-turn-v3`** | 8 M | BSD-2 | 23 (no `ms`) | windowed 8 s | **implemented**. Whisper-Tiny encoder + linear head, 8 MB int8 ONNX |
| **`anyreach-ai/dualturn-endpointing`** | ~1.5 M on frozen Mimi | Apache-2.0 | en | **true streaming, 12.5 Hz** | dual-channel (user + agent); explicit recurrent state |
| `fixie-ai/turntaking-multilingual-llama8b-2a` | 8 B | **none stated** | multilingual | no | Ultravox-family; licence is a blocker |
| Qwen2-Audio EoT (in-house, `Semantic-VAD` repo) | 7 B backbone + 1 M head | Apache-2.0 backbone | trained on 19 configs incl. `ms_*`; **published eval is `en` only** | no | superseded by the Whisper family above. Not on the Hub |
| `TEN-framework/TEN_Turn_Detection` | ~7 B | Apache-2.0 | multi | no | **text**, not audio |
| `KE-Team/KE-SemanticVAD` | 0.5 B | Apache-2.0 | zh/en | no | **text**; also classifies backchannel vs interrupt |

### Scores

From the model cards of the 2026-10-02 and 2026-10-03 releases, scored with
[LiveKit's eot-bench](https://github.com/livekit/eot-bench) on two sets:

- **Telephony**: the test split of Scicom's private Malaysian call-centre set.
  English, Malay and both, 8 kHz. No call in it is in the training data.
- **eot-bench-data**: LiveKit's own 14-language set (`livekit/eot-bench-data`,
  validation), on the same spans as LiveKit's published runs.

| model | telephony cut-offs @ 300 ms | telephony latency @ 5 % | telephony AUC | eot-bench-data cut-offs @ 300 ms | eot-bench-data latency @ 5 % | eot-bench-data AUC |
|---|---:|---:|---:|---:|---:|---:|
| Scicom VAD Telephony | **39.6 %** | **1,663 ms** | **0.871** | 17.0 % | 759 ms | 0.929 |
| Scicom VAD | 40.7 % | 1,779 ms | 0.861 | 15.8 % | 718 ms | 0.937 |
| semantic-vad whisper-small, int8 | 43.4 % | 1,803 ms | 0.850 | 20.1 % | 801 ms | 0.915 |
| semantic-vad whisper-base, int8 | 44.2 % | 1,803 ms | 0.847 | 26.5 % | 861 ms | 0.879 |
| semantic-vad whisper-tiny, int8 | 44.7 % | 1,836 ms | 0.841 | 28.4 % | 896 ms | 0.867 |
| LiveKit Turn Detector v1 (cloud) | not run | | | **14.8 %** | **688 ms** | **0.941** |
| LiveKit Turn Detector v1-mini, audio | 61.6 % | 2,043 ms | 0.748 | 29.9 % | 924 ms | 0.840 |
| smart-turn v3.2, CPU int8 | 69.6 % | 2,274 ms | 0.670 | 39.0 % | 914 ms | 0.789 |
| ultraVAD | not run | | | 38.1 % | 911 ms | 0.797 |
| silence timer only | 77.6 % | 2,020 ms | | 54.9 % | 1,100 ms | |

- **Cut-offs @ 300 ms**: share of mid-turn pauses the agent interrupts when it
  may wait 300 ms on average after a finished turn. Lower is better.
- **Latency @ 5 %**: mean wait after a finished turn when at most 5 % of pauses
  may be interrupted. Lower is better.
- **AUC** is taken 0.2 s into each pause.
- The tiny, base and small rows are the int8 ONNX files `ScicomEoT` loads by
  default.

Against the 2026-09-07 release (int8 for the CPU sizes):

| model | telephony cut-offs | eot-bench-data cut-offs |
|---|---:|---:|
| tiny | 6.0 points fewer | 4.8 points fewer |
| base | 1.0 point more | 1.8 points fewer |
| small | 1.3 points more | 3.5 points fewer |
| Scicom VAD | 1.3 points fewer | 9.3 points fewer |

The old weights are kept on the `release-2026-09-07` branch of each repo.
`ScicomEoT` always loads `main`. If your traffic is telephony only, the old
`base` and `small` are still slightly better there.

**Picking a size.** On telephony the three CPU sizes are within 1.3 points of
each other. On eot-bench-data they are not: small is at 20.1 %, tiny at 28.4 %.
Per prediction on one CPU thread, int8, at export: tiny 28.5 ms, base 56 ms,
small 185 ms. All fit LiveKit's 1 s budget, but small costs 6.5x tiny and shares
a core with the STT and VAD.

- Malaysian telephony on CPU: `tiny`. Close to `small` at about a sixth of the cost.
- Wideband audio or other languages on CPU: `small`, or `base` if CPU is tight.
- A GPU available: Scicom VAD Telephony for phone calls, Scicom VAD for wideband
  audio and LiveKit's 14 languages. About 21 ms per request on an H20, and about 120
  predictions/s per GPU at p90 0.2 s, roughly 800 concurrent calls.

**int8 by default.** It is 2x (tiny) to 3x (small) faster than fp32 at export.
The mean output shift against PyTorch is 0.016 to 0.023, the maximum 0.047
(tiny) to 0.12 (small). On the benchmarks above, int8 and fp32 cut-off rates differ
by at most 1.5 points. If you calibrate a threshold near a decision boundary, pass
`quantized=False` and measure fp32 too.

### Measured here, on the 2026-09-07 release

12 VoiceBank utterances, complete versus truncated mid-utterance, one CPU thread.
This ran on the weights before the re-release and has not been re-run.

| backend | window / normalise | complete | truncated | separation | correct order | ms/call |
|---|---|---:|---:|---:|---:|---:|
| smart-turn-v3 | 8 s / on | 0.970 | 0.656 | **+0.314** | 10/12 | 35.5 |
| scicom tiny | 8 s / off | 0.864 | 0.801 | +0.063 | 9/12 | **24.4** |
| scicom base | 8 s / off | 0.855 | 0.757 | +0.099 | 10/12 | 42.5 |
| scicom small | 8 s / off | 0.851 | 0.663 | +0.188 | 11/12 | 145.2 |

**Do not read this as a ranking.** The corpus is English read speech. That is
inside smart-turn-v3's 23 languages and outside what the Scicom models were
trained on, Malaysian call-centre telephony where a turn ends because the other
party took the floor. The table only shows that the integration is wired
correctly and that every backend moves in the right direction.

It is still internally consistent. Separation orders small > base > tiny, the
same order as the publishers' AUCs. Wrong preprocessing would break that order.

### DualTurn is worth a look before scaling up

It is the only candidate here that streams rather than windows.
`stream_tick.onnx` takes an 80 ms chunk plus full recurrent state (KV cache,
transformer history, LSTM `h`/`c`) and returns updated state with `eot`, `vad`
and `fvad`. So it emits a turn-end probability **every 80 ms** instead of only
when the VAD asks. It also reads **both channels**, and hearing the agent's own
speech is what makes barge-in and overlap calls reliable. All of that for ~1.5 M
trainable parameters on a frozen Mimi encoder.

The limits are just as clear. English only, 24 kHz, and the dual-channel design
needs the agent's audio wired in, which is a bigger change than swapping a
detector. Not implemented here.

## Serving a large model

Nothing in this design forces the model to be small — `RemoteEoT` exists for the
7-8 B case. What it does force is the **1.0 s budget**, and that is where a
naive server fails: one forward pass of a 7 B audio LLM is fine, but per-request
scheduling under concurrent calls is not.

The shape that works is a FastAPI service with **dynamic batching** — a short
collection window (5–10 ms) that gathers whatever requests arrived, runs one
padded batch, and scatters the results. EoT requests are ideal for it: they are
bursty (one per pause, across all concurrent sessions), uniformly shaped (a fixed
audio window), and single-pass (no decode loop to schedule around). Budget
roughly: collection window + batch forward + transport, against 1.0 s.

Three things to hold onto if you build it:

- **Return a raw probability, not a decision.** Thresholding belongs on the
  client, per language, calibrated on your own eval set.
- **Cap the window server-side.** The client streams continuously; the server
  decides how much context the model sees. Match training.
- **Watch the tail, not the mean.** The 1 s timeout is per request, so p99 is
  what determines whether sessions silently degrade — the same argument the
  noise-cancellation benchmark makes about p99 over RTF.

## Calibration

`unlikely_threshold` is the bar for committing a turn — higher waits longer and
interrupts less. The default here is inherited from LiveKit's mini model and is
**not** tuned for whichever backend you plug in; calibrate on your own audio.

The `Semantic-VAD` repo's pipeline benchmark is the cautionary tale: a detector
that *ranked* turns well (AUC 0.91) was useless in production because its
probabilities sat at 1e-5–1e-4 against a 0.5 threshold, so every turn took the
slow timeout path. Ranking quality and threshold calibration are separate
problems and both have to be right.

## Failure behaviour

A backend that raises is treated as **hold** (`p = 0.0`), so the agent waits for
the endpointing timeout rather than interrupting. That is the opposite of the
text plugin in `../turn_detector/`, which returns `1.0` on a parse failure and
interrupts. Failing toward silence is the safer direction for a voice agent, and
the asymmetry is deliberate.

## Note on private API

`detector.py` imports from `livekit.agents.inference.eot.base`, which is not
public API. Verified against **livekit-agents 1.7.x**. A LiveKit upgrade can move
it; the import is wrapped so the failure names the cause instead of surfacing as
an `AttributeError` deep in a call stack.
