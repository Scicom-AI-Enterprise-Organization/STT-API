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
from stt_api.livekit_plugin.semantic_vad import SemanticVAD, SmartTurnV3

session = AgentSession(
    stt=..., llm=..., tts=...,
    vad=ctx.proc.userdata["vad"],          # still required
    turn_detection=SemanticVAD(backend=SmartTurnV3()),
)
```

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

## Which model

Everything below is audio-native unless marked otherwise.

| model | params | licence | languages | streaming | notes |
|---|---|---|---|---|---|
| **`pipecat-ai/smart-turn-v3`** | 8 M | BSD-2 | 23 (no `ms`) | windowed 8 s | **implemented.** Whisper-Tiny encoder + linear head, 8 MB int8 ONNX |
| **`anyreach-ai/dualturn-endpointing`** | ~1.5 M on frozen Mimi | Apache-2.0 | en | **true streaming, 12.5 Hz** | dual-channel (user + agent); explicit recurrent state |
| `fixie-ai/turntaking-multilingual-llama8b-2a` | 8 B | **none stated** | multilingual | no | Ultravox-family; licence is a blocker |
| Qwen2-Audio EoT (in-house, `Semantic-VAD` repo) | 7 B backbone + 1 M head | Apache-2.0 backbone | trained on 19 configs incl. `ms_*`; **published eval is `en` only** | no | not on the Hub — a local checkpoint you would have to serve |
| `TEN-framework/TEN_Turn_Detection` | ~7 B | Apache-2.0 | multi | no | **text**, not audio |
| `KE-Team/KE-SemanticVAD` | 0.5 B | Apache-2.0 | zh/en | no | **text**; also classifies backchannel vs interrupt |

### Measured here — smart-turn-v3, 12 VoiceBank utterances

| input | p(complete) |
|---|---|
| complete utterance | 0.970 |
| truncated mid-utterance | 0.656 |
| silence | 0.987 |

Correct ordering on 10/12, **36 ms per call on one CPU thread**. Comfortable
inside the 1 s budget with room for the STT and VAD sharing the core.

**Malay is not in smart-turn-v3's 23 languages** (Indonesian is, and is a useful
prior). Benchmark before trusting it on `ms`.

That gap is the argument for the in-house Qwen2-Audio model — but state it
carefully. Its training data covers Malay (`ms_dialects`, `ms_imda`,
`ms_malaysian`, `ms_parliament`, `ms_science_english`), yet the published
`eot-v6` evaluation is **English only**, so Malay capability is available rather
than demonstrated. Its measured lead is also distribution-specific: 0.923 AUC vs
LiveKit v1's 0.671 on its own test split, but 0.808 vs 0.939 on
`livekit/eot-bench-data` — the two datasets encode the hold/eot decision
differently (87 % eot spans cut at a fixed 0.5 s versus 33.7 % cut at 1.5 s).
Telephony augmentation is implemented but not yet trained, so there is no
channel-robustness claim either.

### DualTurn is worth a look before scaling up

It is the only candidate here that is *genuinely* streaming rather than windowed:
`stream_tick.onnx` takes an 80 ms chunk plus full recurrent state (KV cache,
transformer history, LSTM `h`/`c`) and returns updated state alongside `eot`,
`vad` and `fvad`. So it emits a turn-end probability **every 80 ms** instead of
only when the VAD asks, and it reads **both channels** — hearing the agent's own
speech is what makes barge-in and overlap judgements reliable. All for ~1.5 M
trainable parameters on a frozen Mimi encoder.

Its limits are equally clear: English only, 24 kHz, and the dual-channel design
wants the agent's audio wired in, which is a bigger change than swapping a
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
