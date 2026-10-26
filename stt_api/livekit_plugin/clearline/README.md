# Clearline

Voice activity detection and noise cancellation for LiveKit agents, on one CPU
core. Two plugins with the interfaces of the two incumbents:

| | replaces | LiveKit interface | passed as |
|---|---|---|---|
| `ClearlineVAD` | `silero.VAD` | `agents.vad.VAD` | `vad=` |
| `ClearlineNC` | `GTCRN` (`../noise_cancellation/`) | `rtc.FrameProcessor[rtc.AudioFrame]` | `noise_cancellation=` |

Both were trained in the Clearline project (VAD-Enhance-CPU; the preprint has the
protocol, the data and every number below), on wideband audio and on a simulated
telephone line with real codecs.

## Use

```python
from livekit.agents import AgentSession
from livekit.agents.voice import room_io

from stt_api.livekit_plugin import ClearlineNC, ClearlineVAD


def prewarm(proc):
    proc.userdata["vad"] = ClearlineVAD.load()     # blocking; downloads the pinned weights once


session = AgentSession(vad=ctx.proc.userdata["vad"], stt=..., llm=..., tts=...)

await session.start(
    agent=MyAgent(),
    room=ctx.room,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            sample_rate=16000,                                     # the models' rate: no resampling
            noise_cancellation=lambda params: ClearlineNC(),       # one instance per stream
        ),
    ),
)
```

Tested (`tests/test_clearline.py`, 47 tests) on `livekit-agents` 1.2.18, 1.3.11 and
1.8.9, the range the `scicom-livekit-plugin` extra allows.

The weights live in Hugging Face repositories, `Scicom-intl/Clearline-VAD` (public)
and `Scicom-intl/Clearline-NC` (private), and are fetched on first use **at a pinned
revision** (`hub.VAD_REVISION`, `hub.NC_REVISION`). The noise filter needs an
`HF_TOKEN` that can read its repository, or a local copy: `ClearlineNC(model_path=...)`
or `CLEARLINE_NC_ONNX_PATH` (for the VAD, `ClearlineVAD.load(onnx_file_path=...)` or
`CLEARLINE_VAD_ONNX_PATH`).

## `ClearlineVAD`

A 55k-parameter GRU on log-mel features: one probability every 16 ms, 61 us of one
core per 32 ms window against silero v6's 101 us (1.7x less), measured pinned on a
Xeon server through ONNX Runtime with one thread. `load()` takes silero's options
(`min_speech_duration`, `min_silence_duration`, `prefix_padding_duration`,
`max_buffered_speech`, `activation_threshold`, `deactivation_threshold`) plus:

| option | default | what |
|---|---|---|
| `window` | 512 | samples per call at 16 kHz: 512 (32 ms, silero's cadence) or 256 (16 ms updates, same weights) |
| `activation_threshold` | 0.80 | **not silero's 0.5**; 0.85 with `window=256` or the distilled variant (see below) |
| `smoothing_alpha` | 0.35 | LiveKit's ExpFilter alpha; **0 turns it off, 1 is refused** (see below) |
| `variant` | `"release"` | `"distilled"`: the better frame classifier, ends turns a little later |
| `inline` | True | run the model on the event loop: the call costs less than the thread-pool hop |
| `onnx_file_path`, `revision` | | a local graph, or another Hugging Face revision |

Against silero v6 on thirteen test sets (meetings, broadcast, films, earnings calls,
private telephone calls, noisy read speech in 40 languages), each model at the
operating point chosen for it on development data and replayed through LiveKit's
own state machine:

| | silero v6 | Clearline-VAD | Clearline-VAD, distilled |
|---|---|---|---|
| CPU per 32 ms (median) | 101 us | **61 us** | 62 us |
| mean frame ROC-AUC, 13 sets | 0.912 | 0.921 (higher on 8) | **0.925** (higher on 9) |
| fresh turns detected | 0.782 | **0.855** | 0.835 |
| median onset delay | 326 ms | **233 ms** | 293 ms |
| turn ends on time | 0.686 | **0.776** | 0.721 |
| false triggers per hour of non-speech | **15.7** | 42.6 | 29.1 |
| on the telephone (AMI dev through codecs): turns / ends / false per hour | 0.800 / 0.639 / **7.7** | **0.947 / 0.864** / 37.2 | 0.852 / 0.758 / 67.7 |

Most of the extra false triggers are on telephone legs, in the line's silences:
background sounds about 25 dB under the caller's own speech (113 and 71 per hour
on the two private telephone sets against silero's 23 and 13; 30 per hour on the
other eight sets against silero's 15). On such lines raise `activation_threshold`
to 0.9-0.95. On telephone audio silero still ranks frames slightly better (frame
ROC-AUC 0.936 against 0.923 on FLEURS through five codecs); the distilled variant
is the wideband frame classifier.

## `ClearlineNC`

A band-gain and deep-filter enhancer on GTCRN's STFT (512-point sqrt-Hann, 256 hop,
32 ms delay at 16 kHz), with GTCRN's constructor (`enabled`, `model_path`,
`num_threads`) plus `mode` and `mix_back`. Through this repo's own benchmark
(`../noise_cancellation/benchmark/`, where it is registered as `clearline` and
`clearline_quality`), frame by frame through `rtc.AudioFrame`:

| | GTCRN | `mode="quality"` | `mode="asr"` (default) |
|---|---|---|---|
| VoiceBank+DEMAND PESQ / SI-SDR | 2.535 / 14.6 dB | 2.610 / **17.7 dB** | **2.612** / 16.3 dB |
| DNS 2020 synthetic PESQ / ESTOI | **2.293 / 0.872** | 2.239 / 0.871 | 2.190 / 0.864 |
| real-time factor; p99 per 20 ms frame | 0.064; 3.24 ms | 0.018; 0.69 ms | **0.018; 0.68 ms** |
| Whisper large-v3 WER, all 824 VoiceBank utterances, wideband / phone path | 2.55% / 4.28% | 2.48% / 6.00% | **2.02% / 4.07%** |

(Unprocessed: 1.45% / 2.95%. Neither mode is trained on VoiceBank, VCTK or DEMAND.)
The same 824 utterances through five telephone codecs at 8 kHz (* never in training),
Whisper's WER after each filter:

| | unprocessed | GTCRN | `mode="quality"` | `mode="asr"` |
|---|---|---|---|---|
| G.711 | 3.25% | 10.82% | 4.95% | **3.90%** |
| GSM | 6.80% | 17.79% | **10.97%** | 12.27% |
| Opus | 4.94% | 13.90% | 7.54% | **6.08%** |
| AMR-NB* | 4.96% | 13.12% | 7.77% | **6.10%** |
| G.723.1* | 7.72% | 19.53% | **12.08%** | 13.53% |

`mode="asr"` is the model trained to keep noise 20 dB down rather than remove it,
with 10% of the input mixed back (`mix_back`): what a recogniser downstream wants,
and the reason it is the default. `mode="quality"` removes all the noise it can, for
a human listener, and costs Whisper more on most lines.

```bash
python -m stt_api.livekit_plugin.noise_cancellation.benchmark --corpus voicebank \
    --models gtcrn,clearline,clearline_quality --limit 0
```

## Traps

- **The default threshold is 0.80, not silero's 0.5.** The models are calibrated
  differently; 0.80 (with the filter at 0.35) is the point chosen on the
  development sets for the pinned release at 512-sample windows, and 0.85 for the
  256-sample window and the distilled variant (`hub.OPERATING_POINT`, one entry per
  graph). It moves with the weights: `hub.VAD_REVISION` and `hub.OPERATING_POINT`
  change together.
- **`smoothing_alpha` reads LiveKit's way**: `ExpFilter` keeps `alpha` of the
  previous value, so silero's 0.35 keeps 65% of each new probability and 1.0 would
  freeze the filter on the first window. `load()` refuses 1.0. `PulseVAD`'s option
  of the same name is documented the other way round: do not copy a value across.
- **One `ClearlineNC` per audio stream.** The recurrent state is per instance (the
  ONNX session is shared); the `lambda params: ClearlineNC()` factory is what
  LiveKit calls once per stream.
- **The download happens at load, not on the first frame.** A missing token raises
  `FileNotFoundError` from `load()` / `ClearlineNC()`, with what to do about it,
  rather than an agent that hears nothing or a filter that passes audio through.
