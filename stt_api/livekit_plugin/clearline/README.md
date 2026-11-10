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

Tested (`tests/test_clearline.py`, 63 tests) on `livekit-agents` 1.2.18, 1.3.11 and
1.8.9, the range the `scicom-livekit-plugin` extra allows.

The weights live in public Hugging Face repositories, `Scicom-intl/Clearline-VAD` and
`Scicom-intl/Clearline-NC-CPU` (named `Clearline-NC` until 2026-10-30; the old id
redirects), and are fetched on first use **at a pinned revision** (`hub.VAD_REVISION`,
`hub.NC_REVISION`); no token is needed. A local copy overrides the download:
`ClearlineNC(model_path=...)` or `CLEARLINE_NC_ONNX_PATH` (for the VAD,
`ClearlineVAD.load(onnx_file_path=...)` or `CLEARLINE_VAD_ONNX_PATH`).

## `ClearlineVAD`

A 55k-parameter GRU on log-mel features: one probability every 16 ms, 61 us of one
core per 32 ms window against silero v6's 101 us (1.7x less), measured pinned on a
Xeon server through ONNX Runtime with one thread. `load()` takes silero's options
(`min_speech_duration`, `min_silence_duration`, `prefix_padding_duration`,
`max_buffered_speech`, `activation_threshold`, `deactivation_threshold`) plus:

| option | default | what |
|---|---|---|
| `window` | 512 | samples per call at 16 kHz: 512 (32 ms, silero's cadence) or 256 (16 ms updates, same weights) |
| `activation_threshold` | 0.825 | **not silero's 0.5**; 0.875 with `window=256`; 0.825 for the distilled variant (0.75 at 256) (see below) |
| `smoothing_alpha` | 0.35 | LiveKit's ExpFilter alpha (0.5 for the distilled variant at 256 samples); **0 turns it off, 1 is refused** (see below) |
| `variant` | `"release"` | `"distilled"`: the better frame classifier (96 units; ahead of silero through all five telephone codecs and in noisy rooms), ends turns a little later |
| `inline` | True | run the model on the event loop: the call costs less than the thread-pool hop |
| `legacy_stream` | False | upstream's stream loop for every stream (LiveKit's resampler once per frame, one inference per stream); see "The stream" |
| `onnx_file_path`, `revision` | | a local graph, or another Hugging Face revision |

Against silero v6 on thirteen test sets (meetings, broadcast, films, earnings calls,
private telephone calls, noisy read speech in 40 languages), each model at the
point it ships with, chosen on the development sets (AMI dev and VoxConverse dev,
wideband and through a phone codec), and silero v6 at its own best point on the same
four (threshold 0.70), replayed through LiveKit's own state machine:

| | silero v6 | Clearline-VAD | Clearline-VAD, distilled |
|---|---|---|---|
| CPU per 32 ms (median) | 103 us | **60 us** | 83 us |
| mean frame ROC-AUC, 13 sets | 0.912 | 0.920 (higher on 8) | **0.921 (higher on 10)** |
| fresh turns detected | 0.801 | **0.835** | 0.801 |
| median onset delay, mean over the sets | **247 ms** | 261 ms | 465 ms (220 without CHiME-6) |
| turn ends on time | 0.718 | **0.757** | 0.728 |
| false triggers per hour of non-speech, 10 sets | 29.6 | 35.1 | **21.4** |
| the same without the two private telephone sets | 28.1 | 23.6 | **14.3** |
| on the telephone (AMI dev through codecs): turns / ends / false per hour | 0.877 / 0.750 / **17.4** | **0.944 / 0.866** / 34.3 | 0.929 / 0.825 / 22.7 |

Both models at this revision were fine-tuned with half their training examples heard
through Clearline-NC first, because a LiveKit agent runs its noise filter before its
VAD: behind a filter that removes the noise outright (Clearline-NC's quality mode,
GTCRN) the weights before them heard the filtered pauses as loud. Behind GTCRN, which
they never heard, they now rank frames better than those weights did (and the
distilled variant better than silero v6); with no filter they are level with them.
Far-field meetings behind a strong filter are the exception (AliMeeting).

The extra false triggers are on telephone legs, in the line's silences: background
sounds about 25 dB under the caller's own speech (91 and 71 per hour on the two
private telephone sets against silero's 45 and 26; on the other eight sets 23.6 per
hour against silero's 28.1). Behind `ClearlineNC` in its default mode they fall to 50
and 37 per hour; without it, raise `activation_threshold` to 0.9-0.95 on such lines.
On FLEURS through five telephone codecs the distilled variant ranks frames ahead of
silero on every codec (frame ROC-AUC 0.939 against 0.936) and the release a little
behind (0.924). The distilled variant misses distant talkers in far-field meetings:
its mean onset delay above is mostly CHiME-6's dinner parties, where it catches 6% of
turns, 2.7 s late (silero v6 15%, at 0.5 s). Use the release for meetings.

### The stream

Until 2026-11-11 the stream around the model was `livekit-plugins-silero`'s with the
model swapped, and most of what a call's VAD cost was that loop, not the model: 60 us
of a 303 us window at 24 kHz. Now pending samples sit in one numpy buffer, a window is
resampled in numpy (8, 24, 32 and 48 kHz input; any other rate, stereo, or
`inline=False` takes upstream's loop), and **streams of one VAD that are pushed the
same frame objects share one inference**. An `AgentSession` with a non-streaming STT
(ours: `WhisperSTT`) pushes every frame to its own VAD stream and to its
`StreamAdapter`'s, so upstream's loop ran the model twice per window. Each stream
keeps its own filter, state machine, speech buffer and events; at 16 kHz the events
are upstream's value for value, at the other rates the same turns within two windows.
`vad.windows_inferred` counts model calls.

| CPU for a call's VAD, us per 32 ms of audio | silero v6 | Clearline, upstream's loop | Clearline |
|---|---|---|---|
| one stream, 16 kHz room | 307 | 178 | **155** |
| one stream, 24 kHz room (LiveKit's default) | 459 | 303 | **168** |
| the agent's two streams, 16 kHz | 588 | 323 | **182** |
| the agent's two streams, 24 kHz | 887 | 569 | **197** |

Inside a `livekit-agents` 1.3.11 agent with the dummy STT, LLM and TTS (each call its
own process, 50 calls per VAD side by side) the VAD's part of a call falls from 3.0%
of a core to 1.6% (silero v6 4.3%; 1.8.9: 2.8% to 1.8%, silero 5.7%), and an
eight-core worker holds about 83 calls at LiveKit's 0.7 load limit, where it held
about 71 (silero v6 about 58).

## `ClearlineNC`

A band-gain and deep-filter enhancer on GTCRN's STFT (512-point sqrt-Hann, 256 hop,
32 ms delay at 16 kHz), with GTCRN's constructor (`enabled`, `model_path`,
`num_threads`) plus `mode` and `mix_back`. Through this repo's own benchmark
(`../noise_cancellation/benchmark/`, where it is registered as `clearline` and
`clearline_quality`), frame by frame through `rtc.AudioFrame`:

| | GTCRN | `mode="quality"` | `mode="asr"` (default) |
|---|---|---|---|
| VoiceBank+DEMAND PESQ / SI-SDR | 2.535 / 14.6 dB | **2.575 / 17.4 dB** | 2.135 / 10.5 dB |
| DNS 2020 synthetic PESQ / ESTOI | 2.293 / 0.872 | **2.389 / 0.886** | 1.697 / 0.827 |
| CPU per 20 ms frame at 16 kHz, one core (paired, quiet server) | 1.09 ms | 0.38 ms | **0.19 ms** |
| plugin CPU per stream at real-time pace, 8 streams, 16 kHz / 48 kHz room (% of a core) | 6.20 / 6.83 | 2.50 / 3.28 | **1.51 / 2.49** |
| Whisper large-v3 WER, all 824 VoiceBank utterances, wideband / phone path | 2.65% / 4.26% | 2.32% / 5.79% | **1.55% / 2.90%** |
| Whisper large-v3 WER, DNS 2020 synthetic (150 clips) | 5.30% | 6.48% | **5.02%** |

(Unprocessed: 1.45% / 2.95% on VoiceBank, 5.38% on DNS. Neither mode is trained on
VoiceBank, VCTK or DEMAND. Since 2026-10-30 the quality mode is the larger graph, 256
units with per-bin levels and gains, distilled from a GPU model that removes background
noise and keeps every voice, `Scicom-intl/Clearline-NC-GPU`; a frame of it costs about
a third of GTCRN's CPU, of the ASR mode about a sixth.) Every WER here is on the
benchmark's harness as fixed: a streaming filter's output is flushed and aligned
before Whisper hears it (`../noise_cancellation/benchmark/README.md`); before
2026-10-27 the harness cut each utterance's last 32 ms for every streaming filter.
On the harness's own 100-item protocol the ASR mode gives 1.65% wideband and 3.18%
through G.711, against 1.53% and 3.82% unprocessed and GTCRN's 2.42% and 4.07%.
The same 824 utterances through five telephone codecs at 8 kHz (* never in training),
Whisper's WER after each filter:

| | unprocessed | GTCRN | `mode="quality"` | `mode="asr"` |
|---|---|---|---|---|
| G.711 | 3.25% | 10.72% | 4.64% | **3.20%** |
| GSM | 7.41% | 15.60% | 9.10% | **6.96%** |
| Opus | 4.94% | 13.46% | 7.32% | **4.81%** |
| AMR-NB* | 4.83% | 13.66% | 6.99% | **5.24%** |
| G.723.1* | 7.72% | 16.62% | 9.89% | **7.78%** |

`mode="asr"` is the model trained to keep noise 20 dB down rather than remove it,
with 70% of the input mixed back (`mix_back`), chosen on development data: a robust
recogniser is hurt by a filter's artifacts more than by the noise it leaves, and at
that share Whisper's word error rate stays where the unprocessed input has it (over
eight test conditions, 1.00 times the unprocessed input's on average; GTCRN 2.17).
It is the default for agents. The price is quality: it leaves most of the noise in.
`mode="quality"` removes background noise for a human listener and keeps every voice
(a second talker loses about 1 dB, crowd babble about 25 dB); it costs Whisper more on
most lines.

```bash
python -m stt_api.livekit_plugin.noise_cancellation.benchmark --corpus voicebank \
    --models gtcrn,clearline,clearline_quality --limit 0
```

### The graphs and the buffers (2026-11-11)

The pinned revision ships both models as **fused graphs**: the same function with a
third of the operators (the plain graph spent two thirds of a call on bookkeeping for
calls of any length), equal to the plain graphs within one LSB through the plugin and
to the fourth decimal on the three test corpora. The plain graphs are in the
repository's `plain/` folder and the plugin reads either (the fused graph's first
input is named `ri`, the plain one's `spec`). The plugin also keeps its samples in
arrays written in place now. Together a 20 ms frame of the quality mode costs 0.35 of
GTCRN's CPU where it cost 0.44, and of the ASR mode 0.17 where it cost 0.27; the
output is the first version's, sample for sample.

And a fault is gone. At any room rate but 16 kHz the plugin kept every 16 kHz sample
it had already resampled and rebuilt that array on every frame, so a call's frames
cost more the longer it ran: at 48 kHz, 0.69 ms a frame in a call's first minute and
4.8 ms in its tenth (now 0.46 ms throughout). `GTCRN`, whose frame path this plugin
was adapted from, had the same fault (1.5 ms in the first minute, 2.9 ms in the
sixth) and is fixed with it. At 16 kHz, which the snippet above asks for, nothing
grew.

## Traps

- **The default threshold is 0.825, not silero's 0.5.** The models are calibrated
  differently; 0.825 (with the filter at 0.35) is the point chosen on the
  development sets for the pinned release at 512-sample windows, 0.875 for the
  256-sample window, and 0.825 with the filter at 0.35 (0.75 with 0.5 at 256) for
  the distilled variant
  (`hub.OPERATING_POINT`, one entry per graph). It moves with the weights: `hub.VAD_REVISION` and `hub.OPERATING_POINT`
  change together.
- **`smoothing_alpha` reads LiveKit's way**: `ExpFilter` keeps `alpha` of the
  previous value, so silero's 0.35 keeps 65% of each new probability and 1.0 would
  freeze the filter on the first window. `load()` refuses 1.0. `PulseVAD`'s option
  of the same name is documented the other way round: do not copy a value across.
- **A room that is not 16 kHz.** Before 2026-11-11 both noise filters slowed down
  through every call at 8, 24 or 48 kHz (above). Fixed; 16 kHz is still the cheapest
  rate for both plugins, since nothing is resampled.
- **Two streams share a model only on the same frame objects.** The VAD's shared
  inference keys on identity: a wrapper that copies frames before one of the two
  streams gets them gives each stream its own model again (correct, twice the cost).
- **One `ClearlineNC` per audio stream.** The recurrent state is per instance (the
  ONNX session is shared); the `lambda params: ClearlineNC()` factory is what
  LiveKit calls once per stream.
- **The download happens at load, not on the first frame.** A failed download raises
  `FileNotFoundError` from `load()` / `ClearlineNC()`, with what to do about it,
  rather than an agent that hears nothing or a filter that passes audio through.
