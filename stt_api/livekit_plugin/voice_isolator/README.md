# Voice isolator plugin for LiveKit Agents

Keeps the caller and removes **every other voice** and the noise, inside the agent's
audio input, before the VAD, the turn detector and the STT hear the call.

`../noise_cancellation/` (GTCRN) removes noise. It cannot remove another person:
another voice *is* speech, and a noise suppressor cleans the noise off it, which makes
the agent's VAD fire on it **more** often, not less (measured below). This plugin is
for that case: a call centre floor behind the caller, a TV, a second person in the
car, someone talking on the other side of the room.

Two backends, like `../semantic_vad/`:

| | `LocalIsolator()` | `RemoteIsolator(url)` |
|---|---|---|
| model | `Scicom-intl/voice-isolator-cpu`, distilled for one CPU thread | `Scicom-intl/voice-isolator`, the release: M + a causal WavLM-Large, 555M |
| runs | in the agent process: onnxruntime + numpy, no torch, no GPU | on a GPU server (`Dockerfile`), one WebSocket per call |
| delay in an agent | 10 ms (the STFT look-ahead) | 100 ms median at 32 calls, 140 ms at 64 (LiveKit's 50 ms frames, the server at 40 ms steps); 50 ms with 20 ms frames and steps |
| cost | one thread per call on the agent's audio loop: 17% of a core in 50 ms frames, 33% in 20 ms (8-bit 12% and 22%) | the agent only queues the frame (0.4 ms p99); 64 calls on one GPU measured |

Both enroll the caller from the call itself and both give the same output whatever
frame size LiveKit hands over (below).

## Usage

```python
from livekit.agents.voice import room_io
from stt_api.livekit_plugin.voice_isolator import LocalIsolator, RemoteIsolator, VoiceIsolator

def prewarm(proc):
    proc.userdata["isolator"] = LocalIsolator()           # CPU: download and open once per process
    # proc.userdata["isolator"] = RemoteIsolator()        # GPU: VOICE_ISOLATOR_URL=ws://gpu-host:8765

await session.start(
    agent=MyAgent(),
    room=ctx.room,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            sample_rate=16000,                             # the model's rate: no resampling
            noise_cancellation=lambda params: VoiceIsolator(backend=ctx.proc.userdata["isolator"]),
        ),
    ),
)
```

`AudioInputOptions.noise_cancellation` accepts any `rtc.FrameProcessor[rtc.AudioFrame]`;
GTCRN uses the same hook. **One `VoiceIsolator` per audio stream** (the selector makes
one per participant): the call's attention rings, LSTMs and enrollment, or its server
connection, are its own. The backend is shared.

Models:

- `LocalIsolator()` reads `Scicom-intl/voice-isolator-cpu` (private: an `HF_TOKEN` that
  can read it), or a local directory: `LocalIsolator("/models/voice-isolator-cpu")` or
  `VOICE_ISOLATOR_MODEL`. `quantized=True` takes the 8-bit export from the repository's
  `int8/`: about 1.4x cheaper per call (1.6x in 10 ms frames), but it agrees with the
  float model to about 50 dB, and with itself across frame sizes to about 40 dB, rather
  than to float rounding, so measure it on your calls before you switch.
- `RemoteIsolator(url)` or `VOICE_ISOLATOR_URL` points at the GPU server. Build and run it
  from this directory's `Dockerfile`; it serves the release with the model repository's
  own server (`python -m vi.server_mp --enroll-loud --chunk-hops 4`: 40 ms steps for
  LiveKit's 50 ms frames; with `AudioInputOptions(frame_size_ms=20)`, `--chunk-hops 2`
  halves the delay for fewer calls per GPU).

Only phone callers, where it matters most:

```python
noise_cancellation=lambda params: (
    VoiceIsolator(backend=isolator)
    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    else None
),
```

A voice you already have (a few seconds of the caller alone, 16 kHz float32) can be
locked in from the first frame: `LocalIsolator(enrollment=clip)` or
`RemoteIsolator(url, enrollment=clip)`.

## Small frames, full context

The model decides who the caller is from context: a speaker vector (the loudest 3 s of
the call heard so far, re-made every half second from 3 s in, fixed at 15.5 s),
attention over the last 10 s, and two LSTMs that have heard the whole call. LiveKit
hands the audio over in 50 ms frames, but all of that context is the call's **state**,
carried from frame to frame. A frame of 10, 20 or 50 ms gives the same output, and it is
the output of the model run offline over the whole recording, 10 ms later
(`tests/test_voice_isolator.py` holds the first; Voice-Isolator's export holds the
second, 124-134 dB against PyTorch on real calls, i.e. float rounding).

The one place context runs short is the start of a call: before 3 s there is no
speaker vector, and the model runs with its learned "no enrollment" vector. The live
enrollment counts from the stream's first frame and is fixed at 15.5 s, so a caller who
says nothing in that time (a long greeting from the agent, a hold) is enrolled on
whatever the line carried; pass `enrollment=` when you have a clip of the caller.

## What it does to the agent's VAD

LiveKit's own VAD (Silero, the ONNX file `livekit-plugins-silero` 1.8.0 ships, with
LiveKit's rule: start after 0.05 s above 0.5, end after 0.55 s below 0.35) behind each
filter. Speech starts per minute of **another voice talking alone**, on real recordings
(each start would make the agent stop talking for nobody):

| | no processing | GTCRN | VoxLock (live) | release (live) |
|---|---|---|---|---|
| Krisp benchmark, 85 min of background speech | 5.1 | 6.5 | 0.16 | 0.13 |
| ai-coustics real calls | 9.0 | 10.5 | 4.5 | 1.8 |
| AMI meetings | 2.5 | 3.9 | 1.2 | 1.7 |

On call-centre calls with another voice mixed in, against the same VAD run on the
caller's own channel:

| | no processing | GTCRN | release (live) |
|---|---|---|---|
| speech heard in the caller's pauses | 48% | 48% | 23% |
| caller's turn end seen late, 90th percentile | 2.4 s | 2.4 s | 1.0 s |
| caller turn ends never seen before they talk again | 28% | 26% | 18% |
| the caller's own speech missed | 0.1% | 0.3% | 8.4% |
| ... on a caller who is alone | 0.0% | 0.6% | 5.2% |

The cost is in the last two rows: the model removes some of the caller's own speech,
most of it in the first seconds of a call, before the enrollment exists. These are the
GPU release's numbers; the CPU model's come with its weights. Method and data:
Voice-Isolator's preprint, section "In a voice agent" (`scripts/vad_eval.py`).

## Streaming benchmark

`benchmark/` measures the filter inside a real LiveKit agent: its own LiveKit server, a
LiveKit Agents worker with `AgentSession` (Silero VAD, dummy STT/LLM/TTS) and callers
streaming recordings into rooms in real time, many at once. It reports the filter's time
per frame, its CPU per call, drift, and the VAD's false starts and turn ends against the
caller alone. See [benchmark/README.md](benchmark/README.md).

## Notes

- **Mono only.** Anything else passes through with one warning, rather than being
  downmixed behind your back.
- **Run it at 16 kHz.** Other rates are resampled in and back out (LOW down, QUICK up,
  GTCRN's choices), which adds the resamplers' delay.
- **Fails loudly at session start.** A model that cannot load, or a server that does not
  answer within `connect_timeout`, raises when the `VoiceIsolator` is built, not as audio
  passed through untouched.
- **A server lost mid-call passes the caller through** for the rest of that call, with one
  warning: the agent keeps hearing the caller, unfiltered. It does not reconnect, because
  the server's state for the call (its enrollment, its last 10 s) went with the connection.
- **A late server costs at most `max_lag_ms` (250) of delay.** Audio whose isolated
  version is not back by then goes out as it came in, unfiltered, rather than ever
  later; delay a stall left behind is won back in the caller's pauses. `delay_ms` (how
  far the output stands behind the input) and `unfiltered_ms` (how much went out
  unfiltered) are there to watch.
- **`_process` runs on the agent's audio read loop.** With `LocalIsolator` that is one step
  of the model per frame on one thread; with `RemoteIsolator` it only queues the frame and
  takes what has come back, and the network lives on one background thread per process.
- `enabled` is live: `nc.enabled = False` bypasses it mid-session.

## Models

Scicom's Voice-Isolator: a causal transformer over 10 ms STFT frames (32 bands, attention
across bands and a small LSTM per band, folded into one token per frame; a transformer
that sees the last 10 s; one frame LSTM; a complex mask), conditioned on the caller
through FiLM from a speaker vector. The release adds WavLM-Large's own weights made
causal. Trained on 2,760 h of speech with simulated rooms, lines and crosstalk plus real
headset crosstalk from AMI, NOTSOFAR and AliMeeting; chosen on development sets only.
The training data carries mixed licences: internal use until reviewed.
