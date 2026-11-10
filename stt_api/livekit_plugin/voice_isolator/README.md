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
| model | `Scicom-intl/voice-isolator-cpu`, distilled for one CPU thread | `Scicom-intl/voice-isolator`, the release: M + a causal WavLM-Large, 555M; or its `agent/` model, for agents that talk back ([below](#agents-that-talk-back-the-agent-model)) |
| runs | in the agent process: onnxruntime + numpy, no torch, no GPU | on a GPU server (`Dockerfile`), one WebSocket per call |
| delay in an agent | 10 ms (the STFT look-ahead) | 40 ms median at 32 calls, 50 ms at 64 (LiveKit's 50 ms frames, the server at 40 ms steps, each frame waiting up to 30 ms for its reply); 30 ms with 20 ms frames and steps |
| cost | one thread per call on the agent's audio loop: 17% of a core in 50 ms frames, 33% in 20 ms (8-bit 12% and 22%) | each frame waits on the agent's audio loop for its reply: 15 ms median, 30 ms at most, no CPU (`wait_ms=0` only queues it: 0.4 ms, but 90 ms behind); 64 calls on one GPU measured |

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

- `LocalIsolator()` reads `Scicom-intl/voice-isolator-cpu` (public), or a local
  directory: `LocalIsolator("/models/voice-isolator-cpu")` or `VOICE_ISOLATOR_MODEL`.
  `LocalIsolator("Scicom-intl/voice-isolator-cpu/agent")` is
  [the agent model](#agents-that-talk-back-the-agent-model) on a CPU.
  `quantized=True` takes the 8-bit export from the repository's `int8/`: about 1.4x
  cheaper per call (1.6x in 10 ms frames), but it agrees with the float model to about
  50 dB, and with itself across frame sizes to about 40 dB, rather than to float
  rounding, so measure it on your calls before you switch.
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
whatever the line carried; pass `enrollment=` when you have a clip of the caller, and
see the voicebot mode below for an agent that speaks first. A model can ask for another
rule in its `config.json` ("serve": `loud_within`, `arrive_db`, `arrive_hold`, or
`"enroll": "none"`), and `live_enroll=True` / `False` overrides it.

## Agents that speak first: the voicebot mode

An agent that greets the caller is interrupted by noise most often during its greeting.
On a voice-agent benchmark (LiveKit agents with Silero VAD at a deployed agent's
settings, holding whole conversations with simulated phone callers over babble, street,
home sounds, music or another voice, through an 8 kHz telephone line; 816 held-out
calls), the agent's speech before the caller's first word is 11% of its speech but holds
about half of the false interruptions behind this plugin (57% with `RemoteIsolator`, 48%
with `LocalIsolator`). In its first second the model lets any voice through, and at 3 s
the live enrollment takes the loudest 3 s heard so far, which during a greeting is the
TV, the music or the other person.

`voicebot=True` changes only the start of each call:

- until the caller is enrolled, the model is told to keep a voice that is in no call (a
  decoy: `resources/voicebot_decoy.npy`, 4 s of LibriTTS-R), so it keeps no voice rather
  than every voice;
- the first enrollment waits until the loudest 3 s heard is 6 dB over the call's first
  3 s (what the line carried during the greeting), or until the call is 10 s old.

```python
LocalIsolator(voicebot=True)
RemoteIsolator("ws://gpu-host:8765", voicebot=True)   # each call opens as ?voicebot=1
```

False interruptions per minute of agent speech on that benchmark, the same 816 calls
through each; the change with its paired 90% interval over call plans:

| | default | `voicebot=True` | change |
|---|---|---|---|
| `RemoteIsolator` (the GPU release) | 1.02 | 0.78 | -0.24 (-0.37 to -0.13) |
| `LocalIsolator()` | 1.36 | 1.13 | -0.23 (-0.34 to -0.13) |
| `LocalIsolator(quantized=True)` | 1.36 | 1.13 | -0.23 (-0.34 to -0.14) |

No filter: 2.77. NVIDIA's Real-time RE-USE on a GPU: 0.95, which the GPU release in the
voicebot mode beats by 0.17 (0.03 to 0.31).

The mode removes 42% of the greeting's false interruptions on the GPU (308 to 180) and
36% on the CPU (342 to 220), and leaves the rest of the call as it was: after the
caller's first word the count is the same (233 and 233 on the GPU, 376 and 378 on the
CPU), and so are the caller's turns the agent never hears (0.6% on the GPU, 0.3% on the
CPU), the cut-ins it catches (91%) and Whisper's word error rate on the caller's turns
(13.4% on the GPU; 11.9% and 11.8% on the CPU). The cost falls on a caller who talks
before the enrollment: one who starts in the first second (an outbound call answered
with "hello?") loses about a quarter of that first turn on a clean line and nearly a
third over background noise, against 0-1% by default, and one who cuts into the greeting
is not heard until their voice is 6 dB over the call's start. So it is for agents that
speak first, not for agents that wait for the caller.

`RemoteIsolator(voicebot=True)` needs a server that knows the mode: the model
repository's server has it from 2026-11-02 (rebuild the `Dockerfile`'s image); an older
one ignores the request and runs the default. `--voicebot` on the server makes it every
call's default (`voicebot=False` sends `?voicebot=0`).

## Agents that talk back: the agent model

`Scicom-intl/voice-isolator/agent` is the GPU release fine-tuned further for agents,
with a head that says every 20 ms whether the caller is talking. Where the caller is
not, the server's step sends exact digital silence instead of what the mask left of the
noise or another voice, so the agent's VAD has nothing to fire on. It adds no delay. On
the same benchmark (816 held-out calls) it paused the agent by mistake 0.54 times a
minute of agent speech, against 0.95 behind NVIDIA's Real-time RE-USE, and it kept about
as much of the caller: the VAD missed 2.7% of the caller's turns (RE-USE 2.6%), Whisper
got 15.6% of the caller's words wrong (16.9%), and the agent stopped for 84.8% of the
caller's cut-ins (87.4%).

Only the server changes; the agent's code stays `RemoteIsolator(url)`:

```bash
python -m vi.server_mp --ckpt Scicom-intl/voice-isolator/agent --port 8765 \
    --workers 8 --enroll-loud --chunk-hops 4 --max-streams 128
# or with this directory's Docker image (it holds the whole model repository; rebuild
# an image built before 2026-11-10):
docker run --gpus all -p 8765:8765 voice-isolator-gpu python -m vi.server_mp \
    --ckpt /model/agent --port 8765 --workers 8 --enroll-loud --chunk-hops 4 --max-streams 128
```

The server reads the gate's setting from `agent/config.json` (`--gate off` serves the
model without it).

On a CPU, inside the agent: `Scicom-intl/voice-isolator-cpu/agent` is the CPU student
trained the same way, with this model as its teacher. `LocalIsolator` applies its gate
itself (`gate="auto"` reads `agent/config.json`; `gate="off"` runs the model without it):

```python
proc.userdata["isolator"] = LocalIsolator("Scicom-intl/voice-isolator-cpu/agent")
# 8 bits: LocalIsolator("Scicom-intl/voice-isolator-cpu/agent", quantized=True)
```

On the same 816 held-out calls, in 8 bits on one thread, it paused the agent by mistake
0.48 times a minute of agent speech (RE-USE 0.95, this plugin's default CPU model 1.36),
and kept about as much of the caller as RE-USE: the VAD missed 2.4% of the caller's
turns (2.6%), Whisper got 16.2% of the caller's words wrong (16.9%), and the agent
stopped for 87.9% of the caller's cut-ins (87.4%). It costs the same CPU as the default
model: 12.5% of one core per call in 8 bits, 50 ms frames.

### The agent models in the voicebot mode

For an agent that greets first, `voicebot=True` with either agent model. Most of what
still pauses an agent behind them is in its greeting, before the caller has said a word
(three quarters of their false interruptions on development calls), and the mode takes
most of that. Held out, the same 816 calls, false interruptions per minute of agent
speech with the paired 90% interval of the change:

| | default | `voicebot=True` | change | RE-USE |
|---|---|---|---|---|
| `LocalIsolator(".../voice-isolator-cpu/agent", quantized=True)` | 0.48 | 0.26 | -0.22 (-0.34 to -0.12) | 0.95 |
| `RemoteIsolator`, the server on `voice-isolator/agent` | 0.54 | 0.32 | -0.22 (-0.33 to -0.12) | 0.95 |

| the caller, in the mode (default; RE-USE) | CPU agent model, 8 bits | GPU agent model |
|---|---|---|
| turns the VAD missed | 1.7% (2.4%; 2.6%) | 2.5% (2.7%; 2.6%) |
| Whisper's word errors | 16.8% (16.2%; 16.9%) | 16.5% (15.6%; 16.9%) |
| cut-ins the agent stopped for | 88.3% (87.9%; 87.4%) | 85.5% (84.8%; 87.4%) |

What the mode does for these two models is in their `config.json` ("serve",
"voicebot"), and the plugin and the server read it from there:

- after the wait the live enrollment runs until the caller has arrived (a voice 6 dB over
  the call's first 3 s, then 15 s more, at most 60 s) instead of stopping when the call
  is 15 s old, so a caller who answers late is enrolled on their own voice;
- while the decoy holds, a frame the model kept is a weaker sign of the caller than
  later, so it opens the gate only from the third in a row on the GPU model and not at
  all on the CPU model (there the head alone opens it).

The cost is the mode's own: a caller who talks in the agent's first second loses about a
quarter of that turn on a clean line (a third over noise on the CPU model, a sixth on
the GPU model), as before these changes. It needs this plugin from 2026-11-11 on and,
for `RemoteIsolator`, the model repository's server code of that date; older code
serves these models as before (0.34 and 0.39 on development calls, against 0.26 and
0.32 with it).

More of the caller instead of fewer pauses, by default, on the CPU agent model:
`LocalIsolator(".../agent", live_enroll=False, gate="0.9:50:0.3:1")` runs it with no
enrollment. Held out it missed 1.4% of the caller's turns and got 14.8% of the words
wrong, for 0.53 false interruptions a minute against 0.48.

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
The training data carries mixed licences, and the licence of the weights is not set yet.
