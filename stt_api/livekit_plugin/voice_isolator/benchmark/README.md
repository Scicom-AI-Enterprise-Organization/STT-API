# Voice isolator benchmark: a real LiveKit agent, streamed in real time

What the filter costs inside an agent and what it does to the agent's VAD, measured the
way an agent runs: a LiveKit server, a LiveKit Agents worker with `AgentSession`, and
callers publishing recordings into rooms as microphones, in real time, many at once.

```bash
LIVEKIT_SERVER=/path/to/livekit-server FILES=calls.txt CALLS=32 \
VI_BENCH_MODEL=/models/voice-isolator-cpu VOICE_ISOLATOR_URL=ws://gpu-host:8765 \
    bash stt_api/livekit_plugin/voice_isolator/benchmark/run.sh none gtcrn local local-int8 remote
```

`FILES` lists one directory per line with `mix.wav` (the call: the caller and another
voice) and `caller.wav` (the caller alone), 16 kHz mono. Needs `livekit-plugins-silero`
(the `[benchmark]` extra) beside the plugin extra; results land in `$ROOT/report.json`.

## What runs

- **Server.** `run.sh` starts its own `livekit-server` on its own ports (`LK_PORT`,
  default 17950) with an API key it generates, and refuses to run if those ports are
  taken. A machine often has another LiveKit server: reusing it would hand the
  benchmark's rooms to that server's agents, and its rooms to the benchmark's worker.
- **Agent** (`agent.py`). A LiveKit Agents worker (`start` mode), one warm process per
  call. Each room runs an `AgentSession` with LiveKit's Silero VAD at its defaults and
  STT-API's dummy STT, LLM and TTS, so every turn runs the whole pipeline at no cost.
  The filter under test is the audio input's `noise_cancellation`, wrapped in `Timed`,
  which LiveKit calls exactly where it would call the filter. `Timed` records each
  frame's arrival, its position in the stream, the filter's time and the input level.
  The VAD's verdicts come from the session's own `user_state_changed` events.
- **Callers** (`callers.py`). One participant per room joins, waits for the agent,
  publishes a microphone track and pushes 10 ms frames paced on the wall clock.
- **Reference.** The same rooms are run first with `caller.wav` and no filter: what the
  agent's VAD does when only the caller is on the line.

## What is measured (`report.py`)

- **frame**: the filter's time per 50 ms frame, median / p99 / max. It runs on the
  agent's audio read loop, so p99 is what a call feels.
- **filter** and **job**: share of one core per call, in the filter and in the whole job
  process (VAD, dummy pipeline, WebRTC, filter).
- **drift**: how far a call's audio fell behind real time inside the agent (p99 over
  calls). A filter that cannot keep up shows here first.
- Against the reference, each room aligned with its recording by its level contour:
  - **false starts per minute**: the VAD opens a turn while the caller is silent;
  - **pauses heard as speech**;
  - **turn end p90**: how much later than on the caller alone the VAD closes the
    caller's turn;
  - **held**: caller turns it does not close before the caller talks again;
  - **caller missed**: the caller's speech the VAD does not hear.

## Results

The trained models' run (32 calls per configuration, `call_test_mix`: 60 s
call-centre calls with a second voice) goes here.

The harness check, 4 calls per configuration with a random 125K-parameter model in
place of the CPU model and a CPU stand-in for the GPU server:

| | frame p50 | frame p99 | filter, core per call | job, core per call | drift p99 |
|---|---|---|---|---|---|
| no filter | 0.0 ms | 0.0 ms | 0 | 0.17 | 53 ms |
| GTCRN | 2.9 ms | 7.6 ms | 0.069 | 0.26 | 24 ms |
| `LocalIsolator` (random tiny model) | 1.2 ms | 4.5 ms | 0.030 | 0.19 | 17 ms |
| `RemoteIsolator` (queues only) | 0.09 ms | 0.85 ms | 0.003 | 0.17 | 24 ms |

With no filter, against the caller-alone run, the VAD misses 0.5% of the caller's speech
over the 28 caller turns. That is the alignment and WebRTC floor of the method; the
offline measurement on the same calls is 0.1%.
