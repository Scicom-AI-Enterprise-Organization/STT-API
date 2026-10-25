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
- **Callers** (`callers.py`). One participant per room joins, waits for its agent,
  publishes a microphone track and pushes 20 ms frames paced on the wall clock, four
  calls to a process (one Python process cannot push dozens of calls in real time).
  Joining is not measured: every caller connects first, and the calls start streaming
  together at a set time, spread over 10 s, so that they overlap for the whole call.
  On a host with several network interfaces a WebRTC participant can take seconds of
  CPU to connect; 64 at once take a minute (the report's **join**).
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

One host carries everything: the LiveKit server, the agent worker, the callers and, for
`RemoteIsolator`, the GPU server (the plugin's `Dockerfile` command; the release, M +
causal WavLM-Large). The calls are `call_test_mix`: 60 s call-centre calls, each with a
second voice; 32 at once unless noted. The reference is the same rooms with each
caller's own channel and no filter.

What the filter costs the agent:

| filter | frame p50 / p99 | filter, core per call | delay p50 / p99 | unfiltered |
|---|---|---|---|---|
| none | 0.0 / 0.0 ms | 0 | 0 | |
| GTCRN | 3.1 / 5.6 ms | 0.066 | | |
| `RemoteIsolator`, LiveKit's 50 ms frames, server at 40 ms | 0.13 / 0.36 ms | 0.003 | 100 / 100 ms | 0.0% |
| `RemoteIsolator`, 20 ms frames, server at 20 ms | 0.10 / 0.24 ms | 0.005 | 50 / 70 ms | 0.0% |
| `RemoteIsolator`, 50 ms frames, server at 40 ms, **64 calls** | 0.15 / 0.40 ms | 0.003 | 140 / 150 ms | 0.0% |

With 64 calls on one GPU a step of all of them took 23 ms of the server's 40 ms period.

What the agent's VAD did, against each caller's own channel:

| filter | false starts per min | speech in the caller's pauses | turn end p90 | turn ends held | caller missed |
|---|---|---|---|---|---|
| none | 2.84 | 47.5% | 2.04 s | 23% | 0.3% |
| GTCRN | 2.88 | 46.8% | 2.03 s | 22% | 0.7% |
| `RemoteIsolator`, 50 ms frames | 1.78 | 25.6% | 1.15 s | 19% | 7.2% |
| `RemoteIsolator`, 20 ms frames | 1.72 | 25.7% | 1.16 s | 17% | 6.6% |
| `RemoteIsolator`, 64 calls | 1.61 | 23.7% | 1.12 s | 24% | 9.7% |

These agree with the same VAD run offline over the recordings (Voice-Isolator's
`scripts/vad_eval.py`: release 22.8% of pauses, 1.94 false starts a minute, 8.4% missed;
no processing 47.9%, 3.33, 0.1%). Noise suppression leaves another voice in: GTCRN
changes nothing here. The cost is the caller's own speech, mostly the first seconds of a
call, before the live enrollment exists.

The `LocalIsolator` rows (the CPU model) come with its weights.

### What the benchmark found on the way

- **The GPU server fell behind under real agents.** LiveKit hands a call's frames over a
  few ms apart, not on a beat, so the server ran a partial step for nearly every chunk
  and every call slid back for the rest of the call (22 of 32 calls at the delay bound).
  It now steps again only within its tick's budget, plus at most one catch-up step every
  fourth tick for calls that fell behind (`vi/server_mp.py`).
- **The plugin waited for a late server without bound** (a call reached 10 s of delay).
  The delay is now capped at `max_lag_ms`; audio whose isolated version comes later
  passes unfiltered (`unfiltered_ms`), and delay a stall left behind is won back in the
  caller's pauses.
- **A stream that opens with silence enrolls on it.** The live enrollment counts from
  the stream's first frame, so the callers publish at the call's start, as a real
  caller does; a caller silent for the first 15 s of a real call has the same problem
  (`enrollment=` avoids it).
