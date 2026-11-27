# remote_silero

Silero VAD scored on a GPU server, as a drop-in for `livekit.plugins.silero.VAD`.
The agent keeps the windowing, LiveKit's ExpFilter, the state machine and the
events. Only the model call moves. The server batches the windows of every call it
holds into one model run.

```python
from stt_api.livekit_plugin import RemoteSileroVAD

def prewarm(proc):
    proc.userdata["vad"] = RemoteSileroVAD.load("ws://gpu-host:8766")  # or SILERO_VAD_URL
```

`SILERO_VAD_API_KEY` in the agent's environment is sent as the Bearer key when
`headers` carries no `Authorization`.

Server, one GPU:

```bash
pip install "silero-vad==6.2.1" --no-deps   # 6.1 for livekit-agents 1.2 / 1.3
CUDA_VISIBLE_DEVICES=0 python -m stt_api.livekit_plugin.remote_silero.server --port 8766
```

## Deployed (tm-h20)

```python
RemoteSileroVAD.load(
    "wss://serverlessgpu.aifoundry.scicom.dev/proxy/silero-vad/",
    headers={"Authorization": f"Bearer {sgpu_key}"},   # an AI Foundry key
)
```

Slurm job `silero-vad-1021` on 1021 GPU 1 (`ucc_slurm-ui-job` `jobs/tm-h20/silero-vad-1021.yaml`), silero-vad
6.2.1 weights, the ones livekit-agents 1.8.x bundles (`ucc_tm-voice-agent`), two workers. A livekit-agents 1.3
agent (`ucc_tm-voice-assist`) bundles 6.0 / 6.1 and needs its own instance (`SILERO_VAD_VERSION=6.1`, another
port); against this one its turns would differ from its local silero.

The gateway proxy `silero-vad` reaches it through the `h20-tunnel-silero-vad` tunnel (ai-foundry-poc), which
goes straight to 1021 since 2026-11-26, and adds the server's own key. A restart of the job deploys STT-API `main`.

### The path is the delay, not the server

One real-time call against this job, 2026-11-27 (`benchmark.py`, `turn_delay.py`):

| from | round trip p50 / p99 | windows scored on the agent | a turn event after local silero's |
|---|---:|---:|---:|
| box 1023, the node's LAN address | 0.9 / 1.6 ms | 0 % | 1 ms |
| the same, 20 calls | 2.7 / 3.4 ms | 0 % | |
| a laptop on the VPN, the public URL | 48 / 82 ms | 0.15 % | starts 66 ms, ends 49 ms |

The server and a LAN hop cost a millisecond. Everything above that is the route: TLS, the gateway, the tunnel
and the network in front of them. A second laptop run had a round trip of 64 ms at the median and 154 ms at the
99th percentile, with 4.6 % of its windows scored on the laptop for being more than `max_lag` late.

An agent that runs in the AI Foundry cluster can skip the gateway and dial the tunnel's Service with the
server's own key:

```bash
SILERO_VAD_URL=ws://h20-tunnel-silero-vad.serverlessgpu.svc.cluster.local:8766
SILERO_VAD_API_KEY=...   # the key the server was started with, not an AI Foundry key
```

That route was not measured: nothing here can run a pod in that cluster. Measure it from an agent pod before
switching, with the command under "Is the server doing the work". Two things to settle first. The Service is
plain `ws://`: the caller's audio crosses the cluster unencrypted up to the tunnel pod, and `ucc_tm-voice-agent`
refuses a `ws://` URL for its isolator for that reason. And the gateway's per-key accounting no longer sees the
traffic.

## Pin the weights to the agents' LiveKit

`livekit-plugins-silero` ships different silero weights by version: 1.2.x and 1.3.x
carry silero-vad 6.0 / 6.1, 1.8.x carries 6.2. The two disagree by up to 0.79 in
p(speech) on the same window. Install the `silero-vad` release that matches the
agents. The server's hello carries the sha256 of its weights, and the client logs a
warning when they are not the ones its local plugin ships.

## One connection per call

An `AgentSession` with a non-streaming STT opens two VAD streams and pushes both the
same frame objects: its own, and its `StreamAdapter`'s. Streams of one
`RemoteSileroVAD` that are pushed the same frames share one connection, one model
state on the server and every window's score. Each stream keeps its own filter,
state machine, speech buffer and events, so nothing an agent sees changes. Without
this every window of a call went to the server twice.

A stream joins the older connection that has taken the frame it holds, on its first
frames and now and then after. Only the younger stream moves, so a stream that has
been running keeps its state. Streams fed different frame objects, even with the same
audio in them, keep a connection each.

## Is the server doing the work

Every window the server does not answer within `max_lag` is scored on the agent's
CPU. That is the fallback working, and it is also the CPU the server was meant to
save, so watch how often it happens:

```python
vad.stats()
# {"windows": 70300, "windows_remote": 70300, "windows_local": 0, "windows_late": 0,
#  "local_share": 0.0, "late_share": 0.0,
#  "round_trip_ms_p50": 2.7, "round_trip_ms_p95": 3.1, "round_trip_ms_p99": 3.4,
#  "streams": 100, "connections": 50, "connects": 50, "outages": 0}
```

`windows_local` counts every window scored here, `windows_late` those of them that
were late rather than lost with the connection. The round trip is of the last 4,096
windows the server scored. The same numbers are logged every `log_every` seconds
(60) while there is traffic.

To measure a route from where the agent runs, before any call uses it:

```bash
python -m stt_api.livekit_plugin.remote_silero.benchmark --arm remote \
    --url "$SILERO_VAD_URL" --key-env SILERO_VAD_API_KEY --streams 1 --seconds 60
python -m stt_api.livekit_plugin.remote_silero.turn_delay \
    --url "$SILERO_VAD_URL" --key-env SILERO_VAD_API_KEY --audio call.wav --seconds 140
```

The first needs no audio file (`--audio` takes one) and prints the round trip and
`local_share_measured`. The second needs a recording with speech in it: it feeds one
call to local silero and to the server and prints how long after local silero each
turn start and end came.

## The delay, and taking it off

A window's score comes one round trip after the window. A turn's start and end are
reported that much later than local silero reports them, and by nothing else.

`RemoteSileroVAD.load(url, compensate_delay=True)` takes the round trip off
`min_silence_duration`: the smoothed delay of the connection, capped at `max_lag`, in
whole windows of 32 ms, so an end is never reported earlier than local silero would.
Through the public URL from a laptop, ends came 49 ms after local silero's without it
(at most 66 ms) and 25 ms with it (at most 32 ms). Starts are not moved: they stay one
round trip late.

The price is that a pause shorter by that much ends the turn. At a 64 ms round trip
`min_silence_duration` 0.55 behaves as 0.486. It is off by default, and it does
nothing on a LAN, where the round trip is under one window.

## What it costs

Measured on box 1023 (H20), livekit-agents 1.8.9, 2026-11-27: 50 real-time calls in
one process pinned to one core, 20 ms frames, a VAD object per call, the harness's
own cost taken off, two runs each (`benchmark.py`). Percent of a core per call:

| agent input | VAD streams per call | local silero | RemoteSileroVAD |
|---|---|---:|---:|
| 16 kHz | one | 0.86 % | 0.16 % |
| 16 kHz | two, the same frames | 1.68 % | 0.26 % (0.31 % before they shared) |
| 24 kHz | one | 1.18 % | 0.23 % |
| 24 kHz | two, the same frames | over 2 %: the core was full | 0.32 % (0.43 % before) |

Every arm found the same 86 speech starts (85 at 24 kHz), except local silero with
two streams at 24 kHz, which found 68: fifty such calls do not fit one core. Round
trip to a server on the same host: p50 2.5 to 3.4 ms, p99 3.7 to 4.9 ms.

Sharing halves what a call sends and what the server scores. It takes 17 % off the
agent's CPU at 16 kHz and 27 % at 24 kHz, not half: most of what a stream costs the
agent is its own filter, state machine, speech buffer and events, and each stream
still has those. The resampling is done once, which is why 24 kHz gains more.

The server's share falls as it fills, because a batch costs about the same at 200
windows as at 50. One server process on GPU 0, two cores, clients of 100 streams
each at 16 kHz, measured in the steady stretch:

| streams | loop | server CPU, of one core | per stream | mean batch | round trip p50 / p99 | windows scored by the clients |
|---:|---|---:|---:|---:|---:|---:|
| 400 | asyncio | 36 % | 0.091 % | 56 | 4.6 / 7.5 ms | 0 |
| 400 | uvloop | 33 % | 0.084 % | 65 | 4.7 / 6.7 ms | 0 |
| 800 | asyncio | 64 % | 0.080 % | 103 | 7.8 / 19 ms | 0 |
| 800 | uvloop | 51 % | 0.064 % | 147 | 8.7 / 22 ms | 0 |
| 1000 | asyncio | 71 % | 0.071 % | 233 | 16 / 78 ms | 0 |
| 1000 | uvloop | 69 % | 0.069 % | 99 | 7.3 / 18 ms | 0 |
| 1200 | asyncio | 84 % | 0.070 % | 193 | 14 / 49 ms | 0 |
| 1200 | uvloop | 76 % | 0.064 % | 168 | 8.5 / 28 ms | 0 |
| 1400 | uvloop | 84 % | 0.060 % | 183 | 11 / 31 ms | 0 |
| 1600 | uvloop | 99.5 % | 0.062 % | 443 | 31 / 161 ms | 0.02 % |

The round trips are the worst client's. So agent plus server is about 0.24 % of a
core per call against 0.86 % for local silero, and 0.34 % against 1.68 % for a
session with two VAD streams. Most of the saving is real, not moved. With a few
streams the batches are small and the server costs more per stream (0.6 % at 50
streams, half of it torch launch overhead); check `GET /stats` at your load.

One process is full at about 1,600 streams on uvloop: its event loop is one core.
uvloop took 3 to 23 % off the server's CPU (the two loops form different batches,
so the rows are not like for like), and its gain shows in the tail from 1,000
streams up. Run `--workers` at about one per 1,000 streams.

**The cliff at 1,000 streams was the open-file limit, not the event loop.** An
earlier version of this table had one process collapse there, with round trips of
seconds and the clients failing over, and blamed the loop. The process was at the
usual soft limit of 1,024 files: it could not accept the calls past about a
thousand, and the refused clients scored on their own CPU. With the limit still at
1,024, 1,000 streams on asyncio lost 400 connections in 30 s and the clients scored
9 % of the windows; on uvloop the 1,000 streams looked healthy and 5 % of them never
connected. With the limit raised, the rows above. The server now takes the hard
limit itself at start and logs it.

The VAD was never a large share of an agent's CPU: GTCRN noise cancellation costs
about 3 % of a core per stream (`PULSE_VAD_BENCH.md`). Look there first for a CPU
problem.

## Parity

* At 16 kHz with the server's `onnx` backend (LiveKit's own graph), every event is
  upstream silero's, value for value (`tests/test_remote_silero.py`).
* The GPU `torch` backend against local silero over 120 s of speech: p within
  2e-06 per window at 16 kHz, the same 11 turns at 16 and 48 kHz.
* At other input rates the plugin resamples with Clearline's numpy resampler, not
  LiveKit's: p moves in the third decimal, the turns do not.

## Failure

* An unreachable server raises in `load()`, at prewarm.
* A window the server has not answered `max_lag` seconds (0.2) after it was sent is
  scored on the agent's CPU at once, by LiveKit's silero warmed on the second of
  audio before it. The connection stays up and the server's replies take over again
  once they are back inside `max_lag`: a slow server is late, not lost.
* A server that sends nothing for `reply_timeout` (0.5 s), or owes a reply 2 s old,
  is lost: every window is scored here while the stream reconnects in the
  background.
* A stream does not wait for its connection to open. The first windows queue behind
  the handshake and are scored here if it takes longer than `max_lag`.
* `fallback=False` scores those windows 0.0 instead: the agent hears nobody until
  the server is back.

## Server notes

* One WebSocket per VAD stream. Binary in: whole 512-sample int16 windows at 16 kHz.
  Binary out: one float32 per window, in order. Text `{"reset": true}` zeroes the
  call's state. The protocol is in `server.py`'s docstring.
* A call has at most one window per batch, since its next window needs the state the
  last one left. Batches grow with load by themselves. `--max-wait-ms` trades latency
  for fewer, larger batches on a shared GPU.
* TorchScript's optimizer is off: it recompiled on new batch sizes and stalled one
  batch for 182 ms.
* `--workers N` runs N processes on one port (SO_REUSEPORT, Linux).
* `pip install uvloop` and the server runs on it (`--loop auto`, the default; the
  start line names the loop). One process's limit is its event loop, not the GPU.
* The server takes the hard limit on open files at start. At the usual soft limit
  of 1,024 a process refuses the calls past about a thousand, and a client that
  cannot connect scores on its own CPU with no error on either side.
* `SILERO_VAD_API_KEY` set on the server makes the WebSocket and `/stats` require
  `Authorization: Bearer <key>`. Pass it as
  `RemoteSileroVAD.load(url, headers={"Authorization": f"Bearer {key}"})`.
  `/health` stays open.
