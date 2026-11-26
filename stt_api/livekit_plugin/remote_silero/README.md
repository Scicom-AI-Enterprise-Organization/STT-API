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

Server, one GPU:

```bash
pip install "silero-vad==6.2.1" --no-deps   # 6.1 for livekit-agents 1.2 / 1.3
CUDA_VISIBLE_DEVICES=0 python -m stt_api.livekit_plugin.remote_silero.server --port 8766
```

## Pin the weights to the agents' LiveKit

`livekit-plugins-silero` ships different silero weights by version: 1.2.x and 1.3.x
carry silero-vad 6.0 / 6.1, 1.8.x carries 6.2. The two disagree by up to 0.79 in
p(speech) on the same window. Install the `silero-vad` release that matches the
agents. The server's hello carries the sha256 of its weights, and the client logs a
warning when they are not the ones its local plugin ships.

## What it costs

Measured on box 1023 (H20, one GPU), livekit-agents 1.8.9, 50 real-time streams in
one process pinned to one core, harness cost subtracted (`benchmark.py`):

| agent input | local silero | RemoteSileroVAD |
|---|---:|---:|
| 16 kHz | 0.84 % of a core per stream | 0.15 % |
| 24 kHz | 1.20 % | 0.21 % |

Both arms found the same 64 speech starts. Round trip to a server on the same host:
p50 2 ms, p99 3 to 8 ms.

The CPU moves; it does not vanish. At that load the server spent about 0.6 % of a
core per stream, half of it torch launch overhead on batches of 8.8 windows. The
saving is on the agent box. Measure the server at your load (`GET /stats`).

## Parity

* At 16 kHz with the server's `onnx` backend (LiveKit's own graph), every event is
  upstream silero's, value for value (`tests/test_remote_silero.py`).
* The GPU `torch` backend against local silero over 120 s of speech: p within
  2e-06 per window at 16 kHz, the same 11 turns at 16 and 48 kHz.
* At other input rates the plugin resamples with Clearline's numpy resampler, not
  LiveKit's: p moves in the third decimal, the turns do not.

## Failure

* An unreachable server raises in `load()`, at prewarm.
* A server lost mid-call, or silent for `reply_timeout` (0.5 s), fails over to
  LiveKit's silero on the agent's CPU for the windows in flight and after, and the
  stream reconnects in the background. `fallback=False` scores those windows 0.0
  instead: the agent hears nobody until the server is back.

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
