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

## Deployed (tm-h20)

```python
RemoteSileroVAD.load(
    "wss://serverlessgpu.aifoundry.scicom.dev/proxy/silero-vad/",
    headers={"Authorization": f"Bearer {sgpu_key}"},   # an AI Foundry key
)
```

Slurm job `silero-vad-1021` on 1021 GPU 1 (`ucc_slurm-ui-job` `jobs/tm-h20/silero-vad-1021.yaml`), silero-vad
6.1 weights for livekit-agents 1.2 / 1.3, two workers. The gateway proxy `silero-vad` reaches it through the
`h20-tunnel-silero-vad` tunnel (ai-foundry-poc) and adds the server's own key. A restart of the job deploys
STT-API `main`.

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

The server's share falls as it fills, because a batch costs about the same at 50
windows as at 5. One server process on GPU 0, two cores, clients of 100 streams each
at 16 kHz:

| streams | server CPU per stream | mean batch | round trip p99 | windows failed over |
|---:|---:|---:|---:|---:|
| 100 | 0.12 % | 48 | 11 ms | 0 |
| 300 | 0.11 % | 55 | 13 ms | 0 |
| 600 | 0.087 % | 80 | 19 ms | 0 |
| 1000 | 0.092 % | 297 | 2.1 s | 87,117 |
| 1000, `--workers 2` | about 0.12 % | 73 | 21 ms | 0 |

So agent plus server is about 0.25 % of a core per stream against 0.84 % for local
silero. Most of the saving is real, not moved. With a few streams the batches are
small and the server costs more per stream (0.6 % at 50 streams, half of it torch
launch overhead); check `GET /stats` at your load.

One process saturates near 1,000 streams: its event loop is one core (92 % busy),
windows queue for seconds and the clients fail over to local silero. Every window
was still scored. Run `--workers` at about one per 500 streams.

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
* `SILERO_VAD_API_KEY` set on the server makes the WebSocket and `/stats` require
  `Authorization: Bearer <key>`. Pass it as
  `RemoteSileroVAD.load(url, headers={"Authorization": f"Bearer {key}"})`.
  `/health` stays open.
