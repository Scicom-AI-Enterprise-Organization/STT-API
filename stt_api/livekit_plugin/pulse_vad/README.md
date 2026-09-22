# PulseVAD for LiveKit Agents

Frame-level voice activity detection from a **2,118-parameter** CNN — a drop-in
for `silero.VAD`, backed by [PulseVAD](https://github.com/AydinAdnan/PulseVAD)
(MIT). 12 KB of fp32 weights against silero's ~1.8 MB.

```python
from stt_api.livekit_plugin.pulse_vad import PulseVAD

def prewarm(proc):
    proc.userdata["vad"] = PulseVAD.load()       # blocking; do it here

session = AgentSession(
    vad=ctx.proc.userdata["vad"],
    stt=..., llm=..., tts=...,
)
```

## This is not a turn detector

There are now three LiveKit interfaces in this package and mixing them up is the
main way this goes wrong:

| | question | input |
|---|---|---|
| [`../turn_detector/`](../turn_detector/) | has the speaker **finished**? | transcript text |
| [`../semantic_vad/`](../semantic_vad/) | has the speaker **finished**? | streaming audio |
| **`pulse_vad/`** (this) | is someone **speaking right now**? | streaming audio |

An `AgentSession` wants a VAD **and** a turn detector. The VAD decides *when* to
ask; the turn detector answers. They are not alternatives, and passing a semantic
VAD as `vad=` or this as `turn_detection=` will not typecheck into anything
useful.

## Two traps that fail silently

### p(speech) saturates at 0.711, not 1.0

The 2.1k model's output range, measured over 12,098 windows of `test_audio/`
plus synthetic noise:

| checkpoint | range | p99.9 |
|---|---|---|
| `2.1k` fp32 | 0.002 – **0.711** | 0.7107 |
| `2.1k` int8 | 0.005 – **0.708** | 0.7078 |
| `81k` fp32 | 0.006 – **0.895** | 0.8949 |

Coming from silero, `activation_threshold=0.8` is an entirely ordinary thing to
write. Here it produces a VAD that fires on **0.0 %** of windows — an agent that
never hears anyone, with no exception and no log line. `PulseVAD.load()` raises
rather than let you deploy it, and `CEILINGS` in `model.py` is the source of
truth, asserted in `tests/test_pulse_vad.py`.

The default is `0.35`, the midpoint of the measured range — not silero's 0.5,
which sits at 70 % of PulseVAD's usable span. The saturation is flat, not a tail:
p99.9 is within 0.0005 of the maximum.

### int8 is not faster, and the file is bigger

| | size | per window |
|---|---|---|
| `2.1k` fp32 | 12.0 KB | **0.018 ms** |
| `2.1k` int8 | 26.8 KB | 0.019 ms |
| `81k` fp32 | 325 KB | 0.098 ms |

Quantisation pays on the microcontrollers PulseVAD targets. In an agent process
it costs a mean |Δp| of 0.008 (max 0.052) for nothing, and QDQ nodes make the
file more than twice the size. So `precision="fp32"` is the default here even
though upstream's `load_pulsevad` defaults to `quantized=True`. `"int8"` stays
available for parity-testing against an edge deployment.

## The window slides; it is not the update rate

PulseVAD's input is a fixed 200 ms patch. Reporting every 200 ms would be a real
regression — LiveKit's duration defaults are calibrated against silero's 32 ms
cadence, and `min_speech_duration=0.05` would be satisfied by a *single* window,
so one spurious inference would open a turn.

So the 200 ms window slides over a **32 ms hop** (`hop_duration`), matching
silero's update interval at 6.25× the inference count — about **0.3 % of one
core**. Consecutive windows overlap 84 %, which is also why exponential
smoothing is off by default: silero needs `ExpFilter(alpha=0.35)` because a
32 ms window is jittery, but here it cut threshold crossings only from 0.3–1.8/s
to 0.3–1.5/s while adding **32–160 ms** to onset. `min_speech_duration` already
debounces.

## Where it actually loses: onset latency

A 200 ms causal window cannot report speech until speech fills enough of it.
Measured from speech start to the first window over threshold, on trimmed clips
padded with known silence:

| | median | worst |
|---|---|---|
| `2.1k` @ 0.35 | 140 ms | 908 ms |
| `81k` @ 0.35 | 172 ms | 236 ms |

The teacher is far more consistent and is worth its 0.098 ms if barge-in latency
matters more than CPU. Note that on *production* turn recordings both detectors
report the same absolute onset, because those clips open with their own lead-in
noise — see the benchmark's paired timing, which differences within an item to
cancel it.

## Benchmark

[`benchmark/`](benchmark/) runs this against silero through the real
`livekit.agents.vad.VAD` interface, on labelled production audio. See its
[README](benchmark/README.md).

```bash
pip install ".[benchmark]"
python -m stt_api.livekit_plugin.pulse_vad.benchmark --audio test_audio/
```

## Weights

Vendored under `resources/`, not taken from the `pulsevad` PyPI package: that
package hard-depends on **scipy and soundfile** for one file-reading helper this
plugin never calls, and declares `requires-python >= 3.11` while this repo
supports 3.10. Dragging both into an agent image for 12 KB of weights is the
extra-boundary violation `CLAUDE.md` warns about. The front end is reimplemented
in `frontend.py`, **bit-exact** with upstream's `frontend_np` and asserted so.

Override with `model_path=` or `PULSEVAD_ONNX_PATH`. Upstream is MIT;
`resources/LICENSE.pulsevad` ships alongside.
