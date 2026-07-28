# Noise cancellation plugin for LiveKit Agents

Self-hosted noise cancellation. Runs a 48.2 K-parameter ONNX model inside the agent
process — no LiveKit Cloud, no licence key, no external service.

## Why this exists

`livekit-plugins-noise-cancellation` (Krisp NC / BVC / BVCTelephony) authorises
against LiveKit Cloud. On a self-hosted server it loads its native library, fails
the entitlement check, logs `noise cancellation is not authorized (404)` and passes
audio through untouched — so agents look configured but get raw audio. ai-coustics
is the only vendor path that permits self-hosting, and that needs a paid licence key.

The hook itself is not gated. `AudioInputOptions.noise_cancellation` accepts either
Krisp's `rtc.NoiseCancellationOptions` **or** any `rtc.FrameProcessor[rtc.AudioFrame]`
— a plain Python object called once per audio frame, entirely in-process. This plugin
is that second form.

Worth knowing where it pays off: a browser microphone already arrives processed,
because `livekit-client` enables WebRTC's `noiseSuppression`, `echoCancellation` and
`voiceIsolation` by default. **Inbound SIP audio gets none of that** — that is the
gap this closes, and it is also the audio your STT accuracy depends on most.

## Usage

```python
from livekit.agents.voice import room_io
from stt_api.livekit_plugin.noise_cancellation import GTCRN

await session.start(
    agent=MyAgent(),
    room=ctx.room,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            # Match the model's rate: skips resampling, and it is what STT and VAD want.
            sample_rate=16000,
            # A selector, so each participant stream gets its own recurrent caches.
            noise_cancellation=lambda params: GTCRN(),
        ),
    ),
)
```

One instance per audio stream. The three recurrent caches are per-stream state, so
sharing an instance across participants would cross-contaminate them. The ONNX
session holding the weights *is* shared — cached per (model, thread count) — so a
second stream costs no extra memory for the model.

To apply it only to phone callers, where it matters most:

```python
noise_cancellation=lambda params: (
    GTCRN()
    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    else None
),
```

## Measured behaviour

3.3 s of speech, one core of an M-series laptop:

| | |
|---|---|
| CPU | 3 % of one core per stream (RTF 0.031, ~1.6 ms per 50 ms frame) |
| Added delay | **32 ms at 16 kHz**, 56 ms at 24/48 kHz, 109 ms at 8 kHz |
| Denoising | +9 to +10 dB SNR at 3–10 dB input SNR; +5.5 dB at 20 dB |
| Noise floor in pauses | −25 to −39 dB |
| Clean speech | 30 dB fidelity, level unchanged (+0.05 dB) |

Reproduce with `pytest tests/test_noise_cancellation.py -v -s`.

The 32 ms at 16 kHz is two hops: one of output priming, one because the first
analysis window is mostly zeros. Off-rate audio costs two soxr stages on top —
which is why the snippet above pins 16 kHz.

## Notes

- **Mono only.** Agent input is mono (`AudioInputOptions.num_channels` defaults to
  1). Anything else passes through unfiltered with a single warning, rather than
  being downmixed behind your back.
- **Fails loudly.** A missing model raises at session start instead of degrading to
  silent passthrough — that quiet degradation is the exact failure mode of the Cloud
  plugin this replaces. If `_process` itself ever throws, LiveKit catches it and
  passes the frame through (`rtc/audio_stream.py`), so a bad frame cannot kill a call.
- **`enabled` is live.** Set `nc.enabled = False` to bypass mid-session; the frame is
  then returned untouched.
- `GTCRN_ONNX_PATH` overrides the bundled model.

## Why not DeepFilterNet

Investigated and deliberately not offered. DeepFilterNet3 is the stronger model, but
nothing about it fits a frame-by-frame Python filter on this runtime:

- **Its published ONNX cannot stream.** Both `DeepFilterNet3_onnx.tar.gz` and the
  `_ll` variant export `enc`/`erb_dec`/`df_dec` over a dynamic time axis with **no
  recurrent state inputs or outputs**. `enc` holds a single `GRU` node whose
  `initial_h` is a `ConstantOfShape` of zeros and whose `Y_h` is discarded. Upstream
  streams these with [tract](https://github.com/sonos/tract)'s *pulse* transform,
  which rewrites a sequence model into a streaming one and carries the state
  internally; onnxruntime has no equivalent, so calling it with `S=1` per frame
  resets the GRU every 10 ms. `conv_lookahead=2` over a time-kernel of 3 means the
  convolutions need ring buffers per frame too.
- **`deepfilterlib`** — the Rust DSP, torch-free — exposes only primitives
  (`DF.analysis`/`synthesis`/`erb_widths`/`fft_window`, `erb`, `erb_norm`,
  `unit_norm`), no model runtime, and publishes **no cp312 wheel**. Agents run on
  Python 3.12.
- **`deepfilternet`** — the torch path — pins `numpy>=1.22,<2.0` against a runtime on
  numpy 2.x with onnxruntime 1.28, and torch would add ~2 GB to the agent image.
- Everything else (STFT, 32-band ERB, the `norm_tau` running normalisation, mask
  application, the 5-tap deep filter, ISTFT) lives outside the graphs and would need
  reimplementing in numpy regardless of route.

Two routes exist if it is ever worth revisiting. Patching the graph to expose
`initial_h`/`Y_h` and splitting the encoder so conv buffers live outside gives true
streaming — see [`shimondoodkin/deepfilter-rt`](https://github.com/shimondoodkin/deepfilter-rt),
MIT/Apache like upstream, which does exactly this in Rust. Running the graphs
unmodified over blocks with a warm-up context prefix avoids the surgery but costs
~6× the CPU and 90–110 ms of latency — worse than GTCRN on both axes, better only on
raw denoising quality. Validate either against the official `deep-filter` binary,
published as a prebuilt release asset.

DF3 is also a 48 kHz model (fft 960, hop 480, 32 ERB bands, `nb_df=96`,
`df_order=5`, `df_lookahead=2`), so adopting it would move the agent's whole input
pipeline to 48 kHz.

## Model

[GTCRN](https://github.com/Xiaobin-Rong/gtcrn) by Xiaobin Rong — MIT licensed,
vendored as `resources/gtcrn_simple.onnx` (523 KB) with its licence in
`resources/LICENSE.gtcrn`. 16 kHz, 512-point STFT, 256-sample hop. Cite:

> X. Rong et al., "GTCRN: A Speech Enhancement Model Requiring Ultralow
> Computational Resources", ICASSP 2024.
