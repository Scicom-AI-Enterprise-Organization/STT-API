"""
Clearline for LiveKit Agents: voice activity detection and noise cancellation on
one CPU core.

- `ClearlineVAD` — a drop-in for `silero.VAD` (`agents.vad.VAD`): a 55k-parameter
  GRU on log-mel features, one probability every 16 ms, 61 us of one core per
  32 ms against silero v6's 101 us, trained for wideband and telephone audio.
- `ClearlineNC` — a drop-in for `GTCRN` (`rtc.FrameProcessor[rtc.AudioFrame]`):
  a band-gain and deep-filter enhancer, 32 ms delay at 16 kHz, 3.8x less CPU than
  GTCRN. Defaults to its speech-recognition mode.

    from stt_api.livekit_plugin import ClearlineNC, ClearlineVAD

    def prewarm(proc):
        proc.userdata["vad"] = ClearlineVAD.load()     # downloads the pinned weights once

    session = AgentSession(vad=ctx.proc.userdata["vad"], stt=..., llm=..., tts=...)
    room_io.AudioInputOptions(sample_rate=16000, noise_cancellation=lambda params: ClearlineNC())

Two things that fail silently when wrong, both in README.md:

* The default threshold is **0.825, not silero's 0.5** (0.85 with 256-sample windows;
  0.85 with the filter at 0.2 for the distilled variant, 0.825 with 0.35 at 256): the operating points chosen
  on development data for the pinned weights (`hub.OPERATING_POINT`).
* `smoothing_alpha` is LiveKit's ExpFilter alpha, the weight of the **previous**
  value (1.0 freezes the filter, so it is refused) — the opposite reading of
  `PulseVAD`'s option of the same name.

The weights live in Hugging Face repositories (`hub.VAD_REPO`, public, and
`hub.NC_REPO`, private), pinned by revision: the noise filter needs an `HF_TOKEN`
that can read its repository, or a local copy via `CLEARLINE_NC_ONNX_PATH` (the VAD's
is `CLEARLINE_VAD_ONNX_PATH`).
"""

from .hub import NC_REPO, OPERATING_POINT, VAD_REPO, nc_model_path, vad_model_path

__all__ = [
    "NC_REPO",
    "OPERATING_POINT",
    "VAD_REPO",
    "ClearlineNC",
    "ClearlineVAD",
    "ClearlineVADStream",
    "nc_model_path",
    "vad_model_path",
]


def __getattr__(name: str):
    # Deferred so `hub` imports without livekit-agents present.
    if name in ("ClearlineVAD", "ClearlineVADStream"):
        from . import vad

        return getattr(vad, name)
    if name == "ClearlineNC":
        from .nc import ClearlineNC

        return ClearlineNC
    raise AttributeError(name)
