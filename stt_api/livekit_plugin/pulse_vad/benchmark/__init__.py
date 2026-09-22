"""
Is PulseVAD good enough to replace silero in a LiveKit agent?

Run it:

    python -m stt_api.livekit_plugin.pulse_vad.benchmark --audio test_audio/
    python -m stt_api.livekit_plugin.pulse_vad.benchmark \
        --s3-uri s3://bucket/prefix/clip.wav --conditions clean telephony

Both detectors run through the real `livekit.agents.vad.VAD` interface on
identical audio, and each is scored at **its own** best threshold — see
`metrics.py` for why scoring both at 0.5 would be a measurement of output scale
rather than of accuracy. Needs the `benchmark` extra.
"""

from .corpus import Clip, Item, Region, build_items, local_clips, parse_s3_uri, s3_clips
from .harness import Run, load_pulse, load_silero, run_vad
from .metrics import (
    Scores,
    agreement,
    balanced_score,
    paired_timing,
    score_run,
    segment,
    sweep,
    verify_replay,
)

__all__ = [
    "Clip",
    "Item",
    "Region",
    "Run",
    "Scores",
    "agreement",
    "balanced_score",
    "build_items",
    "load_pulse",
    "load_silero",
    "local_clips",
    "parse_s3_uri",
    "paired_timing",
    "run_vad",
    "s3_clips",
    "score_run",
    "segment",
    "sweep",
    "verify_replay",
]
