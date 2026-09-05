"""
Benchmark harness for LiveKit noise cancellation.

Compares candidate enhancers under the constraints the agent actually imposes —
fixed frames, in order, no lookahead — on quality, realtime cost and downstream
WER. See README.md in this directory for what the numbers mean and how to read
a disagreement between them.

    python -m stt_api.livekit_plugin.noise_cancellation.benchmark --help
"""

from .corpus import Item, load
from .enhancers import REGISTRY, Enhancer, available, build
from .harness import ItemResult, Summary, run_item, summarize
from .metrics import Quality, score_quality
from .report import format_report, to_json

__all__ = [
    "Enhancer",
    "Item",
    "ItemResult",
    "Quality",
    "REGISTRY",
    "Summary",
    "available",
    "build",
    "format_report",
    "load",
    "run_item",
    "score_quality",
    "summarize",
    "to_json",
]
