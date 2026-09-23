"""LiveKit Agents plugins that ship with STT-API.

Install just these — no server stack, no torch:

    uv pip install "stt-api[scicom-livekit-plugin] @ git+https://github.com/Scicom-AI-Enterprise-Organization/STT-API.git"

Then one import line, whichever plugins the agent uses:

    from stt_api.livekit_plugin import GTCRN, PulseVAD, SemanticVAD, WhisperSTT

- ``WhisperSTT`` / ``DropBlankSTT`` — STT that silence cannot interrupt (``whisper_stt``)
- ``PulseVAD`` — frame-level voice activity, 2,118 parameters (``pulse_vad``)
- ``SemanticVAD`` — end of turn from audio, ahead of the transcript (``semantic_vad``)
- ``MultilingualModel`` — end of turn from text, vLLM-backed (``turn_detector``)
- ``GTCRN`` — self-hosted noise cancellation (``noise_cancellation``)
- ``DummySTT`` / ``DummyLLM`` / ``DummyTTS`` — free load testing (``dummy``)

Each name is resolved on first access rather than at import time. Reaching for
the turn detector therefore does not load onnxruntime, open the dummy TTS audio
file, or register an inference runner for a plugin you are not using — and an
environment that is only partly installed still serves the parts it has.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "GTCRN",
    "MODEL_SAMPLE_RATE",
    "DropBlankSTT",
    "DummyLLM",
    "DummySTT",
    "DummyTTS",
    "MultilingualModel",
    "PulseVAD",
    "PulseVADModel",
    "ScicomEoT",
    "SemanticVAD",
    "SmartTurnV3",
    "WhisperSTT",
]

# Exported name -> (submodule, attribute in that submodule).
_EXPORTS = {
    "WhisperSTT": ("whisper_stt", "WhisperSTT"),
    "DropBlankSTT": ("whisper_stt", "DropBlankSTT"),
    "PulseVAD": ("pulse_vad", "PulseVAD"),
    "PulseVADModel": ("pulse_vad", "PulseVADModel"),
    "SemanticVAD": ("semantic_vad", "SemanticVAD"),
    "SmartTurnV3": ("semantic_vad", "SmartTurnV3"),
    "ScicomEoT": ("semantic_vad", "ScicomEoT"),
    "MultilingualModel": ("turn_detector", "MultilingualModel"),
    "GTCRN": ("noise_cancellation", "GTCRN"),
    "MODEL_SAMPLE_RATE": ("noise_cancellation", "MODEL_SAMPLE_RATE"),
    "DummySTT": ("dummy", "STT"),
    "DummyLLM": ("dummy", "LLM"),
    "DummyTTS": ("dummy", "TTS"),
}

INSTALL_HINT = (
    'uv pip install "stt-api[scicom-livekit-plugin] @ '
    'git+https://github.com/Scicom-AI-Enterprise-Organization/STT-API.git"'
)

if TYPE_CHECKING:  # import-free completion for editors and type checkers
    from .dummy import LLM as DummyLLM
    from .dummy import STT as DummySTT
    from .dummy import TTS as DummyTTS
    from .noise_cancellation import GTCRN, MODEL_SAMPLE_RATE
    from .pulse_vad import PulseVAD, PulseVADModel
    from .semantic_vad import ScicomEoT, SemanticVAD, SmartTurnV3
    from .turn_detector import MultilingualModel
    from .whisper_stt import DropBlankSTT, WhisperSTT


def __getattr__(name: str):
    try:
        submodule, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    try:
        value = getattr(importlib.import_module(f"{__name__}.{submodule}"), attribute)
    except ModuleNotFoundError as exc:
        # A dependency of the plugin is missing, not the plugin itself: say which
        # one, and how to get it, instead of a bare "No module named 'livekit'".
        if exc.name and not exc.name.startswith(__name__.split(".", 1)[0]):
            raise ModuleNotFoundError(
                f"{name} needs {exc.name!r}, which comes with the "
                f"scicom-livekit-plugin extra:\n    {INSTALL_HINT}",
                name=exc.name,
            ) from exc
        raise

    globals()[name] = value  # resolved once; later reads are plain module lookups
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
