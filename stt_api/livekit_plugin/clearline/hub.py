"""
Where the Clearline weights come from.

The ONNX graphs are over the repo's 1 MB pre-commit limit (1.4 MB for the VAD,
1.6 MB for the noise canceller), so they are not vendored: they are fetched from
the Hugging Face repositories below on first use, **pinned to the revision this
plugin was tested against**, and cached by `huggingface_hub` as usual.
Clearline-VAD is public; Clearline-NC is private, so the noise filter needs an
`HF_TOKEN` that can read it.

The pin is not decoration. The VAD's default operating points (`OPERATING_POINT`:
for the release 0.825 at 512-sample windows and 0.875 at 256 with the filter at 0.35,
for the distilled variant 0.825 with the filter at 0.35 at 512 and 0.75 with 0.5 at
256) were chosen on development
data *for these weights*; a newer upload can be
calibrated differently, and the wrong threshold fails silently (a VAD that fires
too often, or late). Bump `VAD_REVISION` and `OPERATING_POINT` together, never
one without the other.

A local file overrides the download: pass `model_path=` / `onnx_file_path=`, or
set `CLEARLINE_VAD_ONNX_PATH` / `CLEARLINE_NC_ONNX_PATH`.
"""

from __future__ import annotations

import os

VAD_REPO = "Scicom-intl/Clearline-VAD"
NC_REPO = "Scicom-intl/Clearline-NC"
VAD_REVISION = "da8f290b566938ee1b1ce30f211b9be94ebf3836"
NC_REVISION = "f0d0e14826ef5b608fc806a82e6e50a1d3bca6d5"

VAD_FILES = {
    ("release", 512): "clearline_vad.onnx",
    ("release", 256): "clearline_vad_256.onnx",
    ("distilled", 512): "clearline_vad_distilled.onnx",
    ("distilled", 256): "clearline_vad_distilled_256.onnx",
}
"""(variant, window samples) -> file. Same weights per variant; the window is how
many samples one call takes (512 = 32 ms, silero's cadence; 256 = 16 ms)."""

OPERATING_POINT = {
    ("release", 512): {"activation_threshold": 0.825, "smoothing_alpha": 0.35},
    ("release", 256): {"activation_threshold": 0.875, "smoothing_alpha": 0.35},
    ("distilled", 512): {"activation_threshold": 0.825, "smoothing_alpha": 0.35},
    ("distilled", 256): {"activation_threshold": 0.75, "smoothing_alpha": 0.5},
}
"""(variant, window) -> the default options, chosen on the development sets (AMI
dev and VoxConverse dev, wideband and through a phone codec) for each graph at
`VAD_REVISION`. The release: the highest mean of LiveKit onset and end-of-turn
recall among points at least as good as silero v6's own tuned point on onset
recall, end recall, false triggers per hour and onset delay. The distilled variant
(a larger graph: 96 units and per-band level trackers, chosen for frames on six
development sets including rooms full of other people's voices) takes the same rule's
point for itself. At this revision both graphs were fine-tuned with half their training
examples heard through Clearline-NC first, because a LiveKit agent runs its noise filter
before its VAD; behind a filter they never heard (GTCRN) they rank frames better than
the weights before them, and with no filter they are level."""

NC_FILES = {"asr": "clearline_nc_asr.onnx", "quality": "clearline_nc.onnx"}


def _local(path: str | os.PathLike | None, env: str) -> str | None:
    path = path or os.environ.get(env)
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Clearline model file not found: {path}")
    return str(path)


def _fetch(repo: str, filename: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo, filename, revision=revision)
    except Exception as e:  # noqa: BLE001 - re-raised with what to do about it
        raise FileNotFoundError(
            f"could not fetch {filename} from {repo}@{revision[:7]} ({type(e).__name__}: {e}). "
            "The repository is private: set an HF_TOKEN that can read it, or point at a "
            "local copy with model_path= or the CLEARLINE_*_ONNX_PATH variables."
        ) from e


def vad_model_path(
    variant: str = "release",
    window: int = 512,
    model_path: str | os.PathLike | None = None,
    revision: str | None = None,
) -> str:
    """The VAD graph for (variant, window): a local override, else the pinned download."""
    if (variant, window) not in VAD_FILES:
        raise ValueError(
            f"no Clearline-VAD graph for variant={variant!r}, window={window}; "
            f"have {sorted(VAD_FILES)}"
        )
    local = _local(model_path, "CLEARLINE_VAD_ONNX_PATH")
    if local:
        return local
    return _fetch(VAD_REPO, VAD_FILES[(variant, window)], revision or VAD_REVISION)


def nc_model_path(
    mode: str = "asr",
    model_path: str | os.PathLike | None = None,
    revision: str | None = None,
) -> str:
    """The noise canceller for `mode`: a local override, else the pinned download."""
    if mode not in NC_FILES:
        raise ValueError(f"mode must be one of {sorted(NC_FILES)}, got {mode!r}")
    local = _local(model_path, "CLEARLINE_NC_ONNX_PATH")
    if local:
        return local
    return _fetch(NC_REPO, NC_FILES[mode], revision or NC_REVISION)
