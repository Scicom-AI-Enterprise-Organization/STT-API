"""
Where the Clearline weights come from.

The ONNX graphs are over the repo's 1 MB pre-commit limit (1.4 MB for the VAD,
1.6 MB for the noise canceller), so they are not vendored: they are fetched from
the Hugging Face repositories below on first use, **pinned to the revision this
plugin was tested against**, and cached by `huggingface_hub` as usual. The
repositories are private: the machine needs an `HF_TOKEN` that can read them.

The pin is not decoration. The VAD's default operating point (threshold 0.85,
filter 0.35) was chosen on development data *for these weights*; a newer upload
can be calibrated differently, and the wrong threshold fails silently (a VAD
that fires too often, or late). Bump `VAD_REVISION` and `OPERATING_POINT`
together, never one without the other.

A local file overrides the download: pass `model_path=` / `onnx_file_path=`, or
set `CLEARLINE_VAD_ONNX_PATH` / `CLEARLINE_NC_ONNX_PATH`.
"""

from __future__ import annotations

import os

VAD_REPO = "Scicom-intl/Clearline-VAD"
NC_REPO = "Scicom-intl/Clearline-NC"
VAD_REVISION = "d6fab67a78ee7b5b71c1f58585a39cc8ed6461e3"
NC_REVISION = "af3437b9983fbff043f623ed548797795f777f30"

VAD_FILES = {
    ("release", 512): "clearline_vad.onnx",
    ("release", 256): "clearline_vad_256.onnx",
    ("distilled", 512): "clearline_vad_distilled.onnx",
    ("distilled", 256): "clearline_vad_distilled_256.onnx",
}
"""(variant, window samples) -> file. Same weights per variant; the window is how
many samples one call takes (512 = 32 ms, silero's cadence; 256 = 16 ms)."""

OPERATING_POINT = {"activation_threshold": 0.85, "smoothing_alpha": 0.35}
"""Chosen on the development sets (AMI dev, VoxConverse dev) for both variants and
both windows at `VAD_REVISION`: the highest mean of LiveKit onset and end-of-turn
recall at no more false triggers per hour than silero v6 at its own defaults."""

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
