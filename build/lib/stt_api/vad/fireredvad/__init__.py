from dataclasses import dataclass
from typing import List, Tuple, Union

import logging
import torch
import numpy as np
import os

logger = logging.getLogger(__name__)

from .constants import FRAME_LENGTH_SAMPLE, FRAME_PER_SECONDS
from .audio_feat import AudioFeat
from .detect_model import DetectModel
from .stream_vad_postprocessor import StreamVadPostprocessor, StreamVadFrameResult
from huggingface_hub import snapshot_download

@dataclass
class FireRedStreamVadConfig:
    use_gpu: bool = False
    smooth_window_size: int = 5
    speech_threshold: float = 0.5
    pad_start_frame : int = 5
    min_speech_frame: int = 8
    max_speech_frame: int = 2000  # 20s
    min_silence_frame: int = 20
    chunk_max_frame: int = 30000  # 300s
    def __post_init__(self):
        if self.speech_threshold < 0 or self.speech_threshold > 1:
            raise ValueError("speech_threshold must be in [0, 1]")
        if self.min_speech_frame <= 0:
            raise ValueError("min_speech_frame must be positive")


class FireRedStreamVad:
    @classmethod
    def from_pretrained(cls, model_name="FireRedTeam/FireRedVAD", config=FireRedStreamVadConfig()):
        # Feat
        folder_name = snapshot_download(repo_id=model_name)
        model_dir = os.path.join(folder_name, "Stream-VAD")
        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = AudioFeat(cmvn_path)

        # Load & Build Model
        vad_model = DetectModel.from_pretrained(model_dir)
        if config.use_gpu:
            vad_model.cuda()
        else:
            vad_model.cpu()

        # Build Postprocessor
        postprocessor = StreamVadPostprocessor(
            config.smooth_window_size,
            config.speech_threshold,
            config.pad_start_frame,
            config.min_speech_frame,
            config.max_speech_frame,
            config.min_silence_frame)
        return cls(feat_extractor, vad_model, postprocessor, config)

    def __init__(self, audio_feat, vad_model, postprocessor, config):
        self.audio_feat = audio_feat
        self.vad_model = vad_model
        self.postprocessor = postprocessor
        self.config = config
        self.model_caches = None

    def reset(self):
        self.model_caches = None
        self.audio_feat.reset()
        self.postprocessor.reset()

    @classmethod
    def results_to_timestamps(cls, results):
        results = sorted(results, key=lambda r: r.frame_idx)
        # Get frame index (0-based)
        frame_timestamps = []
        start, end = -1, -1
        for r in results:
            if r.is_speech_start:
                if start != -1: logger.warning("start should be -1")
                start = max(0, r.speech_start_frame - 1)
                end = -1
            elif r.is_speech_end:
                assert end == -1
                end = max(0, r.speech_end_frame - 1)
                frame_timestamps.append((start, end))
                start, end = -1, -1
        if start != -1:
            assert end == -1
            end = results[-1].frame_idx - 1
            frame_timestamps.append((start, end))
        # Convert to seconds
        timestamps = []
        for s, e in frame_timestamps:
            timestamps.append((s/FRAME_PER_SECONDS, e/FRAME_PER_SECONDS))
        return timestamps