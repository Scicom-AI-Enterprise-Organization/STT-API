"""
Local implementation of NemoSpeakerVector with FP16 support.

Copied from malaya-speech.
Source: malaya_speech/torch_model/nemo.py and malaya_speech/supervised/classification.py

Uses local nemo_featurization (copied and modified to support FP16) instead of malaya_speech's version.
"""

import torch
import yaml
import numpy as np
import torch.nn as nn
from typing import List, Tuple
from references.malaya_speech.utils.padding import sequence_1d
from app.nemo_featurization import AudioToMelSpectrogramPreprocessor
from references.malaya_speech.nemo import conv_asr
from references.malaya_speech.nemo.conv_asr import SpeakerDecoder
from app.torch_utils import to_tensor_cuda, to_numpy
from app.huggingface import download_files


class SpeakerVector(torch.nn.Module):
    """
    Speaker embedding model using NeMo architecture.

    Copied from malaya_speech.torch_model.nemo.SpeakerVector
    """

    def __init__(self, config, pth, model, name):
        super().__init__()

        with open(config) as stream:
            try:
                d = yaml.safe_load(stream)
            except yaml.YAMLError:
                raise ValueError("invalid yaml")

        preprocessor = d["preprocessor"].copy()
        preprocessor.pop("_target_")

        encoder = d["encoder"].copy()
        encoder_target = encoder.pop("_target_").split(".")[-1]

        decoder = d["decoder"].copy()
        decoder.pop("_target_")

        self.preprocessor = AudioToMelSpectrogramPreprocessor(**preprocessor)
        self.encoder = getattr(conv_asr, encoder_target)(**encoder)
        self.decoder = SpeakerDecoder(**decoder)

        self.load_state_dict(torch.load(pth, map_location="cpu"))

        self.__model__ = model
        self.__name__ = name

        self._is_half = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def half(self):
        """
        Override half() to only convert encoder and decoder, not preprocessor
        Preprocessor stays FP32 (STFT requirement)
        BatchNorm layers need special handling
        """
        self.encoder = self.encoder.half()
        self.decoder = self.decoder.half()

        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.float()
        for module in self.decoder.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.float()

        self._is_half = True
        return self

    def forward(self, inputs):
        """
        Vectorize inputs.

        Parameters
        ----------
        inputs: List[np.array]
        """
        inputs = inputs
        cuda = next(self.parameters()).is_cuda
        inputs, lengths = sequence_1d(inputs, return_len=True)

        # preprocessor always runs in fp32 (stft)
        inputs = to_tensor_cuda(torch.Tensor(inputs.astype(np.float32)), cuda).to(
            dtype=torch.float16 if cuda else torch.float32
        )
        lengths = to_tensor_cuda(torch.Tensor(lengths), cuda).to(
            dtype=torch.float16 if cuda else torch.float32
        )

        # preprocessor output is fp32
        o_processor = self.preprocessor(inputs, lengths)

        # IMPORTANT:
        # We cannot reliably infer "model is half" by looking at the *first* encoder
        # parameter's dtype because BatchNorm layers are explicitly cast back to fp32.
        # That can cause fp32 features to be fed into fp16 conv layers -> dtype mismatch:
        #   "Input type (float) and bias type (c10::Half) should be the same"
        if self._is_half:
            o_processor = (o_processor[0].half(), o_processor[1])

        o_encoder = self.encoder(*o_processor)
        return self.decoder(*o_encoder)

    def vectorize(self, inputs):
        """
        Vectorize inputs.

        Parameters
        ----------
        inputs: List[np.array]

        Returns
        -------
        result: np.array
        """
        r = self.forward(inputs=inputs)
        return to_numpy(r[1])

    def prep_batch(self, batch: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch: pad sequences and create pinned tensors on CPU.

        Returns:
            inputs_pinned: Pinned tensor [B, T] on CPU
            lengths_pinned: Pinned tensor [B] on CPU
        """
        inputs, lengths = sequence_1d(batch, return_len=True)
        inputs_tensor = torch.Tensor(inputs.astype(np.float32))
        lengths_tensor = torch.Tensor(lengths)

        # Create pinned memory
        inputs_pinned = inputs_tensor.pin_memory()
        lengths_pinned = lengths_tensor.pin_memory()

        return inputs_pinned, lengths_pinned

    def compute_batch(
        self, inputs_gpu: torch.Tensor, lengths_gpu: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute embeddings from pre-transferred tensors.

        Args:
            inputs_gpu: [B, T] tensor on GPU
            lengths_gpu: [B] tensor on GPU

        Returns:
            batch_emb: [B, D] embeddings on GPU
        """
        # preprocessor always runs in fp32 (stft)
        o_processor = self.preprocessor(inputs_gpu, lengths_gpu)

        # IMPORTANT: Handle FP16 conversion
        if self._is_half:
            o_processor = (o_processor[0].half(), o_processor[1])

        o_encoder = self.encoder(*o_processor)
        logits, batch_emb = self.decoder(*o_encoder)
        return batch_emb

    def __call__(self, inputs):
        return self.vectorize(inputs)


def nemo_speaker_vector(model, **kwargs):
    """
    Load NeMo speaker vector model with FP16 casting.

    Copied from malaya_speech.supervised.classification.nemo_speaker_vector
    and modified to add FP16 casting support.

    Parameters
    ----------
    model: str
        Model identifier (e.g., 'huseinzol05/nemo-titanet_large')
    **kwargs: dict
        Additional arguments passed to download_files

    Returns
    -------
    model: SpeakerVector
        Model instance cast to FP16 and moved to GPU if available
    """
    s3_file = {
        "config": "model_config.yaml",
        "model": "model_weights.ckpt",
    }
    path = download_files(model, s3_file, **kwargs)

    # Create model instance
    speaker_model = SpeakerVector(
        config=path["config"],
        pth=path["model"],
        model=model,
        name="speaker-vector-nemo",
    )

    speaker_model.eval()

    if torch.cuda.is_available():
        speaker_model = speaker_model.cuda()
        # this is the part that convert weights to fp16
        speaker_model = speaker_model.half()

    return speaker_model
