"""
Local implementation of NemoSpeakerVector with FP16 support.

Copied from malaya-speech.
Source: malaya_speech/torch_model/nemo.py and malaya_speech/supervised/classification.py

Uses local nemo_featurization (copied and modified to support FP16) instead of malaya_speech's version.
"""

from operator import length_hint
import torch
import yaml
import numpy as np
import torch.nn as nn
from references.malaya_speech.utils.padding import sequence_1d
from app.nemo_featurization import AudioToMelSpectrogramPreprocessor
from references.malaya_speech.nemo import conv_asr
from references.malaya_speech.nemo.conv_asr import SpeakerDecoder
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy
from malaya_boilerplate.huggingface import download_files


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
            except yaml.YAMLError as exc:
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
        inputs = to_tensor_cuda(torch.Tensor(inputs.astype(np.float32)), cuda)
        lengths = to_tensor_cuda(torch.Tensor(lengths), cuda)

        # preprocessor output is fp32
        o_processor = self.preprocessor(inputs, lengths)

        if next(self.encoder.parameters()).dtype == torch.float16:
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
