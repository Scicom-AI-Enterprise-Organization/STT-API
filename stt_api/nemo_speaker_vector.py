"""
Local implementation of NemoSpeakerVector with FP16 support.

Copied from malaya-speech.
Source: malaya_speech/torch_model/nemo.py and malaya_speech/supervised/classification.py

Uses local nemo_featurization (copied and modified to support FP16) instead of malaya_speech's version.
"""

import os

import torch
import yaml
import numpy as np
import torch.nn as nn
from typing import List, Tuple
from app.nemo_featurization import AudioToMelSpectrogramPreprocessor
from app.nemo import conv_asr
from app.nemo.conv_asr import SpeakerDecoder
from app.torch_utils import to_tensor_cuda, to_numpy
from app.huggingface import download_files

def sequence_1d(
    seq, maxlen=None, padding: str = 'post', pad_int=0, return_len=False
):
    """
    padding sequence of 1d to become 2d array.

    Parameters
    ----------
    seq: List[List[int]]
    maxlen: int, optional (default=None)
        If None, will calculate max length in the function.
    padding: str, optional (default='post')
        If `pre`, will add 0 on the starting side, else add 0 on the end side.
    pad_int, int, optional (default=0)
        padding value.

    Returns
    --------
    result: np.array
    """
    if padding not in ['post', 'pre']:
        raise ValueError('padding only supported [`post`, `pre`]')

    if not maxlen:
        maxlen = max([len(s) for s in seq])

    # Fast path for numpy input, which is what both callers here pass (audio).
    #
    # The generic path below round-trips every sample through Python: `.tolist()`
    # turns a 48,000-sample array into 48,000 Python floats, list concatenation
    # copies them again, and `np.array` re-parses the lot. At batch 16 that is
    # ~768k objects and measured **50 ms per batch** against 8 ms of actual GPU
    # work — the padding, not the model, was the bottleneck. Filling a
    # preallocated array is a memcpy per row.
    if all(isinstance(s, np.ndarray) and s.ndim == 1 for s in seq):
        lengths = [len(s) for s in seq]
        dtype = seq[0].dtype if seq else np.float32
        out = np.full((len(seq), maxlen), pad_int, dtype=dtype)
        for i, s in enumerate(seq):
            n = min(len(s), maxlen)
            if padding == 'post':
                out[i, :n] = s[:n]
            else:
                out[i, maxlen - n:] = s[:n]
        return (out, lengths) if return_len else out

    padded_seqs, length = [], []
    for s in seq:
        if isinstance(s, np.ndarray):
            s = s.tolist()
        if padding == 'post':
            padded_seqs.append(s + [pad_int] * (maxlen - len(s)))
        if padding == 'pre':
            padded_seqs.append([pad_int] * (maxlen - len(s)) + s)
        length.append(len(s))
    if return_len:
        return np.array(padded_seqs), length
    return np.array(padded_seqs)

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

        # Reusable pinned staging buffers for prep_batch; grown on demand and
        # never shrunk, so a steady-state workload allocates page-locked memory
        # exactly once. See prep_batch for why reuse is safe here.
        self._pin_buf = None
        self._pin_len = None
        # Off by default — see prep_batch for the measurement.
        self._pin_memory = os.environ.get("SPEAKER_PIN_MEMORY", "0") == "1"

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
        inputs = np.ascontiguousarray(inputs, dtype=np.float32)

        # Pageable by default, and that is a measurement, not an oversight.
        #
        # Pinned (page-locked) staging exists to make the host-to-device copy
        # faster and overlappable. Here the H2D transfer of a 16x48000 batch is
        # **0.07 ms** — there is nothing to accelerate. Meanwhile copying 3 MB
        # *into* the pinned buffer measured **10.4 ms** on this host (~280 MB/s,
        # far below a normal memcpy), so the staging cost is 40x the transfer it
        # was supposed to optimise. End to end: pinned 10.4 ms, pageable 0.25 ms.
        #
        # `non_blocking=True` on a pageable tensor silently degrades to a
        # synchronous copy, which is fine at 0.25 ms — the CUDA-stream overlap in
        # `extract_embeddings_batched` was buying back a quarter-millisecond at a
        # ten-millisecond price.
        #
        # Set SPEAKER_PIN_MEMORY=1 to restore pinning on a host where page-locked
        # writes are fast; measure before assuming yours is.
        inputs_tensor = torch.from_numpy(inputs)
        lengths_tensor = torch.tensor(lengths, dtype=torch.float32)

        if not self._pin_memory:
            return inputs_tensor, lengths_tensor

        need = inputs.shape
        buf = self._pin_buf
        if buf is None or buf.shape[0] < need[0] or buf.shape[1] < need[1]:
            buf = torch.empty(need, dtype=torch.float32).pin_memory()
            self._pin_buf = buf
        # Safe to reuse across batches: `extract_embeddings_batched` synchronises
        # `compute_stream` (which waits on `h2d_stream`) before preparing the
        # next batch, so the previous copy has always landed.
        inputs_pinned = buf[: need[0], : need[1]]
        inputs_pinned.copy_(inputs_tensor)

        lbuf = self._pin_len
        if lbuf is None or lbuf.shape[0] < len(lengths):
            lbuf = torch.empty(len(lengths), dtype=torch.float32).pin_memory()
            self._pin_len = lbuf
        lengths_pinned = lbuf[: len(lengths)]
        lengths_pinned.copy_(lengths_tensor)

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
