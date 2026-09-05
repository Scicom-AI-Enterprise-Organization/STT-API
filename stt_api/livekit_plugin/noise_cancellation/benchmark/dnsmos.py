"""
DNSMOS P.835 and P.808 — reference-free neural MOS.

This is the metric the DNS Challenge ranks on, and the only one here that works
on `dns-real`, where there is no clean reference to compare against. It is also
the one that best predicts what a human would say, which matters because PESQ and
STOI were designed for codecs and reverberation, not for the specific artefacts a
neural suppressor produces. A model can win PESQ while introducing exactly the
kind of musical noise DNSMOS-BAK is built to notice.

Three numbers from P.835, all on a 1-5 MOS scale:

    SIG   speech quality, ignoring the background
    BAK   background intrusiveness — how well the noise went away
    OVRL  overall

plus a separate P.808 single-opinion score.

The preprocessing is transcribed from Microsoft's `dnsmos_local.py` rather than
re-derived, down to the 9.01-second window, the one-second hop, the repeat-padding
of short clips and the `[:-160]` trim before the P.808 mel spectrogram. These are
not arbitrary: the networks were trained on exactly this framing, and a plausible
substitute (say, zero-padding instead of repeat-padding) shifts the scores while
still looking perfectly reasonable. The polynomial mappings are likewise the
published coefficients — the raw network outputs are not calibrated MOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = ["DNSMOS", "DnsmosScore"]

_RATE = 16000
_INPUT_LENGTH = 9.01
_LEN_SAMPLES = int(_INPUT_LENGTH * _RATE)  # 144160, matching the ONNX input

_REPO = "anchor-flux/dnsmos-onnx"
_P835 = "sig_bak_ovr.onnx"
_P808 = "model_v8.onnx"

# Published calibration from DNS-Challenge/DNSMOS/dnsmos_local.py (non-personalised).
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


@dataclass(frozen=True)
class DnsmosScore:
    sig: float
    bak: float
    ovrl: float
    p808: float


@lru_cache(maxsize=1)
def _sessions():
    import onnxruntime as ort

    from .corpus import fetch_model

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return (
        ort.InferenceSession(
            str(fetch_model(_REPO, _P835)), opts, providers=["CPUExecutionProvider"]
        ),
        ort.InferenceSession(
            str(fetch_model(_REPO, _P808)), opts, providers=["CPUExecutionProvider"]
        ),
    )


class DNSMOS:
    """
    Callable scorer. Sessions are process-wide and thread-count-pinned, so scoring
    never competes with the enhancer being timed for cores.
    """

    def __init__(self) -> None:
        self._p835, self._p808 = _sessions()
        self._p808_frames = int(self._p808.get_inputs()[0].shape[1])

    def _melspec(self, seg: np.ndarray) -> np.ndarray:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=seg, sr=_RATE, n_fft=321, hop_length=160, n_mels=120
        )
        mel = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
        mel = mel.T
        # librosa's framing has shifted across versions; the graph's time axis is
        # fixed. Trim or edge-pad to the exact width the model was exported with
        # rather than letting a version bump turn into a shape error at run time.
        if mel.shape[0] > self._p808_frames:
            mel = mel[: self._p808_frames]
        elif mel.shape[0] < self._p808_frames:
            mel = np.pad(mel, ((0, self._p808_frames - mel.shape[0]), (0, 0)), mode="edge")
        return mel.astype(np.float32)

    def __call__(self, x: np.ndarray, rate: int = _RATE) -> DnsmosScore:
        if rate != _RATE:
            from .audio import StreamResampler

            x = StreamResampler(rate, _RATE).push(x)
        audio = np.asarray(x, dtype=np.float32)

        # Repeat-pad, exactly as upstream does. Zero-padding a short clip would
        # hand BAK nine seconds of digital silence to admire and inflate it.
        while len(audio) < _LEN_SAMPLES:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / _RATE) - _INPUT_LENGTH) + 1
        sig, bak, ovr, p808 = [], [], [], []
        for idx in range(max(num_hops, 1)):
            seg = audio[idx * _RATE : idx * _RATE + _LEN_SAMPLES]
            if len(seg) < _LEN_SAMPLES:
                continue
            raw = self._p835.run(None, {"input_1": seg[np.newaxis, :].astype(np.float32)})[0][0]
            mel = self._melspec(seg[:-160])[np.newaxis, :, :]
            p808.append(float(self._p808.run(None, {"input_1": mel})[0][0][0]))
            sig.append(float(_P_SIG(raw[0])))
            bak.append(float(_P_BAK(raw[1])))
            ovr.append(float(_P_OVR(raw[2])))

        if not sig:
            return DnsmosScore(float("nan"), float("nan"), float("nan"), float("nan"))
        return DnsmosScore(
            sig=float(np.mean(sig)),
            bak=float(np.mean(bak)),
            ovrl=float(np.mean(ovr)),
            p808=float(np.mean(p808)),
        )
