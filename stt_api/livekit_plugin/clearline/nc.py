"""
Clearline-NC as a LiveKit Agents audio filter: an `rtc.FrameProcessor[rtc.AudioFrame]`
for `AudioInputOptions.noise_cancellation`, a drop-in for `../noise_cancellation/`'s
`GTCRN` (the same 512-point sqrt-Hann STFT and 256 hop, the same 32 ms delay at
16 kHz, the same priming and resampler choices, whose reasoning is in gtcrn.py).

    from stt_api.livekit_plugin import ClearlineNC

    room_io.AudioInputOptions(sample_rate=16000, noise_cancellation=lambda params: ClearlineNC())

One instance per audio stream (the recurrent state is per stream; the ONNX session
is shared). What differs from GTCRN's filter: every whole hop a frame completes goes
through ONE model call (a 50 ms frame carries three hops), and the STFT and
overlap-add of those hops are vectorised, so a frame costs one session call, not one
per hop. That, not arithmetic, is most of the 3.8x.

Two modes. "asr" (the default, for an agent whose listener is a speech recogniser):
the model trained to keep noise 20 dB down rather than remove it, with 10% of the
input mixed back; over all 824 VoiceBank test utterances it costs Whisper large-v3
1.90% WER against GTCRN's 2.55% (wideband) and 4.18% against 4.28% through a G.711
phone path. "quality": the model trained to remove all the noise it can, for a human
listener (VoiceBank+DEMAND PESQ 2.633 against GTCRN's 2.535). Neither mode is
trained on VoiceBank, VCTK or DEMAND.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from livekit import rtc
from livekit.rtc.audio_resampler import AudioResamplerQuality

from . import hub

logger = logging.getLogger(__name__)

__all__ = ["MODEL_SAMPLE_RATE", "ClearlineNC"]

MODEL_SAMPLE_RATE = 16000
_N_FFT, _HOP = 512, 256
_MIX_BACK = {"asr": 0.1, "quality": 0.0}
_PRIME_16K = _HOP
_NATIVE_SLACK_MS = 5
_MAX_PRIME_ROUNDS = 40


def _sqrt_hann(n: int) -> np.ndarray:
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)
    return np.sqrt(hann).astype(np.float32)


@lru_cache(maxsize=4)
def _load_session(path: str, num_threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = num_threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


class ClearlineNC(rtc.FrameProcessor[rtc.AudioFrame]):
    def __init__(
        self,
        *,
        enabled: bool = True,
        mode: str = "asr",
        model_path: str | None = None,
        num_threads: int = 1,
        mix_back: float | None = None,
        revision: str | None = None,
    ) -> None:
        """
        Args:
            enabled: False passes every frame through untouched.
            mode: "asr" (default) or "quality"; see the module docstring.
            model_path: a local graph instead of the pinned download (also
                `CLEARLINE_NC_ONNX_PATH`).
            num_threads: ONNX Runtime threads; 1 is right for this size.
            mix_back: share of the unprocessed input added back (0 = none; default
                0.1 in "asr" mode, 0 in "quality"). A little of the noisy input under
                the enhanced signal is what keeps an enhancer from costing ASR accuracy.
            revision: a Hugging Face revision instead of the pinned one.

        Raises at construction, not on the first frame, if the model cannot be
        loaded: a filter that silently passes audio through looks like it works.
        """
        if mode not in _MIX_BACK:
            raise ValueError(f"mode must be one of {sorted(_MIX_BACK)}, got {mode!r}")
        self._enabled = enabled
        self._mode = mode
        self._mix = float(_MIX_BACK[mode] if mix_back is None else mix_back)
        self._sess = _load_session(
            hub.nc_model_path(mode, model_path, revision), num_threads
        )
        ins = self._sess.get_inputs()
        self._state_names = [i.name for i in ins[1:]]
        meta = self._sess.get_modelmeta().custom_metadata_map
        init = {
            "band_m": float(meta.get("band_m0", -60.0)),
            "df_m": float(meta.get("df_m0", 1e-3)),
        }
        self._state = {
            i.name: np.full(
                [int(d) for d in i.shape], init.get(i.name, 0.0), np.float32
            )
            for i in ins[1:]
        }
        self._win = _sqrt_hann(_N_FFT)
        self._analysis = np.zeros(_N_FFT - _HOP, dtype=np.float32)  # the previous hop
        self._ola = np.zeros(_N_FFT, dtype=np.float32)
        self._in16 = np.zeros(0, dtype=np.float32)
        self._out16 = np.zeros(_PRIME_16K, dtype=np.float32)
        self._native_rate: int | None = None
        self._down: rtc.AudioResampler | None = None
        self._up: rtc.AudioResampler | None = None
        self._out_native = np.zeros(0, dtype=np.float32)
        self._warned_channels = False
        self._warned_underrun = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        n, channels, rate = (
            frame.samples_per_channel,
            frame.num_channels,
            frame.sample_rate,
        )
        if channels != 1:
            if not self._warned_channels:
                self._warned_channels = True
                logger.warning(
                    "Clearline-NC expects mono audio, got %d channels: passing through",
                    channels,
                )
            return frame
        if rate == MODEL_SAMPLE_RATE:
            pcm = np.frombuffer(frame.data, dtype=np.int16, count=n)
            self._in16 = np.concatenate((self._in16, pcm.astype(np.float32) / 32768.0))
            self._run_hops()
            y = self._take(n, native=False)
        else:
            self._ensure_resamplers(rate)
            self._feed_down(frame)
            self._emit_up(self._run_hops())
            y = self._take(n, native=True)
        pcm_out = np.clip(np.rint(y * 32768.0), -32768, 32767).astype(np.int16)
        return rtc.AudioFrame(pcm_out.tobytes(), rate, channels, n)

    def _run_hops(self) -> np.ndarray:
        """Every whole hop in the input buffer, through one model call."""
        T = self._in16.size // _HOP
        if T == 0:
            return np.zeros(0, dtype=np.float32)
        new, self._in16 = self._in16[: T * _HOP], self._in16[T * _HOP :]
        buf = np.concatenate((self._analysis, new))  # 256 + T * 256
        idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(T)[:, None]
        spec = np.fft.rfft(buf[idx] * self._win, axis=-1)  # [T, 257]
        feed = {
            "spec": np.stack((spec.real, spec.imag), -1).astype(np.float32),
            **self._state,
        }
        out, *st = self._sess.run(None, feed)
        self._state = dict(zip(self._state_names, st))
        wave = (
            np.fft.irfft(out[..., 0] + 1j * out[..., 1], n=_N_FFT, axis=-1).astype(
                np.float32
            )
            * self._win
        )
        if self._mix:
            wave = (1 - self._mix) * wave + self._mix * buf[idx] * self._win * self._win
        produced = np.empty(T * _HOP, dtype=np.float32)
        ola = self._ola
        for t in range(T):  # overlap-add, one hop out per frame
            ola += wave[t]
            produced[t * _HOP : (t + 1) * _HOP] = ola[:_HOP]
            ola = np.concatenate((ola[_HOP:], np.zeros(_HOP, dtype=np.float32)))
        self._ola = ola
        self._analysis = buf[-(_N_FFT - _HOP) :]
        self._out16 = np.concatenate((self._out16, produced))
        return produced

    # ----------------------------------------------------------------- rate conversion (as GTCRN's plugin)
    def _feed_down(self, data) -> None:
        for out in self._down.push(data):  # type: ignore[union-attr]
            chunk = np.frombuffer(
                out.data, dtype=np.int16, count=out.samples_per_channel
            )
            self._in16 = np.concatenate(
                (self._in16, chunk.astype(np.float32) / 32768.0)
            )

    def _emit_up(self, produced: np.ndarray) -> None:
        if not produced.size:
            return
        pcm = np.clip(np.rint(produced * 32768.0), -32768, 32767).astype(np.int16)
        for out in self._up.push(bytearray(pcm.tobytes())):  # type: ignore[union-attr]
            chunk = np.frombuffer(
                out.data, dtype=np.int16, count=out.samples_per_channel
            )
            self._out_native = np.concatenate(
                (self._out_native, chunk.astype(np.float32) / 32768.0)
            )

    def _ensure_resamplers(self, rate: int) -> None:
        if self._native_rate == rate:
            return
        self._native_rate = rate
        self._down = rtc.AudioResampler(
            rate, MODEL_SAMPLE_RATE, num_channels=1, quality=AudioResamplerQuality.LOW
        )
        self._up = rtc.AudioResampler(
            MODEL_SAMPLE_RATE, rate, num_channels=1, quality=AudioResamplerQuality.QUICK
        )
        self._out_native = np.zeros(0, dtype=np.float32)
        target = _HOP * rate // MODEL_SAMPLE_RATE + rate * _NATIVE_SLACK_MS // 1000
        silence = bytearray(np.zeros(rate // 100, dtype=np.int16).tobytes())
        for _ in range(_MAX_PRIME_ROUNDS):
            if self._out_native.size >= target:
                break
            self._feed_down(silence)
            self._emit_up(self._run_hops())

    def _take(self, n: int, *, native: bool) -> np.ndarray:
        buf = self._out_native if native else self._out16
        if buf.size < n:
            if native:
                rate = self._native_rate or MODEL_SAMPLE_RATE
                slack = (
                    _HOP * rate // MODEL_SAMPLE_RATE + rate * _NATIVE_SLACK_MS // 1000
                )
                buf = np.concatenate(
                    (np.zeros(n - buf.size + slack, dtype=np.float32), buf)
                )
            else:
                if not self._warned_underrun:
                    self._warned_underrun = True
                    logger.warning(
                        "Clearline-NC output underrun (%d < %d): padding", buf.size, n
                    )
                buf = np.concatenate((buf, np.zeros(n - buf.size, dtype=np.float32)))
        head, rest = buf[:n], buf[n:]
        if native:
            self._out_native = rest
        else:
            self._out16 = rest
        return head

    def _close(self) -> None:
        self._state = {}
        self._in16 = self._out16 = self._out_native = np.zeros(0, dtype=np.float32)
        self._down = self._up = None
