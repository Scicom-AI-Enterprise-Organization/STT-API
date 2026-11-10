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
the model trained to keep noise 20 dB down rather than remove it, with 70% of the
input mixed back, chosen on development data. On the benchmark's harness (streaming
output flushed and aligned) over all 824 VoiceBank test utterances, Whisper large-v3
scores 1.55% WER after it against 1.45% unprocessed and GTCRN's 2.65% (wideband), and
2.90% against 2.95% and 4.26% through a G.711 phone path. "quality": for a human
listener, a larger graph (256 units with per-bin levels and gains) distilled from a
GPU model (Scicom-intl/Clearline-NC-GPU) that removes background noise and keeps every
voice: VoiceBank+DEMAND PESQ 2.575 and DNS 2020 synthetic 2.389, against GTCRN's 2.535
and 2.293, at about half GTCRN's CPU. Neither mode is trained on VoiceBank, VCTK or
DEMAND.

The model file is one of two graphs of the same function: the fused graph (two fifths
of the operators; its first input is named "ri"), which the pinned revision ships, or the
plain one (first input "spec"). The samples around the model sit in arrays written in
place: the first version joined a new array for every buffer on every frame, and at any
room rate but 16 kHz it never let go of the 16 kHz output it had already resampled, so
a call's frames grew slower the longer it ran. The output is the first version's,
sample for sample.
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
_MIX_BACK = {"asr": 0.7, "quality": 0.0}
_PRIME_16K = _HOP
_NATIVE_SLACK_MS = 5
_MAX_PRIME_ROUNDS = 40


_INV = np.float32(
    1.0 / 32768.0
)  # int16 -> [-1, 1): a power of two, so the product is the quotient exactly
_EMPTY = np.zeros(0, dtype=np.float32)


class _Fifo:
    """float32 samples in at the end, out at the front, in one array."""

    __slots__ = ("a", "n")

    def __init__(self, primed: int = 0, cap: int = 4096) -> None:
        self.a = np.zeros(max(cap, primed), dtype=np.float32)
        self.n = primed  # samples held (the first `primed` are zeros)

    def room(self, k: int) -> np.ndarray:
        """The next k samples to write; `self.n += k` once they are written."""
        if self.n + k > self.a.size:
            a = np.zeros(max(2 * self.a.size, self.n + k), dtype=np.float32)
            a[: self.n] = self.a[: self.n]
            self.a = a
        return self.a[self.n : self.n + k]

    def push(self, x: np.ndarray) -> None:
        self.room(x.size)[:] = x
        self.n += x.size

    def push_int16(self, pcm: np.ndarray) -> None:
        np.multiply(pcm, _INV, out=self.room(pcm.size))
        self.n += pcm.size

    def drop(self, k: int) -> None:
        rest = self.n - k
        if rest > 0:
            self.a[:rest] = self.a[k : self.n]
        self.n = max(rest, 0)

    def take(self, k: int) -> np.ndarray:
        head = self.a[:k].copy()
        self.drop(k)
        return head


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
                0.7 in "asr" mode, 0 in "quality"). The noisy input under the enhanced
                signal is what keeps an enhancer from costing a recogniser words: its
                artifacts, not the noise left in, are what cost them.
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
        self._fast = (
            ins[0].name == "ri"
        )  # the fused graph: each hop's spectrum as a real row and an imaginary row
        self._state_names = [i.name for i in ins[1:]]
        meta = self._sess.get_modelmeta().custom_metadata_map
        self._max_hops = (
            int(meta.get("max_hops", "1")) if self._fast else 0
        )  # hops one call of that graph takes
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
        self._prev = np.zeros(
            _N_FFT - _HOP, dtype=np.float32
        )  # the previous hop: the first half of the next frame
        self._tail = np.zeros(
            _HOP, dtype=np.float32
        )  # the second half of the last synthesis frame, still to be added
        self._frames: dict[
            int, np.ndarray
        ] = {}  # work arrays [T, 512], one per number of hops in a call
        self._in16 = _Fifo()
        self._out16 = _Fifo(primed=_PRIME_16K)
        self._native_rate: int | None = None
        self._down: rtc.AudioResampler | None = None
        self._up: rtc.AudioResampler | None = None
        self._out_native = _Fifo()
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
            self._in16.push_int16(np.frombuffer(frame.data, dtype=np.int16, count=n))
            self._out16.push(self._run_hops())
            y = self._take(n, native=False)
        else:
            self._ensure_resamplers(rate)
            self._feed_down(frame)
            self._emit_up(self._run_hops())
            y = self._take(n, native=True)
        pcm_out = np.clip(np.rint(y * 32768.0), -32768, 32767).astype(np.int16)
        return rtc.AudioFrame(pcm_out.tobytes(), rate, channels, n)

    def _run_hops(self) -> np.ndarray:
        """Every whole hop in the input buffer, through one model call: the 16 kHz samples they produce."""
        T = self._in16.n // _HOP
        if T == 0:
            return _EMPTY
        fr = self._frames.get(T)
        if fr is None:
            fr = self._frames[T] = np.empty((T, _N_FFT), dtype=np.float32)
        fr[:, _HOP:] = self._in16.a[: T * _HOP].reshape(
            T, _HOP
        )  # frame t: the hop before it, then its own
        fr[0, :_HOP] = self._prev
        if T > 1:
            fr[1:, :_HOP] = fr[:-1, _HOP:]
        self._prev[:] = fr[-1, _HOP:]
        self._in16.drop(T * _HOP)
        spec = np.fft.rfft(fr * self._win, axis=-1)  # [T, 257]
        if self._fast:
            ri = np.empty((T, 2, spec.shape[1]), dtype=np.float32)
            ri[:, 0], ri[:, 1] = spec.real, spec.imag
            feed, names, outs = self._state, self._state_names, []
            for a in range(0, T, self._max_hops):
                feed["ri"] = ri[a : a + self._max_hops]
                out, *st = self._sess.run(None, feed)
                for k, v in zip(names, st):
                    feed[k] = v
                outs.append(out)
            del feed["ri"]
            out = outs[0] if len(outs) == 1 else np.concatenate(outs)
            enh = np.empty((T, out.shape[2]), dtype=np.complex64)
            enh.real, enh.imag = out[:, 0], out[:, 1]
        else:
            feed = {
                "spec": np.stack((spec.real, spec.imag), -1).astype(
                    np.float32, copy=False
                ),
                **self._state,
            }
            out, *st = self._sess.run(None, feed)
            self._state = dict(zip(self._state_names, st))
            enh = np.empty(out.shape[:2], dtype=np.complex64)
            enh.real, enh.imag = out[..., 0], out[..., 1]
        wave = (
            np.fft.irfft(enh, n=_N_FFT, axis=-1).astype(np.float32, copy=False)
            * self._win
        )
        if self._mix:
            wave = (1 - self._mix) * wave + self._mix * fr * self._win * self._win
        # overlap-add at half a window: each hop out is the first half of its frame over the second half of the one
        # before
        produced = np.empty((T, _HOP), dtype=np.float32)
        np.add(self._tail, wave[0, :_HOP], out=produced[0])
        if T > 1:
            np.add(wave[:-1, _HOP:], wave[1:, :_HOP], out=produced[1:])
        self._tail[:] = wave[-1, _HOP:]
        return produced.reshape(-1)

    # ----------------------------------------------------------------- rate conversion (as GTCRN's plugin)
    def _feed_down(self, data) -> None:
        for out in self._down.push(data):  # type: ignore[union-attr]
            self._in16.push_int16(
                np.frombuffer(out.data, dtype=np.int16, count=out.samples_per_channel)
            )

    def _emit_up(self, produced: np.ndarray) -> None:
        if not produced.size:
            return
        pcm = np.clip(np.rint(produced * 32768.0), -32768, 32767).astype(np.int16)
        for out in self._up.push(bytearray(pcm.tobytes())):  # type: ignore[union-attr]
            self._out_native.push_int16(
                np.frombuffer(out.data, dtype=np.int16, count=out.samples_per_channel)
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
        self._out_native = _Fifo()
        target = _HOP * rate // MODEL_SAMPLE_RATE + rate * _NATIVE_SLACK_MS // 1000
        silence = bytearray(np.zeros(rate // 100, dtype=np.int16).tobytes())
        for _ in range(_MAX_PRIME_ROUNDS):
            if self._out_native.n >= target:
                break
            self._feed_down(silence)
            self._emit_up(self._run_hops())

    def _take(self, n: int, *, native: bool) -> np.ndarray:
        buf = self._out_native if native else self._out16
        if buf.n < n:
            if native:  # the resamplers are behind: silence in front of what there is, with some slack
                rate = self._native_rate or MODEL_SAMPLE_RATE
                slack = (
                    _HOP * rate // MODEL_SAMPLE_RATE + rate * _NATIVE_SLACK_MS // 1000
                )
                have = buf.a[: buf.n].copy()
                buf.n = 0
                buf.push(np.zeros(n - have.size + slack, dtype=np.float32))
                buf.push(have)
            else:
                if not self._warned_underrun:
                    self._warned_underrun = True
                    logger.warning(
                        "Clearline-NC output underrun (%d < %d): padding", buf.n, n
                    )
                buf.push(np.zeros(n - buf.n, dtype=np.float32))
        return buf.take(n)

    def _close(self) -> None:
        self._state = {}
        self._in16, self._out16, self._out_native = _Fifo(), _Fifo(), _Fifo()
        self._down = self._up = None
