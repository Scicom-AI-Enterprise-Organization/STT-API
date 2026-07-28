"""
GTCRN speech enhancement as a LiveKit Agents audio filter.

LiveKit's own noise cancellation (`livekit-plugins-noise-cancellation`, i.e. Krisp
NC/BVC) authorises against LiveKit Cloud. On a self-hosted server the plugin loads
its native library, fails the entitlement check, logs
`noise cancellation is not authorized (404)` and passes audio through untouched.

The hook it plugs into is *not* Cloud-gated. `AudioInputOptions.noise_cancellation`
accepts either those gated `rtc.NoiseCancellationOptions` or any
`rtc.FrameProcessor[rtc.AudioFrame]` — a plain Python object that runs entirely
inside the agent process, with nothing to authorise. This module is that second
form, backed by a 48.2 K-parameter ONNX model.

Where it earns its keep is inbound SIP audio. A browser microphone already arrives
processed, because livekit-client enables WebRTC's `noiseSuppression`,
`echoCancellation` and `voiceIsolation` by default — but a phone call reaches the
agent with none of that.

Model: GTCRN (https://github.com/Xiaobin-Rong/gtcrn, MIT), 16 kHz, 512-point STFT
with a 256-sample hop, streamed one hop at a time through three recurrent caches.

Measured on this machine over 3.3 s of speech, 48.2 K parameters at 33 MMACs/s:

    cost          3 % of one core per stream (RTF 0.031, ~1.6 ms per 50 ms frame)
    added delay   32 ms at 16 kHz · 56 ms at 24/48 kHz · 109 ms at 8 kHz
    denoising     +9 to +10 dB SNR at 3-10 dB input SNR
    noise floor   -25 to -39 dB during pauses
    clean speech  30 dB fidelity, level unchanged (+0.05 dB)

Run it at 16 kHz — `AudioInputOptions(sample_rate=16000)`. Any other rate adds two
resampler stages whose latency dwarfs the model's, and 16 kHz is what the STT and
the VAD want anyway.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from livekit import rtc
from livekit.rtc.audio_resampler import AudioResamplerQuality

logger = logging.getLogger(__name__)

_DOWN_QUALITY = AudioResamplerQuality.LOW
_UP_QUALITY = AudioResamplerQuality.QUICK

MODEL_SAMPLE_RATE = 16000
"""The only rate the model runs at. Other rates are resampled in and back out."""

_N_FFT = 512
_HOP = 256
_MODEL_FILE = "gtcrn_simple.onnx"

# Zeros pre-loaded into the output buffer so a frame can always be answered with a
# frame of the same length. Input arrives in whatever size the SFU delivers
# (`frame_size_ms=50` by default) while the model consumes fixed 256-sample hops, so
# without priming the early frames would come up short. Feeding N samples produces
# floor(N / HOP) * HOP, i.e. a standing shortfall of strictly less than one hop, so
# one hop of priming is exactly enough and never runs dry afterwards.
_PRIME_16K = _HOP

# End-to-end delay at 16 kHz is two hops, 32 ms: one from the priming above, one
# because the first analysis window is three-quarters zeros and its output hop
# describes the silence *before* the first real sample.
_NATIVE_SLACK_MS = 5
"""Extra margin on top of one hop when priming the resampled path."""

_MAX_PRIME_ROUNDS = 40
"""Cap on the 10 ms silence pushes used to prime the resampled path."""

_MAX_NATIVE_PRIMES = 2
"""Reactive top-ups tolerated before warning that latency is climbing."""


def _sqrt_hann(n: int) -> np.ndarray:
    """
    Periodic Hann (what `torch.hann_window` produces), square-rooted.

    Applied on analysis and again on synthesis it multiplies back to a plain Hann,
    and a periodic Hann at 50 % overlap sums to exactly 1.0 — so overlap-add
    reconstructs the input sample-for-sample when the model is a no-op. Using
    `np.hanning` here instead would be a symmetric window, which does not sum to
    unity and would put a slow ripple on the output.
    """
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)
    return np.sqrt(hann).astype(np.float32)


def _model_path(explicit: str | None = None) -> str:
    path = explicit or os.environ.get("GTCRN_ONNX_PATH")
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"GTCRN model not found at {path}")
        return path
    bundled = Path(__file__).parent / "resources" / _MODEL_FILE
    if not bundled.exists():
        raise FileNotFoundError(
            f"GTCRN model missing from the package at {bundled}. Reinstall stt-api, or "
            f"point GTCRN_ONNX_PATH at a copy of {_MODEL_FILE}."
        )
    return str(bundled)


@lru_cache(maxsize=4)
def _load_session(path: str, num_threads: int):
    """
    One InferenceSession per (model, thread count), shared by every stream.

    The weights are ~500 KB and ORT sessions are safe to call concurrently, so
    there is no reason for each participant to hold its own copy. The per-stream
    state that genuinely cannot be shared — the three recurrent caches and the
    STFT buffers — lives on the `GTCRN` instance instead.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    # 33 MMACs/s of work: thread pools cost more than they save, and an agent pod
    # runs one of these per concurrent session.
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = num_threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


class GTCRN(rtc.FrameProcessor[rtc.AudioFrame]):
    """
    Self-hosted noise cancellation for an agent's audio input.

        from stt_api.livekit_plugin.noise_cancellation import GTCRN

        await session.start(
            agent=MyAgent(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    # Matching the model's rate skips resampling entirely.
                    sample_rate=16000,
                    # A selector, so every participant stream gets its own caches.
                    noise_cancellation=lambda params: GTCRN(),
                ),
            ),
        )

    One instance per audio stream — the recurrent caches are per-stream state and
    sharing an instance across participants would cross-contaminate them.

    A missing or unreadable model raises here, at session start, rather than
    degrading to silent passthrough. That is deliberate: a filter that quietly
    does nothing is the exact failure mode of the Cloud plugin it replaces.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        model_path: str | None = None,
        num_threads: int = 1,
    ) -> None:
        self._enabled = enabled
        self._sess = _load_session(_model_path(model_path), num_threads)

        input_names = [i.name for i in self._sess.get_inputs()]
        # First input is the spectrum, the rest are recurrent caches. Shapes are
        # read from the model rather than hardcoded, so a re-exported GTCRN with
        # different cache dimensions still loads.
        self._spec_name = input_names[0]
        self._cache_names = input_names[1:]
        self._caches = {
            i.name: np.zeros([int(d) for d in i.shape], dtype=np.float32)
            for i in self._sess.get_inputs()[1:]
        }

        self._win = _sqrt_hann(_N_FFT)
        self._analysis = np.zeros(_N_FFT, dtype=np.float32)
        self._ola = np.zeros(_N_FFT, dtype=np.float32)
        self._in16 = np.zeros(0, dtype=np.float32)
        self._out16 = np.zeros(_PRIME_16K, dtype=np.float32)

        # Built on the first frame, once the stream's actual rate is known.
        self._native_rate: int | None = None
        self._down: rtc.AudioResampler | None = None
        self._up: rtc.AudioResampler | None = None
        self._out_native = np.zeros(0, dtype=np.float32)
        self._native_primes = 0

        self._warned_channels = False
        self._warned_underrun = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        n = frame.samples_per_channel
        channels = frame.num_channels
        rate = frame.sample_rate

        if channels != 1:
            # Agent input is mono (`AudioInputOptions.num_channels` defaults to 1).
            # Downmixing and re-fanning would silently change the track's content,
            # so leave anything else alone and say so once.
            if not self._warned_channels:
                self._warned_channels = True
                logger.warning(
                    "GTCRN expects mono audio, got %d channels — passing through unfiltered",
                    channels,
                )
            return frame

        if rate == MODEL_SAMPLE_RATE:
            pcm = np.frombuffer(frame.data, dtype=np.int16, count=n)
            self._in16 = np.concatenate((self._in16, pcm.astype(np.float32) / 32768.0))
            self._run_hops()
            y = self._take(n, at_native_rate=False)
        else:
            self._ensure_resamplers(rate)
            self._feed_down(frame)
            self._emit_up(self._run_hops())
            y = self._take(n, at_native_rate=True)

        pcm_out = np.clip(np.rint(y * 32768.0), -32768, 32767).astype(np.int16)
        return rtc.AudioFrame(pcm_out.tobytes(), rate, channels, n)

    def _feed_down(self, data: rtc.AudioFrame | bytearray) -> None:
        """Native rate -> 16 kHz, into the model's input buffer."""
        for out in self._down.push(data):  # type: ignore[union-attr]
            chunk = np.frombuffer(out.data, dtype=np.int16, count=out.samples_per_channel)
            self._in16 = np.concatenate((self._in16, chunk.astype(np.float32) / 32768.0))

    def _emit_up(self, produced: np.ndarray) -> None:
        """16 kHz -> native rate, into the buffer `_take` answers frames from."""
        if not produced.size:
            return
        pcm = np.clip(np.rint(produced * 32768.0), -32768, 32767).astype(np.int16)
        for out in self._up.push(bytearray(pcm.tobytes())):  # type: ignore[union-attr]
            chunk = np.frombuffer(out.data, dtype=np.int16, count=out.samples_per_channel)
            self._out_native = np.concatenate(
                (self._out_native, chunk.astype(np.float32) / 32768.0)
            )

    def _ensure_resamplers(self, rate: int) -> None:
        if self._native_rate == rate:
            return
        self._native_rate = rate
        # Resampler quality is a latency decision here, not an audio-fidelity one.
        # soxr's default MEDIUM costs 35.8 ms of latency *each way* (91.8 ms at
        # 8 kHz), which measured as 96 ms and 246 ms end-to-end — far too much to
        # put in front of an STT.
        #
        # Down needs a real anti-alias filter: at QUICK, 48 kHz -> 16 kHz folds
        # everything above 8 kHz back into the band the model is trying to clean.
        # LOW costs 12.5 ms and filters properly. Up cannot alias — the signal is
        # already band-limited to 8 kHz — so QUICK is free there.
        self._down = rtc.AudioResampler(
            rate, MODEL_SAMPLE_RATE, num_channels=1, quality=_DOWN_QUALITY
        )
        self._up = rtc.AudioResampler(
            MODEL_SAMPLE_RATE, rate, num_channels=1, quality=_UP_QUALITY
        )
        self._out_native = np.zeros(0, dtype=np.float32)
        self._native_primes = 0

        # Prime by running silence through the whole chain until the output buffer
        # holds a working surplus. Output arrives in lumps of one hop, so a frame
        # can land just before a lump and find the buffer short; carrying one lump
        # plus a little slack absorbs that for the rest of the stream.
        #
        # Doing it here, with a target derived from the rates, rather than reacting
        # to the first shortfall: reacting compounds — each round adds a frame's
        # worth of delay, and measured end-to-end latency reached 107 ms at 24 kHz
        # and never settled at 8 kHz. Silence is the right primer because it is
        # what the buffer would otherwise be padded with, and it warms the model's
        # recurrent caches on quiet input.
        target = _HOP * rate // MODEL_SAMPLE_RATE + rate * _NATIVE_SLACK_MS // 1000
        silence = bytearray(np.zeros(rate // 100, dtype=np.int16).tobytes())  # 10 ms
        for _ in range(_MAX_PRIME_ROUNDS):
            if self._out_native.size >= target:
                break
            self._feed_down(silence)
            self._emit_up(self._run_hops())

    def _run_hops(self) -> np.ndarray:
        """Consume whole hops from the input buffer; return the 16 kHz audio produced."""
        out: list[np.ndarray] = []
        while self._in16.size >= _HOP:
            hop, self._in16 = self._in16[:_HOP], self._in16[_HOP:]
            # Slide the analysis window forward. Only history is used, so the model
            # adds no lookahead delay of its own.
            self._analysis = np.concatenate((self._analysis[_HOP:], hop))

            spec = np.fft.rfft(self._analysis * self._win)
            feed = {
                self._spec_name: np.stack((spec.real, spec.imag), axis=-1)
                .reshape(1, -1, 1, 2)
                .astype(np.float32),
                **self._caches,
            }
            enh, *caches = self._sess.run(None, feed)
            # Outputs are the updated caches, in the same order as the inputs.
            self._caches = dict(zip(self._cache_names, caches))

            bins = enh.reshape(-1, 2)
            wave = np.fft.irfft(bins[:, 0] + 1j * bins[:, 1], n=_N_FFT).astype(np.float32)

            self._ola += wave * self._win
            out.append(self._ola[:_HOP].copy())
            # The leading hop is complete — the next window starts one hop later and
            # cannot contribute to it.
            self._ola = np.concatenate((self._ola[_HOP:], np.zeros(_HOP, dtype=np.float32)))

        if not out:
            return np.zeros(0, dtype=np.float32)
        produced = np.concatenate(out)
        self._out16 = np.concatenate((self._out16, produced))
        return produced

    def _take(self, n: int, *, at_native_rate: bool) -> np.ndarray:
        buf = self._out_native if at_native_rate else self._out16
        if buf.size < n:
            if at_native_rate:
                # Self-tuning priming for the resampled path. Its standing deficit
                # cannot be computed up front: soxr contributes a latency in each
                # direction, and output arrives in lumps of one hop rather than
                # smoothly, so the buffer has to cover a whole frame's consumption
                # between lumps. Prepending the shortfall plus a hop's worth of
                # slack converges after a couple of frames and then never fires
                # again. Prepending rather than appending is what makes this safe —
                # it delays the audio instead of dropping it.
                rate = self._native_rate or MODEL_SAMPLE_RATE
                slack = _HOP * rate // MODEL_SAMPLE_RATE + rate * _NATIVE_SLACK_MS // 1000
                buf = np.concatenate((np.zeros(n - buf.size + slack, dtype=np.float32), buf))
                self._native_primes += 1
                if self._native_primes > _MAX_NATIVE_PRIMES and not self._warned_underrun:
                    # Converged priming should take one or two rounds. More than that
                    # means the assumption above is wrong for this rate, and latency
                    # is climbing frame by frame.
                    self._warned_underrun = True
                    logger.warning(
                        "GTCRN re-primed %d times at %d Hz — output latency is growing; "
                        "prefer sample_rate=%d on AudioInputOptions",
                        self._native_primes,
                        rate,
                        MODEL_SAMPLE_RATE,
                    )
            else:
                # One hop of priming is provably enough at 16 kHz, so this is
                # unreachable; pad rather than hand back a short frame.
                if not self._warned_underrun:
                    self._warned_underrun = True
                    logger.warning(
                        "GTCRN output underrun (%d < %d) — padding with silence", buf.size, n
                    )
                buf = np.concatenate((buf, np.zeros(n - buf.size, dtype=np.float32)))
        head, rest = buf[:n], buf[n:]
        if at_native_rate:
            self._out_native = rest
        else:
            self._out16 = rest
        return head

    def _close(self) -> None:
        # The session is shared through the lru_cache, so it is deliberately not
        # torn down here; only this stream's state is dropped.
        self._caches = {}
        self._in16 = np.zeros(0, dtype=np.float32)
        self._out16 = np.zeros(0, dtype=np.float32)
        self._out_native = np.zeros(0, dtype=np.float32)
        self._down = None
        self._up = None
