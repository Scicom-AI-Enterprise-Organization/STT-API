"""
The candidate enhancers, behind one interface.

Every entry here is driven the same way the LiveKit audio path drives the real
filter: fixed-size frames, one at a time, in order, with no lookahead beyond
whatever the algorithm buffers internally. That is the only way the numbers mean
anything. An offline benchmark that hands a model the whole utterance measures a
model nobody can deploy — it silently grants unlimited lookahead and lets slow
models look free.

The one deliberate exception is `dfn3`, which is marked `streaming = False` and
run whole-file on purpose: it is a *quality ceiling*, not a candidate. See its
class docstring.

Rate handling is also deliberate. Enhancers are driven at the pipeline's rate and
each converts internally to whatever it needs, because that conversion is a real
cost the pipeline would pay. RNNoise and DeepFilterNet are 48 kHz models; running
them in a 16 kHz agent means two resampler stages, and hiding that would flatter
them against a model that is natively 16 kHz.
"""

from __future__ import annotations

import abc
import os
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import StreamResampler

__all__ = ["Enhancer", "available", "build", "REGISTRY"]

_CACHE = Path(
    os.environ.get(
        "NC_BENCHMARK_CACHE", Path.home() / ".cache" / "stt-api" / "nc-benchmark"
    )
)


class Enhancer(abc.ABC):
    """
    One frame in, one frame of the same length out.

    Implementations must be stateful across frames and must not peek at future
    frames. `reset()` returns the instance to its start-of-stream state so one
    object can be reused across corpus items without leaking context between
    them — a leak that would show up as an unearned quality gain on short items.
    """

    name: str = "?"
    note: str = ""
    native_rate: int = 16000
    streaming: bool = True
    generative: bool = False
    """Resynthesises the waveform rather than masking it. Waveform-comparison
    metrics (PESQ, STOI, SI-SDR) do not apply — see `generative.py`."""

    @abc.abstractmethod
    def reset(self) -> None: ...

    def process(self, frame: np.ndarray, rate: int) -> np.ndarray:
        """Streaming path: one frame, returns the same number of samples."""
        raise NotImplementedError

    def process_all(self, x: np.ndarray, rate: int) -> np.ndarray:
        """Offline path, used only when `streaming` is False."""
        raise NotImplementedError


class _HopStreamer:
    """
    Turns a fixed-hop, fixed-rate algorithm into something that survives arbitrary
    frame sizes at an arbitrary rate.

    Three jobs, all of which GTCRN's production filter also has to do and which
    every other candidate here would otherwise have to reinvent:

    1. Resample into the algorithm's native rate and back out, statefully.
    2. Buffer, because a 50 ms frame is not a whole number of the algorithm's hops.
    3. Prime the output, so a frame can always be answered with a frame of equal
       length. Feeding N samples yields floor(N / hop) * hop, a standing shortfall
       of strictly less than one hop, so one hop of priming covers it forever.

    Priming costs latency, which is real and is what the measured delay will
    report. It is not hidden.
    """

    def __init__(self, native_rate: int, hop: int, step: Callable[[np.ndarray], np.ndarray]):
        self.native_rate = native_rate
        self.hop = hop
        self.step = step
        self._rate: int | None = None
        self._down: StreamResampler | None = None
        self._up: StreamResampler | None = None
        self._in = np.zeros(0, dtype=np.float32)
        self._out = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._rate = None
        self._down = self._up = None
        self._in = np.zeros(0, dtype=np.float32)
        self._out = np.zeros(0, dtype=np.float32)

    def _ensure(self, rate: int) -> None:
        if self._rate == rate:
            return
        self._rate = rate
        self._down = StreamResampler(rate, self.native_rate)
        self._up = StreamResampler(self.native_rate, rate)
        self._out = np.zeros(0, dtype=np.float32)
        # Prime with silence rather than by padding the first frame's output:
        # silence is what the buffer would be padded with anyway, and pushing it
        # through the whole chain also settles the resamplers' internal state and
        # warms any recurrent state on quiet input, which is the condition those
        # models are least surprised by.
        #
        # Primed to exactly one hop, with no comfort margin. Every primed sample
        # becomes measured latency, and latency is one of the numbers this
        # benchmark exists to compare — padding it "to be safe" would quietly
        # penalise whichever model has the largest hop. One hop is the provable
        # bound on the steady-state shortfall; the reactive top-up in `push`
        # covers the resampler warm-up transient on top of it.
        target = -(-self.hop * rate // self.native_rate)  # ceil
        silence = np.zeros(max(1, rate // 200), dtype=np.float32)  # 5 ms
        for _ in range(100):
            if self._out.size >= target:
                break
            self._pump(silence)

    def _pump(self, frame: np.ndarray) -> None:
        self._in = np.concatenate((self._in, self._down.push(frame)))  # type: ignore[union-attr]
        produced: list[np.ndarray] = []
        while self._in.size >= self.hop:
            block, self._in = self._in[: self.hop], self._in[self.hop :]
            produced.append(self.step(block))
        if produced:
            self._out = np.concatenate((self._out, self._up.push(np.concatenate(produced))))  # type: ignore[union-attr]

    def push(self, frame: np.ndarray, rate: int) -> np.ndarray:
        self._ensure(rate)
        self._pump(frame)
        n = len(frame)
        if self._out.size < n:
            # Prepend rather than append: this delays the audio instead of
            # dropping it, which keeps the signal intact for the metrics and
            # shows up honestly as latency.
            #
            # The margin is one hop expressed *at the pipeline rate*. Using the
            # native hop directly would over-pad every model whose native rate is
            # above the pipeline's — 480 samples of RNNoise's 48 kHz hop is 30 ms
            # when prepended to a 16 kHz buffer, not the 10 ms it represents.
            margin = -(-self.hop * rate // self.native_rate)
            self._out = np.concatenate(
                (np.zeros(n - self._out.size + margin, dtype=np.float32), self._out)
            )
        head, self._out = self._out[:n], self._out[n:]
        return head


class Passthrough(Enhancer):
    """
    The control. Not a strawman — it is the number every other row has to beat.

    A denoiser that scores below this on any metric is actively harmful on that
    axis, and that happens more often than the literature suggests, particularly
    for WER: suppression that sounds cleaner to a listener regularly removes the
    exact low-energy consonant detail an acoustic model relies on.
    """

    name = "passthrough"
    note = "unprocessed input (control)"

    def reset(self) -> None:
        pass

    def process(self, frame: np.ndarray, rate: int) -> np.ndarray:
        return frame


class GTCRNEnhancer(Enhancer):
    """
    The incumbent, driven through its real production path.

    This deliberately goes through `rtc.AudioFrame` and the plugin's own
    `_process`, int16 quantisation and internal resamplers included, rather than
    calling the ONNX session directly. The thing being benchmarked is the filter
    as deployed, not the model in isolation.
    """

    name = "gtcrn"
    note = "incumbent, 48.2K params, 16 kHz native"
    native_rate = 16000

    def __init__(self) -> None:
        from stt_api.livekit_plugin.noise_cancellation import GTCRN

        self._cls = GTCRN
        self._nc = GTCRN()

    def reset(self) -> None:
        self._nc = self._cls()

    def process(self, frame: np.ndarray, rate: int) -> np.ndarray:
        from livekit import rtc

        n = len(frame)
        pcm = np.clip(np.rint(frame * 32768.0), -32768, 32767).astype(np.int16)
        out = self._nc._process(rtc.AudioFrame(pcm.tobytes(), rate, 1, n))
        y = np.frombuffer(out.data, dtype=np.int16, count=out.samples_per_channel)
        return y.astype(np.float32) / 32768.0


class DTLNEnhancer(Enhancer):
    """
    DTLN — two stacked LSTM stages, the closest real competitor on a CPU budget.

    Stage 1 predicts a magnitude mask on a 257-bin spectrum; stage 2 is a learned
    1-D convolutional filter applied to the time-domain signal that stage 1
    reconstructed. Both carry explicit LSTM state in and out, which is what makes
    it genuinely streamable — unlike DeepFilterNet's published export.

    Frame geometry and the unwindowed overlap-add below are not a simplification:
    they mirror the reference `real_time_processing.py` exactly. DTLN applies no
    analysis window and relies on stage 2 having learned the overlap-add scaling,
    so imposing a Hann window here would quietly change what the model was trained
    to receive.
    """

    name = "dtln"
    note = "2x LSTM, 512/128 STFT, 16 kHz native"
    native_rate = 16000
    _BLOCK, _HOP = 512, 128

    def __init__(self) -> None:
        import onnxruntime as ort

        from .corpus import fetch_model

        p1 = fetch_model("niobures/DTLN", "models/DTLN/onnx/model_1.onnx")
        p2 = fetch_model("niobures/DTLN", "models/DTLN/onnx/model_2.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._s1 = ort.InferenceSession(str(p1), opts, providers=["CPUExecutionProvider"])
        self._s2 = ort.InferenceSession(str(p2), opts, providers=["CPUExecutionProvider"])
        # Bind by shape, not by position or name: these exports carry autogenerated
        # names (`input_2`, `input_3`) whose order is not guaranteed across
        # re-exports, and silently swapping the spectrum for the LSTM state would
        # still run and still produce audio, just wrong audio.
        self._x1, self._h1 = self._io(self._s1, self._BLOCK // 2 + 1)
        self._x2, self._h2 = self._io(self._s2, self._BLOCK)
        self._state_shape = [int(d) for d in self._s1.get_inputs()[self._h1].shape]
        self.reset()

    @staticmethod
    def _io(sess, feat_dim: int) -> tuple[int, int]:
        feat = state = None
        for i, inp in enumerate(sess.get_inputs()):
            if len(inp.shape) == 3 and int(inp.shape[-1]) == feat_dim:
                feat = i
            else:
                state = i
        if feat is None or state is None:
            raise RuntimeError(f"unexpected DTLN input signature: {[i.shape for i in sess.get_inputs()]}")
        return feat, state

    def reset(self) -> None:
        z = lambda: np.zeros(self._state_shape, dtype=np.float32)  # noqa: E731
        self._st1, self._st2 = z(), z()
        self._inbuf = np.zeros(self._BLOCK, dtype=np.float32)
        self._outbuf = np.zeros(self._BLOCK, dtype=np.float32)
        self._stream = _HopStreamer(self.native_rate, self._HOP, self._step)

    def _step(self, hop: np.ndarray) -> np.ndarray:
        self._inbuf = np.concatenate((self._inbuf[self._HOP :], hop))
        spec = np.fft.rfft(self._inbuf)
        mag = np.abs(spec).reshape(1, 1, -1).astype(np.float32)

        n1 = self._s1.get_inputs()
        mask, self._st1 = self._s1.run(
            None, {n1[self._x1].name: mag, n1[self._h1].name: self._st1}
        )
        est = np.fft.irfft(mag * mask * np.exp(1j * np.angle(spec))).reshape(1, 1, -1)

        n2 = self._s2.get_inputs()
        block, self._st2 = self._s2.run(
            None,
            {n2[self._x2].name: est.astype(np.float32), n2[self._h2].name: self._st2},
        )

        self._outbuf = np.concatenate(
            (self._outbuf[self._HOP :], np.zeros(self._HOP, dtype=np.float32))
        )
        self._outbuf = self._outbuf + np.squeeze(block)
        return self._outbuf[: self._HOP].copy()

    def process(self, frame: np.ndarray, rate: int) -> np.ndarray:
        return self._stream.push(frame, rate)


class RNNoiseEnhancer(Enhancer):
    """
    Xiph's RNNoise, through the actual C library.

    Deliberately not the ONNX export that circulates on the Hub. That graph is
    only the GRU: it takes 42 hand-built features (Bark-band BFCCs, their deltas,
    pitch correlation and period) and returns 22 band gains, leaving the entire
    DSP front-end and the pitch comb filter to the caller. Reimplementing that in
    numpy is several hundred lines of exactly the kind of code whose bugs look
    like "this model just scores badly" — so the benchmark would be measuring my
    feature extraction, not RNNoise.

    Fixed at 48 kHz with 10 ms frames; the C API offers no other geometry. On a
    16 kHz pipeline that means an up/down resampler pair, which is a real cost and
    is charged to it here.
    """

    name = "rnnoise"
    note = "Xiph C library, 48 kHz native, 10 ms frames"
    native_rate = 48000

    def __init__(self) -> None:
        from pyrnnoise.rnnoise import FRAME_SIZE, create, destroy, process_mono_frame

        self._create, self._destroy, self._proc = create, destroy, process_mono_frame
        self._hop = FRAME_SIZE
        self._state = None
        self.reset()

    def __del__(self) -> None:
        try:
            if getattr(self, "_state", None) is not None:
                self._destroy(self._state)
        except Exception:
            pass

    def reset(self) -> None:
        if self._state is not None:
            self._destroy(self._state)
        self._state = self._create()
        self._stream = _HopStreamer(self.native_rate, self._hop, self._step)

    def _step(self, hop: np.ndarray) -> np.ndarray:
        # The library works in int16 range. Frames round-trip through int16 here,
        # which matches both how RNNoise is deployed and what LiveKit frames
        # already are, so this costs no fairness against the other candidates.
        pcm = np.clip(np.rint(hop * 32768.0), -32768, 32767).astype(np.int16)
        out, _ = self._proc(self._state, pcm)
        return out.astype(np.float32) / 32768.0

    def process(self, frame: np.ndarray, rate: int) -> np.ndarray:
        return self._stream.push(frame, rate)


class WienerEnhancer(Enhancer):
    """
    Classical decision-directed Wiener filtering. No model, no weights, pure numpy.

    Its job is attribution. Without it, every neural gain gets reported against
    raw noisy input, which overstates what the *learning* bought — a good part of
    the improvement on stationary noise is available from 1980s DSP for nothing.
    Anything a neural model wins over this row is the part that actually needed a
    neural model.

    Noise PSD comes from minimum statistics over a sliding window (Martin), the a
    priori SNR from the Ephraim-Malah decision-directed update. Both are causal.
    """

    name = "wiener"
    note = "decision-directed Wiener + minimum statistics, numpy only"
    native_rate = 16000
    _N_FFT, _HOP = 512, 256
    _ALPHA = 0.98  # decision-directed smoothing
    _MIN_GAIN = 10 ** (-18 / 20)  # floor at -18 dB; deeper sounds "watery"
    _WIN_FRAMES = 60  # ~1 s of minimum-statistics history
    _PSD_SMOOTH = 0.85
    """
    Recursive smoothing applied to the periodogram *before* minimum tracking.

    Not optional, and the reason is worth recording. A raw periodogram bin is
    chi-square with 2 degrees of freedom — its variance equals its mean — so the
    minimum over a 60-frame window sits far below the true noise power. Measured
    on this corpus it underestimates by 22x (13.5 dB). Feed that to the gain rule
    and the posterior SNR reads ~15x too high everywhere, the gain pins at 1.0,
    and the filter silently becomes a passthrough that still looks like it is
    working. Smoothing first cuts the estimator variance and brings the minimum
    to within a factor of two of the true floor.
    """
    _BIAS = 1.5
    """Bias compensation for the smoothed minimum. The minimum of any noisy
    estimate is below its mean, so it must be scaled up; with smoothing already
    applied the residual bias is small, unlike the 20x+ correction the unsmoothed
    periodogram would need."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        n = self._N_FFT
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)
        self._win = np.sqrt(hann).astype(np.float32)
        self._analysis = np.zeros(n, dtype=np.float32)
        self._ola = np.zeros(n, dtype=np.float32)
        self._hist: list[np.ndarray] = []
        self._psd = None
        self._prev_gain = None
        self._prev_power = None
        self._stream = _HopStreamer(self.native_rate, self._HOP, self._step)

    def _step(self, hop: np.ndarray) -> np.ndarray:
        self._analysis = np.concatenate((self._analysis[self._HOP :], hop))
        spec = np.fft.rfft(self._analysis * self._win)
        power = np.abs(spec) ** 2

        # Smooth, then track the minimum of the smoothed estimate. Order matters —
        # see `_PSD_SMOOTH`.
        self._psd = power if self._psd is None else (
            self._PSD_SMOOTH * self._psd + (1.0 - self._PSD_SMOOTH) * power
        )
        self._hist.append(self._psd)
        if len(self._hist) > self._WIN_FRAMES:
            self._hist.pop(0)
        # Minimum statistics: the minimum over a window longer than any plausible
        # syllable tracks the noise floor without ever needing a speech/silence
        # decision, which is what makes it robust on non-stationary input.
        noise = np.maximum(np.minimum.reduce(self._hist) * self._BIAS, 1e-12)

        post = power / noise
        if self._prev_gain is None:
            prio = np.maximum(post - 1.0, 0.0)
        else:
            prio = self._ALPHA * (self._prev_gain**2) * (self._prev_power / noise) + (
                1 - self._ALPHA
            ) * np.maximum(post - 1.0, 0.0)
        prio = np.maximum(prio, 1e-6)

        gain = np.maximum(prio / (1.0 + prio), self._MIN_GAIN)
        self._prev_gain, self._prev_power = gain, power

        wave_out = np.fft.irfft(spec * gain, n=self._N_FFT).astype(np.float32)
        self._ola = self._ola + wave_out * self._win
        out = self._ola[: self._HOP].copy()
        self._ola = np.concatenate(
            (self._ola[self._HOP :], np.zeros(self._HOP, dtype=np.float32))
        )
        return out

    def process(self, frame: np.ndarray, rate: int) -> np.ndarray:
        return self._stream.push(frame, rate)


class DeepFilterNet3Offline(Enhancer):
    """
    DeepFilterNet3 via the official `deep-filter` binary. A ceiling, not a candidate.

    Two decisions worth stating plainly.

    **Why offline.** DF3's published ONNX cannot stream: the encoder's GRU has no
    state input or output, so calling it per frame resets the recurrence every
    10 ms. Upstream only streams it through tract's pulse transform, which
    onnxruntime has no equivalent of. The plugin README already works through this
    in detail. Running it whole-file here grants it unlimited lookahead — which is
    exactly the point. It answers "how much quality is left on the table?", and
    the honest answer needs DF3 at its best, not a hobbled version of it.

    **Why the binary and not the ONNX graph.** A verified export exists
    (`soniqo/DeepFilterNet3-ONNX`) but it is the neural graph only — no STFT, no
    32-band ERB bank, no `norm_tau` running normalisation, no 5-tap deep filter,
    no synthesis. Reimplementing that stack to benchmark against is how you end up
    measuring your own DSP bugs and publishing them as DF3's score. The prebuilt
    binary is the reference implementation, and the plugin README already names it
    as the thing to validate against.

    Its numbers are therefore not comparable on the cost axis and are reported as
    such. Nothing here is deployable in the agent as it stands.
    """

    name = "dfn3"
    note = "OFFLINE ceiling — full lookahead, not deployable"
    native_rate = 48000
    streaming = False

    _VERSION = "0.5.6"
    _ASSETS = {
        ("Darwin", "arm64"): "deep-filter-{v}-aarch64-apple-darwin",
        ("Darwin", "x86_64"): "deep-filter-{v}-x86_64-apple-darwin",
        ("Linux", "x86_64"): "deep-filter-{v}-x86_64-unknown-linux-musl",
        ("Linux", "aarch64"): "deep-filter-{v}-aarch64-unknown-linux-gnu",
    }

    def __init__(self) -> None:
        self._bin = self._ensure_binary()

    @classmethod
    def _ensure_binary(cls) -> Path:
        import platform

        key = (platform.system(), platform.machine())
        asset = cls._ASSETS.get(key)
        if asset is None:
            raise RuntimeError(
                f"no prebuilt deep-filter for {key}; build it from "
                "https://github.com/Rikorose/DeepFilterNet or drop a binary at "
                "DEEP_FILTER_BIN"
            )
        override = os.environ.get("DEEP_FILTER_BIN")
        if override:
            if not os.path.exists(override):
                raise FileNotFoundError(f"DEEP_FILTER_BIN set but missing: {override}")
            return Path(override)

        name = asset.format(v=cls._VERSION)
        dest = _CACHE / "bin" / name
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://github.com/Rikorose/DeepFilterNet/releases/download/"
            f"v{cls._VERSION}/{name}"
        )
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        tmp.replace(dest)
        return dest

    def reset(self) -> None:
        pass

    def process_all(self, x: np.ndarray, rate: int) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            src, outdir = Path(td) / "in.wav", Path(td) / "out"
            outdir.mkdir()
            _write_wav(src, x, rate)
            proc = subprocess.run(
                [str(self._bin), "-o", str(outdir), str(src)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            produced = sorted(outdir.glob("*.wav"))
            if not produced:
                raise RuntimeError(
                    f"deep-filter produced nothing (rc={proc.returncode}): "
                    f"{proc.stderr.strip()[:300]}"
                )
            y, out_rate = _read_wav(produced[0])
        if out_rate != rate:
            # The binary resamples to 48 kHz internally and writes at that rate.
            y = StreamResampler(out_rate, rate).push(y)
        return y


def _write_wav(path: Path, x: np.ndarray, rate: int) -> None:
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
        ch = w.getnchannels()
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        y = y.reshape(-1, ch).mean(axis=1)
    return y, rate


REGISTRY: dict[str, type[Enhancer]] = {
    Passthrough.name: Passthrough,
    GTCRNEnhancer.name: GTCRNEnhancer,
    DTLNEnhancer.name: DTLNEnhancer,
    RNNoiseEnhancer.name: RNNoiseEnhancer,
    WienerEnhancer.name: WienerEnhancer,
    DeepFilterNet3Offline.name: DeepFilterNet3Offline,
}


def _register_generative() -> None:
    """
    Fold in the generative restorers, if their module imports.

    Kept lazy and failure-tolerant: `generative.py` reaches for interpreters under
    NC_GEN_ROOT that only exist on a machine set up for them, and an acoustic-only
    run on a laptop must not be blocked by their absence.
    """
    try:
        from .generative import GENERATIVE
    except Exception:  # noqa: BLE001
        return
    REGISTRY.update(GENERATIVE)


_register_generative()

DEFAULT = ["passthrough", "wiener", "rnnoise", "dtln", "gtcrn"]
"""Cheap-to-run set. `dfn3` is excluded: it downloads a 28 MB binary and, being a
ceiling rather than a candidate, is not something you need on every run."""


def build(name: str) -> Enhancer:
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"unknown enhancer {name!r}; available: {', '.join(sorted(REGISTRY))}"
        ) from None
    return cls()


def available() -> dict[str, str | None]:
    """
    Map each registered name to why it cannot run, or None if it can.

    Checked by construction rather than by import probing, because most failures
    here are missing *model files* or a missing binary, not missing packages.
    """
    out: dict[str, str | None] = {}
    for name, cls in REGISTRY.items():
        try:
            cls()
            out[name] = None
        except Exception as e:  # noqa: BLE001 - reporting, not handling
            out[name] = f"{type(e).__name__}: {e}"
    return out
