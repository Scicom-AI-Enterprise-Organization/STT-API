"""
The CPU voice isolator's live stream on onnxruntime and numpy: no torch.

One `OnnxStream` per call. Feed it 16 kHz mono float32 audio in any amounts; it
returns the isolated caller as fast as whole 10 ms hops arrive, `delay` (160)
samples behind the input: output sample n is the caller at input sample n - 160,
the STFT window's look-ahead.

A model directory holds `config.json`, `step.onnx` and `enroll.onnx`, written by
Voice-Isolator's `scripts/export_onnx.py`. `step.onnx` is one streaming step of
the transformer: the STFT of C new frames, the speaker vector, every layer's ring
of keys and values and the LSTM states in; the masked spectrum, the C new keys and
values and the next LSTM states out. Everything a call keeps is an input and an
output, so one session serves every call in the process.

Here: the STFT and overlap-add, the rings (each layer keeps the rotated keys and
the values of the last `window - 1` frames; a step attends over those plus its own
new frames, so writing them back is a few rows, never a copy of the ring), the
rotary angles and the attention mask.

The live enrollment, "selfl": from 3 s in, the speaker vector comes from the
loudest 3 s of the input heard so far, re-made at every half second while the call
is at most 15.5 s old, then fixed. Before 3 s the model runs with its learned "no
enrollment" vector. Runs of the step never cross a half second, so the vector
changes exactly where the offline rule changes it: the export is checked against
the PyTorch model running that rule (124-134 dB on real calls, float rounding).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


@lru_cache(maxsize=8)
def load_model(model_dir: str, num_threads: int = 1):
    """
    (config, step session, enrollment session), shared by every stream.

    onnxruntime sessions are safe to call from several threads and hold no state of
    a call, so a process needs one per (model, thread count) however many calls it
    carries.
    """
    import onnxruntime as ort

    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text())
    opts = ort.SessionOptions()
    # One thread by default: it runs inside the agent beside the VAD, the STT client
    # and everything else on the core, one step per call per frame.
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    cpu = ["CPUExecutionProvider"]
    step = ort.InferenceSession(str(d / cfg["step_file"]), opts, providers=cpu)
    enroll = ort.InferenceSession(str(d / cfg["enroll_file"]), opts, providers=cpu)
    return cfg, step, enroll


def sqrt_hann(n: int) -> np.ndarray:
    """Periodic Hann (`torch.hann_window`), square-rooted: analysis and synthesis."""
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)
    return np.sqrt(w).astype(np.float32)


class OnnxStream:
    """One call's state and its step. `push` audio in, isolated audio out."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        num_threads: int = 1,
        live_enroll: bool = True,
        max_frames: int = 8,
        loud_s: float = 3.0,
        loud_within: float = 15.0,
        step_s: float = 0.5,
    ) -> None:
        self.cfg, self.step, self.enc = load_model(str(model_dir), num_threads)
        c = self.cfg
        self.win, self.hop = c["win"], c["hop"]
        self.delay = self.win - self.hop
        self.delay_s = self.delay / SAMPLE_RATE
        self.window, self.R = c["window"], c["window"] - 1
        self.L, self.heads, self.hd = c["layers"], c["heads"], c["head_dim"]
        self.w = sqrt_hann(self.win)
        ola = np.zeros(self.win)
        for k in range(0, self.win, self.hop):
            ola += np.roll(self.w.astype(np.float64) ** 2, k)
        self.norm = float(ola.mean())
        self.inv = c["rope_base"] ** (
            -np.arange(0, self.hd, 2, dtype=np.float64) / self.hd
        )
        self.null = np.asarray(c["null"], np.float32)[None]
        self.max_frames = max_frames
        self.live = live_enroll
        self.loud_s, self.loud_until = loud_s, loud_within + step_s
        self.step_n = int(step_s * SAMPLE_RATE)
        self.reset()

    def reset(self) -> None:
        """A new call: every state back to its start."""
        self.inbuf = np.zeros(self.delay, np.float32)
        self.outbuf = np.zeros(self.delay, np.float32)
        ring = (1, self.heads, self.R, self.hd)
        self.k = [np.zeros(ring, np.float32) for _ in range(self.L)]
        self.v = [np.zeros(ring, np.float32) for _ in range(self.L)]
        self.pos = np.full(self.R, -1, np.int64)  # the frame each slot holds, -1 empty
        self.t = 0  # the next frame
        self.states = {
            s["name"]: np.zeros(s["shape"], np.float32) for s in self.cfg["states"]
        }
        self.emb = self.null.copy()
        self.enrolled_at: float | None = None  # seconds in, first enrollment
        self.pending = np.zeros(0, np.float32)
        self.n_in = 0  # input samples stepped
        n_hist = int(self.loud_until * SAMPLE_RATE) + self.step_n
        self.hist = np.zeros(n_hist, np.float32) if self.live else None
        self.loud_st = None

    def close(self) -> None:
        """The call is over: drop its rings (the shared sessions stay)."""
        self.k, self.v, self.hist = [], [], None

    def set_enrollment(self, clip: np.ndarray) -> None:
        """Lock the stream to the voice in `clip` (16 kHz float32, the caller
        alone, a few seconds); the live rule then stays off for this call."""
        self.emb = self.embed(clip)
        self.live = False

    def embed(self, clip: np.ndarray) -> np.ndarray:
        """The speaker vector of a clip, computed as the model's `embed` does."""
        x = np.asarray(clip, np.float32)
        S = len(x)
        x = np.pad(x, (self.delay, self.delay + (-S) % self.hop))
        T = (len(x) - self.win) // self.hop + 1
        idx = np.arange(T)[:, None] * self.hop + np.arange(self.win)[None]
        X = np.fft.rfft(x[idx] * self.w, n=self.win)
        logp = np.log((X.real**2 + X.imag**2) + 1e-8).astype(np.float32)[None]
        valid = np.arange(T) < min(T, max(1, S // self.hop))
        mask = valid.astype(np.float32)[None]
        emb = self.enc.run(None, {"logpow": logp, "mask": mask})[0]
        return emb.astype(np.float32)

    def push(self, x: np.ndarray) -> np.ndarray:
        """Audio in (any length), the isolated audio of every whole hop out."""
        self.pending = np.concatenate([self.pending, np.asarray(x, np.float32)])
        out = []
        while len(self.pending) >= self.hop:
            n = min(len(self.pending) // self.hop, self.max_frames)
            if self.live and self.n_in < self.loud_until * SAMPLE_RATE:
                # a run never crosses a half second: the rule's boundaries
                n = min(n, (self.step_n - self.n_in % self.step_n) // self.hop)
            chunk = self.pending[: n * self.hop]
            self.pending = self.pending[n * self.hop :]
            out.append(self._step(chunk))
            if self.live and self.n_in < len(self.hist):
                m = min(len(chunk), len(self.hist) - self.n_in)
                self.hist[self.n_in : self.n_in + m] = chunk[:m]
            self.n_in += len(chunk)
            if self.live and self.n_in % self.step_n == 0:
                self._loud_rule()
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def _loud_rule(self) -> None:
        t = self.n_in / SAMPLE_RATE
        if not (self.loud_s <= t <= self.loud_until):
            return
        m = self.n_in // self.hop
        w = int(min(self.loud_s, t) * SAMPLE_RATE / self.hop)
        if m < w:
            return
        # float32 energies and a running sum by convolution, exactly as the
        # evaluated rule computes them: ties included, it picks the same window
        e = (self.hist[: m * self.hop].reshape(m, self.hop) ** 2).mean(1)
        st = int(np.convolve(e, np.ones(w), "valid").argmax())
        if (st, w) != self.loud_st:
            self.emb = self.embed(self.hist[st * self.hop : (st + w) * self.hop])
            self.loud_st = (st, w)
            if self.enrolled_at is None:
                self.enrolled_at = t

    def _step(self, chunk: np.ndarray) -> np.ndarray:
        hop, win, R = self.hop, self.win, self.R
        C = len(chunk) // hop
        x = np.concatenate([self.inbuf, chunk])
        self.inbuf = x[-self.delay :].copy()
        frames = x[np.arange(C)[:, None] * hop + np.arange(win)[None]] * self.w
        X = np.fft.rfft(frames, n=win)
        p = self.t + np.arange(C)
        ang = np.remainder(p[:, None].astype(np.float64) * self.inv[None], 2.0 * np.pi)
        old = (self.pos[None] >= 0) & (self.pos[None] > p[:, None] - self.window)
        new = np.arange(C)[None] <= np.arange(C)[:, None]
        allowed = np.concatenate([old, new], 1)
        bias = np.where(allowed, 0.0, -1e9).astype(np.float32)[None, None]
        feed = {
            "re": X.real.astype(np.float32)[None],
            "im": X.imag.astype(np.float32)[None],
            "emb": self.emb,
            "cos": np.cos(ang).astype(np.float32),
            "sin": np.sin(ang).astype(np.float32),
            "bias": bias,
            **self.states,
        }
        for i in range(self.L):
            feed[f"k{i}"], feed[f"v{i}"] = self.k[i], self.v[i]
        yre, yim, knew, vnew, *st = self.step.run(None, feed)
        self.states = {s["name"]: v for s, v in zip(self.cfg["states"], st)}
        slots = p % R
        for i in range(self.L):
            self.k[i][:, :, slots] = knew[i]
            self.v[i][:, :, slots] = vnew[i]
        self.pos[slots] = p
        self.t += C
        y = np.fft.irfft(yre[0] + 1j * yim[0], n=win).astype(np.float32)
        y *= self.w / self.norm
        acc = np.zeros((C - 1) * hop + win, np.float32)
        acc[: self.delay] += self.outbuf
        for c in range(C):
            acc[c * hop : c * hop + win] += y[c]
        self.outbuf = acc[C * hop :].copy()
        return acc[: C * hop]


def isolate(model_dir: str | Path, x: np.ndarray, chunk: int = 800, **kw) -> np.ndarray:
    """A whole recording through the live stream in `chunk`-sample pieces, aligned
    with the input and the same length."""
    s = OnnxStream(model_dir, **kw)
    x = np.asarray(x, np.float32)
    parts = [s.push(x[i : i + chunk]) for i in range(0, len(x), chunk)]
    parts.append(s.push(np.zeros(s.delay + s.hop, np.float32)))
    return np.concatenate(parts)[s.delay : s.delay + len(x)]
