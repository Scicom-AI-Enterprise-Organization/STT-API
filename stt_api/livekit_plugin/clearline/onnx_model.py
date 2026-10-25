"""
Clearline-VAD's ONNX graph behind the interface silero's `VADStream` expects
from its `OnnxModel`: `window_size_samples`, `sample_rate`, `reset()` and
`__call__(float32 window) -> p(speech)`.

The graph takes one window of 512 samples (two 16 ms hops) or 256 (one hop) at
16 kHz, with its state passed in and out: the last 256 input samples (the
analysis window reaches back past the chunk), the running log-mel mean and one
GRU state per layer. It returns one probability per hop; the window's
probability is its last hop's, the most recent evidence.

The state is per stream and the session is shared, which is why `ClearlineVAD`
opens one session and hands every stream its own `OnnxModel`.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000


def new_inference_session(path: str, num_threads: int = 1):
    import onnxruntime  # type: ignore

    opts = onnxruntime.SessionOptions()
    # A spinning worker burns a core between 30-microsecond calls.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = num_threads
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return onnxruntime.InferenceSession(
        path, providers=["CPUExecutionProvider"], sess_options=opts
    )


class OnnxModel:
    def __init__(self, *, onnx_session) -> None:
        self._sess = onnx_session
        ins = onnx_session.get_inputs()
        self._window_size_samples = int(ins[0].shape[1])
        # The state is every input but x, in order: ctx, mean, then one GRU state
        # per layer. Outputs come back in the same order after p.
        self._names = [i.name for i in ins[1:]]
        self._shapes = [[int(d) for d in i.shape] for i in ins[1:]]
        meta = onnx_session.get_modelmeta().custom_metadata_map
        # The running mean starts at the log-mel of a quiet line, not at zero, so a
        # stream's first frames read as quiet rather than as a jump in level.
        self._mean0 = float(meta.get("mean0", -12.0))
        self.reset()

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def window_size_samples(self) -> int:
        return self._window_size_samples

    def reset(self) -> None:
        self._state = {
            n: (
                np.full(sh, self._mean0, np.float32)
                if n == "mean"
                else np.zeros(sh, np.float32)
            )
            for n, sh in zip(self._names, self._shapes)
        }
        self._x = np.zeros((1, self._window_size_samples), np.float32)

    def __call__(self, x: np.ndarray) -> float:
        self._x[0] = x
        out = self._sess.run(None, {"x": self._x, **self._state})
        self._state = dict(zip(self._names, out[1:]))
        return float(out[0][0, -1])
