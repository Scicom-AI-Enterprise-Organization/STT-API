"""
The silero VAD server behind `RemoteSileroVAD`: every call's windows, batched.

    python -m stt_api.livekit_plugin.remote_silero.server --port 8766 --device cuda

Protocol, one WebSocket per VAD stream:

* on connect the server sends one text message, its hello:
  `{"model": ..., "weights_sha256": ..., "sample_rate": 16000, "window": 512}`;
* the client sends binary messages of 16 kHz mono int16 PCM, a whole number of
  512-sample windows each;
* the server answers every window with one little-endian float32, p(speech), in order;
* a text message `{"reset": true}` zeroes the call's model state between two windows.

Batching: a call has at most one window in a batch (its next window needs the state
the previous one leaves). A batch is every call with a window ready, up to
`--max-batch`, scored in one run on a worker thread while the event loop keeps
reading sockets; the windows that arrive meanwhile make the next batch, so batches
grow with the load by themselves. `--max-wait-ms` makes the first ready window wait
for others first: fewer, larger batches (for a GPU shared with other models), at
that much latency. The model state lives
in one tensor on the device with a row per call, so a batch is a gather, a forward
and a scatter.

Backends:

* `torch` (default): silero's TorchScript, from the installed `silero-vad` package or
  `--model`. Pin the package to the weights the agents' `livekit-plugins-silero` ships:
  `silero-vad==6.1` for livekit-agents 1.2 / 1.3, `silero-vad==6.2.1` for 1.8. The
  hello carries the sha256 of the ONNX graph from the same package, and the client
  warns when it is not the one it has locally.
* `onnx`: an ONNX graph through onnxruntime (CUDA when onnxruntime-gpu is installed,
  else CPU). By default LiveKit's own `silero_vad.onnx`, so it matches the agents
  exactly. The tests use it.

`GET /health` answers `{"ok": true}`; `GET /stats` the counters since start.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import hashlib
import json
import logging
import os
import resource
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("remote_silero.server")

SAMPLE_RATE = 16000
WINDOW = 512
CONTEXT = 64  # samples of the previous window silero v5+ prepends
_RESET = object()


def _key(name: str):
    from aiohttp import web

    return web.AppKey(name, object) if hasattr(web, "AppKey") else name


BATCHER = _key("batcher")
_TASK = _key("batcher_task")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TorchBackend:
    """silero's TorchScript core (`model._model(x, state)`), its state passed explicitly."""

    def __init__(self, jit: str | None, device: str) -> None:
        import torch

        if jit is None:
            import importlib.metadata
            import importlib.util

            # the package's files, not the package: importing it pulls torchaudio in
            spec = importlib.util.find_spec("silero_vad")
            if spec is None or not spec.submodule_search_locations:
                raise ModuleNotFoundError(
                    "the torch backend reads silero-vad's TorchScript: pip install "
                    "silero-vad (--no-deps is enough), or pass --model"
                )
            data = Path(list(spec.submodule_search_locations)[0]) / "data"
            jit = str(data / "silero_vad.jit")
            self.name = f"silero-vad {importlib.metadata.version('silero-vad')}"
            onnx = data / "silero_vad.onnx"
            self.weights_sha256 = _sha256(onnx) if onnx.exists() else None
        else:
            self.name, self.weights_sha256 = f"silero jit {Path(jit).name}", None
        # TF32 moves p by ~1e-3 for no speed at these sizes: keep float32 exact
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # TorchScript's optimizer recompiles when it meets a new batch size: one
        # stall of 182 ms on an H20, mid-call. Off, a batch costs 0.07 ms more and
        # never stalls.
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(False)
        self.torch, self.device = torch, torch.device(device)
        self.core = torch.jit.load(jit, map_location=self.device)._model.eval()
        self.cap = 0
        self._grow(256)
        with torch.inference_mode():  # warm the kernels up
            for b in (1, 8, 64):
                self.infer(np.zeros((b, WINDOW), np.int16), np.arange(b), [])

    def _grow(self, need: int) -> None:
        cap = max(need, 2 * self.cap, 256)
        st = self.torch.zeros(2, cap, 128, device=self.device)
        cx = self.torch.zeros(cap, CONTEXT, device=self.device)
        if self.cap:
            st[:, : self.cap] = self.state
            cx[: self.cap] = self.ctx
        self.state, self.ctx, self.cap = st, cx, cap

    def infer(
        self, x16: np.ndarray, slots: np.ndarray, resets: list[int]
    ) -> np.ndarray:
        torch = self.torch
        top = max(
            int(slots.max()) + 1 if len(slots) else 0, max(resets, default=-1) + 1
        )
        if top > self.cap:
            self._grow(top)
        with torch.inference_mode():
            if resets:
                r = torch.as_tensor(resets, device=self.device)
                self.state[:, r] = 0
                self.ctx[r] = 0
            if not len(slots):
                return np.empty(0, np.float32)
            idx = torch.from_numpy(slots).to(self.device)
            x = torch.from_numpy(x16).to(self.device).float().div_(32767.0)
            x = torch.cat([self.ctx.index_select(0, idx), x], 1)
            out, st = self.core(x, self.state.index_select(1, idx))
            self.state.index_copy_(1, idx, st)
            self.ctx.index_copy_(0, idx, x[:, -CONTEXT:])
            return out[:, 0].float().cpu().numpy()


class OnnxBackend:
    """A silero ONNX graph with a batch dimension; the state rows live in numpy."""

    def __init__(self, path: str | None, device: str) -> None:
        import onnxruntime as ort

        if path is None:
            import importlib.resources

            path = str(
                importlib.resources.files("livekit.plugins.silero.resources")
                / "silero_vad.onnx"
            )
        self.name, self.weights_sha256 = f"silero onnx {Path(path).name}", _sha256(path)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        if (
            device.startswith("cuda")
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers.insert(0, "CUDAExecutionProvider")
        self.sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
        self.sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.cap = 0
        self.state = np.zeros((2, 0, 128), np.float32)
        self.ctx = np.zeros((0, CONTEXT), np.float32)

    def infer(
        self, x16: np.ndarray, slots: np.ndarray, resets: list[int]
    ) -> np.ndarray:
        top = max(
            int(slots.max()) + 1 if len(slots) else 0, max(resets, default=-1) + 1
        )
        if top > self.cap:
            cap = max(top, 2 * self.cap, 256)
            self.state = np.concatenate(
                [self.state, np.zeros((2, cap - self.cap, 128), np.float32)], 1
            )
            self.ctx = np.concatenate(
                [self.ctx, np.zeros((cap - self.cap, CONTEXT), np.float32)], 0
            )
            self.cap = cap
        for r in resets:
            self.state[:, r] = 0
            self.ctx[r] = 0
        if not len(slots):
            return np.empty(0, np.float32)
        x = np.concatenate([self.ctx[slots], x16.astype(np.float32) / 32767.0], 1)
        out, st = self.sess.run(
            None,
            {
                "input": x,
                "state": np.ascontiguousarray(self.state[:, slots]),
                "sr": self.sr,
            },
        )
        self.state[:, slots] = st
        self.ctx[slots] = x[:, -CONTEXT:]
        return out[:, 0].astype(np.float32)


class _Call:
    __slots__ = ("ws", "slot", "pending", "queued", "busy", "closed", "reset")

    def __init__(self, ws, slot: int) -> None:
        self.ws, self.slot = ws, slot
        self.pending: collections.deque = (
            collections.deque()
        )  # windows and _RESET, in order
        self.queued = False  # in the batcher's ready queue
        self.busy = False  # a window of this call is being scored
        self.closed = False
        self.reset = True  # a new slot starts from zero state


class Batcher:
    def __init__(self, backend, max_batch: int, max_wait: float) -> None:
        self.backend, self.max_batch, self.max_wait = backend, max_batch, max_wait
        self.ready: collections.deque[_Call] = collections.deque()
        self.wake = asyncio.Event()
        self.full = asyncio.Event()
        self.pool = concurrent.futures.ThreadPoolExecutor(
            1, thread_name_prefix="silero"
        )
        self.free: list[int] = []
        self.next_slot = 0
        self.live: set[_Call] = set()
        self.calls = 0
        self.stats = collections.Counter()
        self.batch_ms: collections.deque[float] = collections.deque(maxlen=2000)
        self.wait_ms: collections.deque[float] = collections.deque(maxlen=2000)

    def open(self, ws) -> _Call:
        slot = self.free.pop() if self.free else self.next_slot
        if slot == self.next_slot:
            self.next_slot += 1
        self.calls += 1
        self.stats["calls_total"] += 1
        c = _Call(ws, slot)
        self.live.add(c)
        return c

    def close(self, c: _Call) -> None:
        if c.closed:
            return
        self.live.discard(c)
        c.closed = True
        c.pending.clear()
        self.calls -= 1
        if not c.busy:
            self.free.append(c.slot)

    def submit(self, c: _Call) -> None:
        if not c.queued and not c.busy and not c.closed and c.pending:
            c.queued = True
            self.ready.append(c)
            self.wake.set()
            if len(self.ready) >= self.max_batch:
                self.full.set()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self.wake.wait()
            if self.max_wait > 0 and len(self.ready) < self.max_batch:
                try:
                    await asyncio.wait_for(self.full.wait(), self.max_wait)
                except asyncio.TimeoutError:
                    pass
            self.wake.clear()
            self.full.clear()
            batch: list[_Call] = []
            wins: list[np.ndarray] = []
            resets: list[int] = []
            t_first = None
            while self.ready and len(batch) < self.max_batch:
                c = self.ready.popleft()
                c.queued = False
                if c.closed:
                    continue
                while c.pending and c.pending[0] is _RESET:
                    c.pending.popleft()
                    c.reset = True
                if c.reset:
                    resets.append(c.slot)
                    c.reset = False
                if not c.pending:
                    continue
                win, t = c.pending.popleft()
                t_first = t if t_first is None else min(t_first, t)
                c.busy = True
                batch.append(c)
                wins.append(win)
            if not batch and not resets:
                continue
            x = np.stack(wins) if wins else np.empty((0, WINDOW), np.int16)
            slots = np.array([c.slot for c in batch], dtype=np.int64)
            t0 = time.perf_counter()
            try:
                probs = await loop.run_in_executor(
                    self.pool, self.backend.infer, x, slots, resets
                )
            except Exception:  # noqa: BLE001 - the calls in it are dropped, not the server
                logger.exception("a batch of %d failed", len(batch))
                for c in batch:
                    c.busy = False
                    if not c.closed:
                        await c.ws.close()
                continue
            t1 = time.perf_counter()
            if batch:
                self.batch_ms.append((t1 - t0) * 1e3)
                self.wait_ms.append((t0 - t_first) * 1e3)
                self.stats["batches"] += 1
                self.stats["windows"] += len(batch)
            for c, p in zip(batch, probs):
                c.busy = False
                if c.closed:
                    self.free.append(c.slot)
                    continue
                try:
                    await c.ws.send_bytes(np.float32(p).tobytes())
                except Exception:  # noqa: BLE001 - its reader sees the close
                    continue
                self.submit(c)
            if self.ready:
                self.wake.set()


def make_app(backend, max_batch: int = 1024, max_wait_ms: float = 0.0):
    from aiohttp import WSMsgType, web

    batcher = Batcher(backend, max_batch, max_wait_ms / 1000.0)
    hello = json.dumps(
        {
            "model": backend.name,
            "weights_sha256": backend.weights_sha256,
            "sample_rate": SAMPLE_RATE,
            "window": WINDOW,
        }
    )

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=1 << 20)
        await ws.prepare(request)
        call = batcher.open(ws)
        try:
            await ws.send_str(hello)
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    n = len(msg.data) // (2 * WINDOW)
                    if n * 2 * WINDOW != len(msg.data):
                        await ws.close(
                            message=b"binary messages are whole 512-sample int16 windows"
                        )
                        break
                    pcm = np.frombuffer(msg.data, dtype="<i2").reshape(n, WINDOW)
                    t = time.perf_counter()
                    call.pending.extend((w, t) for w in pcm)
                    batcher.submit(call)
                elif msg.type == WSMsgType.TEXT:
                    if json.loads(msg.data).get("reset"):
                        call.pending.append(_RESET)
                        batcher.submit(call)
        finally:
            batcher.close(call)
        return ws

    async def health(_request):
        return web.json_response({"ok": True, "model": backend.name})

    async def stats(_request):
        ru = resource.getrusage(resource.RUSAGE_SELF)
        b = sorted(batcher.batch_ms)
        w = sorted(batcher.wait_ms)
        pct = lambda a, q: round(a[min(len(a) - 1, int(q * len(a)))], 3) if a else None  # noqa: E731
        s = batcher.stats
        return web.json_response(
            {
                "calls": batcher.calls,
                "calls_total": s["calls_total"],
                "windows": s["windows"],
                "batches": s["batches"],
                "mean_batch": round(s["windows"] / s["batches"], 2)
                if s["batches"]
                else None,
                "batch_ms_p50": pct(b, 0.5),
                "batch_ms_p99": pct(b, 0.99),
                "queue_ms_p50": pct(w, 0.5),
                "queue_ms_p99": pct(w, 0.99),
                "cpu_s": round(ru.ru_utime + ru.ru_stime, 3),
            }
        )

    async def start(app):
        app[_TASK] = asyncio.create_task(batcher.run())

    async def shutdown(app):
        # close every call's socket, so its agent fails over now rather than at its
        # reply timeout
        from aiohttp import WSCloseCode

        for c in list(batcher.live):
            await c.ws.close(code=WSCloseCode.GOING_AWAY, message=b"server shutdown")

    async def stop(app):
        app[_TASK].cancel()
        batcher.pool.shutdown(wait=False)

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/stats", stats)
    app.router.add_get("/{tail:.*}", ws_handler)
    app.on_startup.append(start)
    app.on_shutdown.append(shutdown)
    app.on_cleanup.append(stop)
    app[BATCHER] = batcher
    return app


def build_backend(kind: str, device: str, path: str | None):
    if kind == "torch":
        return TorchBackend(path, device)
    if kind == "onnx":
        return OnnxBackend(path, device)
    raise ValueError(f"unknown backend {kind!r}: torch or onnx")


def _serve(args, reuse_port: bool) -> None:
    from aiohttp import web

    backend = build_backend(args.backend, args.device, args.model)
    app = make_app(backend, args.max_batch, args.max_wait_ms)
    logger.info(
        "serving %s on %s:%d (pid %d)", backend.name, args.host, args.port, os.getpid()
    )
    web.run_app(
        app, host=args.host, port=args.port, reuse_port=reuse_port or None, print=None
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:1, cpu")
    ap.add_argument(
        "--model",
        default=None,
        help="a TorchScript file (torch) or an ONNX graph (onnx) instead of the default",
    )
    ap.add_argument("--max-batch", type=int, default=1024)
    ap.add_argument(
        "--max-wait-ms",
        type=float,
        default=0.0,
        help="how long the first window of a batch waits for others (0: none)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes on one port (SO_REUSEPORT, Linux); each batches its own calls",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.workers <= 1:
        _serve(args, reuse_port=False)
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_serve, args=(args, True)) for _ in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
