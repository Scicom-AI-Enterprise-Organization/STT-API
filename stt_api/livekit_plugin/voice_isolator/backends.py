"""
The models behind the voice isolator: one local, one remote, one interface.

A voice isolator keeps the caller and removes every other voice and the noise,
before the VAD, the turn detector and the STT hear the call. A noise suppressor
such as `../noise_cancellation/` cannot do the second half: another person's
voice *is* speech, and cleaning the noise off it makes the VAD fire on it more,
not less.

Two backends, because the useful deployments are genuinely different:

* `LocalIsolator`: the CPU model, `Scicom-intl/voice-isolator-cpu`, distilled from
  the GPU release into a transformer that one CPU thread steps live. onnxruntime
  and numpy inside the agent process: no GPU, no torch, nothing to deploy.
* `RemoteIsolator`: the GPU release, `Scicom-intl/voice-isolator-m-wavlm` (M with a
  causal WavLM-Large branch, 555M parameters), on a GPU server that steps every
  call it holds in one batched step per 20 ms. One WebSocket per call. The server
  is the model repository's own `vi.server_mp` (see `Dockerfile`).

Both satisfy `Backend`: `open()` gives one call's stream, which takes 16 kHz
float32 audio in any amounts and returns the isolated audio that is ready. The
frame processor does not know or care which it has.

Both enroll the caller from the call itself: from 3 s in, the loudest 3 s heard
so far, re-made every half second, fixed once the call is 15.5 s old. Pass
`enrollment=` (16 kHz float32, a few seconds of the caller alone) to lock a known
voice in from the first frame instead.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .onnx_stream import OnnxStream, load_model

__all__ = ["Backend", "LocalIsolator", "RemoteIsolator", "Stream"]

logger = logging.getLogger(__name__)


class Stream(Protocol):
    """One call. Audio in, the isolated audio that is ready out."""

    def push(self, pcm: np.ndarray) -> np.ndarray:
        """`pcm` is mono float32 in [-1, 1] at 16 kHz, any length."""
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class Backend(Protocol):
    def open(self) -> Stream:
        """A stream for one call. Raises if the model or server is unusable."""
        ...


def _to_pcm16(x: np.ndarray) -> bytes:
    return np.clip(np.rint(x * 32768.0), -32768, 32767).astype("<i2").tobytes()


class LocalIsolator:
    """
    The CPU model in the agent process, on onnxruntime.

    `model` is a Hugging Face repository (default `Scicom-intl/voice-isolator-cpu`,
    private: an `HF_TOKEN` that can read it) or a local directory with
    `config.json`, `step.onnx` and `enroll.onnx`; `VOICE_ISOLATOR_MODEL` overrides
    the default. `quantized=True` takes the 8-bit export from the repository's
    `int8/` folder: about 2.7x cheaper per call, at the price of agreeing with the
    float model to tens of dB rather than to float rounding.

    One instance per process is enough: the sessions are shared and each `open()`
    starts a call from a clean state.
    """

    REPO = "Scicom-intl/voice-isolator-cpu"

    def __init__(
        self,
        model: str | None = None,
        *,
        quantized: bool = False,
        num_threads: int = 1,
        live_enroll: bool = True,
        enrollment: np.ndarray | None = None,
    ) -> None:
        self.model_dir = self._resolve(
            model or os.environ.get("VOICE_ISOLATOR_MODEL") or self.REPO, quantized
        )
        self.num_threads = num_threads
        self.live_enroll = live_enroll
        self.enrollment = (
            None if enrollment is None else np.asarray(enrollment, np.float32)
        )
        # Open the sessions now: a model that cannot load fails at session start,
        # not as audio passed through untouched on the first frame.
        load_model(self.model_dir, num_threads)

    @staticmethod
    def _resolve(model: str, quantized: bool) -> str:
        d = Path(model)
        if not d.is_dir():
            from huggingface_hub import snapshot_download

            files = ["config.json", "step.onnx", "enroll.onnx"]
            if quantized:
                files = [f"int8/{f}" for f in files]
            d = Path(snapshot_download(model, allow_patterns=files))
        if quantized and (d / "int8" / "config.json").is_file():
            d = d / "int8"
        if not (d / "config.json").is_file():
            raise FileNotFoundError(
                f"{d} holds no config.json: not a model written by Voice-Isolator's "
                "scripts/export_onnx.py"
            )
        return str(d.resolve())

    def open(self) -> OnnxStream:
        s = OnnxStream(
            self.model_dir, num_threads=self.num_threads, live_enroll=self.live_enroll
        )
        if self.enrollment is not None:
            s.set_enrollment(self.enrollment)
        return s


# One event loop on one daemon thread carries the WebSockets of every remote call
# in the process: `_process` is synchronous and runs on the agent's audio read
# loop, so the network has to live somewhere else.
_IO_LOOP: asyncio.AbstractEventLoop | None = None
_IO_LOCK = threading.Lock()


def _io_loop() -> asyncio.AbstractEventLoop:
    global _IO_LOOP
    with _IO_LOCK:
        if _IO_LOOP is None or not _IO_LOOP.is_running():
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name="voice-isolator-io", daemon=True
            ).start()
            _IO_LOOP = loop
        return _IO_LOOP


class RemoteIsolator:
    """
    The GPU release on a server, one WebSocket per call.

    The protocol is the model repository's server (`vi.server_mp`): a connection
    is a call; binary messages carry 16 kHz mono int16 PCM, any length, in both
    directions; a text message `{"enroll": "<base64 int16 PCM>"}` sets the
    caller's voice. Started with `--enroll-loud`, the server runs the same live
    enrollment rule as `LocalIsolator`.

    `url` defaults to `VOICE_ISOLATOR_URL`. An unreachable server raises at
    `open()`. A connection lost mid-call logs one warning and passes the caller's
    audio through untouched for the rest of the call, so the agent keeps hearing
    them; it does not reconnect, because the server's state for the call (its
    enrollment, its last 10 s) went with the connection.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        connect_timeout: float = 5.0,
        enrollment: np.ndarray | None = None,
    ) -> None:
        url = url or os.environ.get("VOICE_ISOLATOR_URL")
        if not url:
            raise ValueError(
                "RemoteIsolator needs url= or VOICE_ISOLATOR_URL (ws://host:port)"
            )
        self.url = url
        self.headers = headers or {}
        self.connect_timeout = connect_timeout
        self.enrollment = (
            None if enrollment is None else np.asarray(enrollment, np.float32)
        )

    def open(self) -> _RemoteStream:
        return _RemoteStream(
            self.url, self.headers, self.connect_timeout, self.enrollment
        )


class _RemoteStream:
    def __init__(self, url, headers, timeout, enrollment) -> None:
        self.url = url
        self._loop = _io_loop()
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._down = False
        self._closed = False
        fut = asyncio.run_coroutine_threadsafe(
            self._connect(headers, enrollment), self._loop
        )
        try:
            fut.result(timeout)
        except Exception as exc:  # noqa: BLE001 - surfaced as one clear error
            fut.cancel()
            raise ConnectionError(
                f"voice isolator server at {url} is unusable: {exc!r}"
            ) from exc

    async def _connect(self, headers, enrollment) -> None:
        import aiohttp

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(
                self.url, headers=headers, max_msg_size=0
            )
        except BaseException:
            await self._session.close()
            raise
        if enrollment is not None:
            pcm = base64.b64encode(_to_pcm16(enrollment)).decode("ascii")
            await self._ws.send_str(json.dumps({"enroll": pcm}))
        self._q: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._tasks = [
            asyncio.ensure_future(self._send()),
            asyncio.ensure_future(self._recv()),
        ]

    async def _send(self) -> None:
        try:
            while (b := await self._q.get()) is not None:
                await self._ws.send_bytes(b)
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)

    async def _recv(self) -> None:
        import aiohttp

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    with self._lock:
                        self._buf.extend(msg.data)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
            return
        self._fail(ConnectionError("the server closed the connection"))

    def _fail(self, exc: BaseException) -> None:
        if not self._down and not self._closed:
            logger.warning(
                "voice isolator server at %s lost mid-call (%r): passing the caller's "
                "audio through untouched for the rest of this call",
                self.url,
                exc,
            )
        self._down = True

    def push(self, pcm: np.ndarray) -> np.ndarray:
        if self._down:
            return np.asarray(pcm, np.float32)
        self._loop.call_soon_threadsafe(self._q.put_nowait, _to_pcm16(pcm))
        with self._lock:
            out = bytes(self._buf)
            self._buf.clear()
        return np.frombuffer(out, "<i2").astype(np.float32) / 32768.0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def _close() -> None:
            self._q.put_nowait(None)
            for t in self._tasks:
                t.cancel()
            await self._ws.close()
            await self._session.close()

        try:
            asyncio.run_coroutine_threadsafe(_close(), self._loop).result(2.0)
        except Exception:  # noqa: BLE001 - closing a dead connection is not an error
            pass
