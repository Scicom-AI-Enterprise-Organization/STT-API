"""
Tests for `stt_api/livekit_plugin/remote_silero/`: silero scored on a server that
batches every call's windows.

What would fail silently in production, and is guarded here:

* **Parity.** The plugin claims to be `silero.VAD` with the model moved. At 16 kHz,
  with the server running LiveKit's own graph, every event must be upstream's, value
  for value. A drift in the state machine or the window bookkeeping would only show
  up as turns cut a little differently.
* **Isolation in a batch.** Two calls scored in one batch must not see each other's
  state: each gives exactly what it gives alone.
* **A server that goes away.** The agent must keep hearing the caller (the local
  fallback scores the windows in flight and after), and come back to the server when
  it returns. A server that accepts and never answers must count as gone.

The server here is the `onnx` backend on CPU in a thread of this process; the GPU
backend's parity is measured on the GPU box (README.md).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import wave

import numpy as np
import pytest

rtc = pytest.importorskip("livekit.rtc", reason="livekit not installed")
pytest.importorskip("onnxruntime", reason="livekit extra not installed")
pytest.importorskip("aiohttp", reason="livekit extra not installed")
pytest.importorskip(
    "livekit.plugins.silero", reason="livekit-plugins-silero is the reference"
)

from aiohttp import web  # noqa: E402
from livekit import agents  # noqa: E402
from livekit.plugins import silero  # noqa: E402

from stt_api.livekit_plugin.remote_silero import RemoteSileroVAD  # noqa: E402
from stt_api.livekit_plugin.remote_silero import server as srv  # noqa: E402

RATE = 16000
SPEECH_WAV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stt_api",
    "livekit_plugin",
    "dummy",
    "audio",
    "tawaran.wav",
)


def load_speech(rate: int) -> np.ndarray:
    if not os.path.exists(SPEECH_WAV):
        pytest.skip(f"speech fixture missing: {SPEECH_WAV}")
    w = wave.open(SPEECH_WAV)
    x = (
        np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        / 32768.0
    )
    if w.getframerate() != rate:
        n = int(len(x) * rate / w.getframerate())
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(
            np.float32
        )
    return x


def padded(rate: int, pad_s: float = 1.0) -> np.ndarray:
    z = np.zeros(int(pad_s * rate), np.float32)
    return np.concatenate([z, load_speech(rate), z])


class Server:
    """The real server app (onnx backend, CPU) on a loop of its own thread."""

    def __init__(self, app_factory=None, **kw) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.app_factory = app_factory or (
            lambda: srv.make_app(srv.OnnxBackend(None, "cpu"), **kw)
        )
        self.port = 0
        self.start()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(30)

    def start(self) -> None:
        async def go():
            self.app = self.app_factory()
            self.runner = web.AppRunner(self.app, shutdown_timeout=1.0)
            await self.runner.setup()
            site = web.TCPSite(self.runner, "127.0.0.1", self.port)
            await site.start()
            self.port = site._server.sockets[0].getsockname()[1]

        self._run(go())
        self.url = f"ws://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._run(self.runner.cleanup())

    def stats(self) -> dict:
        b = self.app[srv.BATCHER]
        return {"windows": b.stats["windows"], "batches": b.stats["batches"]}

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


def _row(ev) -> tuple:
    return (
        ev.type.value,
        ev.samples_index,
        round(ev.timestamp, 6),
        round(ev.speech_duration, 6),
        round(ev.silence_duration, 6),
        round(float(ev.probability), 6),
        ev.speaking,
        sum(f.samples_per_channel for f in ev.frames),
    )


async def _drive(
    vad,
    x: np.ndarray,
    rate: int,
    frame_ms: int = 20,
    pace: float = 0.0,
    flush_at: float | None = None,
    on_frame=None,
) -> list[tuple]:
    """Every event of one stream fed `x` in `frame_ms` frames; `pace` seconds of sleep
    after each frame (0: as fast as the loop goes)."""
    stream = vad.stream()
    rows: list[tuple] = []

    async def drain():
        async for ev in stream:
            rows.append(_row(ev))

    task = asyncio.create_task(drain())
    n = rate * frame_ms // 1000
    pcm = np.clip(np.rint(x * 32767.0), -32768, 32767).astype(np.int16)
    for i, k in enumerate(range(0, len(pcm) - n + 1, n)):
        if flush_at is not None and k == int(flush_at * rate) // n * n:
            stream.flush()
        stream.push_frame(rtc.AudioFrame(pcm[k : k + n].tobytes(), rate, 1, n))
        if on_frame is not None:
            on_frame(i)
        await asyncio.sleep(pace)
    stream.end_input()
    await task
    await stream.aclose()
    return rows


def events(vad, x, rate, **kw) -> list[tuple]:
    return asyncio.run(_drive(vad, x, rate, **kw))


def turns(rows) -> list[tuple[str, float]]:
    return [(r[0], r[2]) for r in rows if r[0] != "inference_done"]


# ----------------------------------------------------------------------------- parity


def test_at_16k_every_event_is_upstream_silero(server):
    x = padded(RATE)
    ref = events(silero.VAD.load(), x, RATE)
    got = events(RemoteSileroVAD.load(server.url), x, RATE)
    assert len(ref) > 100 and any(r[0] == "start_of_speech" for r in ref)
    assert got == ref


@pytest.mark.parametrize("rate", [24000, 48000])
def test_other_rates_cut_the_same_turns(server, rate):
    # Upstream resamples with LiveKit's resampler, this plugin with Clearline's numpy
    # one: p moves in the third decimal, the turns must not.
    x = padded(rate)
    ref = turns(events(silero.VAD.load(), x, rate))
    got = turns(events(RemoteSileroVAD.load(server.url), x, rate))
    assert [k for k, _ in got] == [k for k, _ in ref]
    assert max(abs(a - b) for (_, a), (_, b) in zip(got, ref)) <= 0.1


def test_a_flush_resets_as_upstream_does(server):
    x = padded(RATE)
    ref = events(silero.VAD.load(), x, RATE, flush_at=2.0)
    got = events(RemoteSileroVAD.load(server.url), x, RATE, flush_at=2.0)
    assert got == ref


def test_the_hello_names_livekits_own_weights(server, caplog):
    # the onnx backend serves LiveKit's graph: no weights warning
    with caplog.at_level("WARNING"):
        RemoteSileroVAD.load(server.url)
    assert "other silero weights" not in caplog.text


# ----------------------------------------------------------------------------- batching


def test_calls_in_one_batch_do_not_share_state(server):
    # Two streams, different audio, pushed in step so the server batches them.
    a = padded(RATE)
    b = np.roll(a, RATE // 3)
    vad = RemoteSileroVAD.load(server.url)
    alone_a, alone_b = events(vad, a, RATE), events(vad, b, RATE)

    async def both():
        return await asyncio.gather(
            _drive(vad, a, RATE, pace=0.001), _drive(vad, b, RATE, pace=0.001)
        )

    before = server.stats()
    got_a, got_b = asyncio.run(both())
    after = server.stats()
    windows = after["windows"] - before["windows"]
    batches = after["batches"] - before["batches"]
    assert windows / batches > 1.2, "the two calls were never batched together"
    assert got_a == alone_a and got_b == alone_b
    assert alone_a != alone_b


def test_batched_onnx_backend_matches_one_call_at_a_time():
    be = srv.OnnxBackend(None, "cpu")
    rng = np.random.default_rng(0)
    pcm = (padded(RATE) * 32767).astype(np.int16)
    n = len(pcm) // 512
    wins = pcm[: n * 512].reshape(n, 512)
    other = (rng.standard_normal((n, 512)) * 3000).astype(np.int16)
    alone = np.array(
        [
            be.infer(w[None], np.array([0]), [0] if i == 0 else [])[0]
            for i, w in enumerate(wins)
        ]
    )
    pair = np.array(
        [
            be.infer(np.stack([o, w]), np.array([1, 2]), [1, 2] if i == 0 else [])[1]
            for i, (w, o) in enumerate(zip(wins, other))
        ]
    )
    assert np.abs(alone - pair).max() < 1e-5


# ----------------------------------------------------------------------------- failure


def test_an_unreachable_server_raises_at_load():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(ConnectionError):
        RemoteSileroVAD.load(f"ws://127.0.0.1:{port}", connect_timeout=1.0)


def test_no_url_is_refused(monkeypatch):
    monkeypatch.delenv("SILERO_VAD_URL", raising=False)
    with pytest.raises(ValueError):
        RemoteSileroVAD.load()


def test_a_server_lost_mid_call_falls_back_and_comes_back():
    s = Server()
    try:
        x = np.concatenate([padded(RATE), padded(RATE), padded(RATE)])
        vad = RemoteSileroVAD.load(s.url)
        n_frames = len(x) // (RATE // 50)

        def on_frame(i):
            if i == n_frames // 4:
                s.stop()  # gone at a quarter
            elif i == n_frames // 2:
                s.start()  # back at half, on the same port

        rows = events(vad, x, RATE, pace=0.004, on_frame=on_frame)
        windows = len(x) // 512
        assert sum(r[0] == "inference_done" for r in rows) == windows, "a window lost"
        assert vad.windows_local > 0, "the outage was never scored locally"
        assert vad.windows_remote > windows // 3, "it never came back to the server"
        starts = [t for k, t in turns(rows) if k == "start_of_speech"]
        assert len(starts) == 3, f"each of the three utterances is heard: {starts}"
    finally:
        s.close()


def test_without_fallback_the_outage_hears_nobody():
    s = Server()
    try:
        vad = RemoteSileroVAD.load(s.url, fallback=False)
        s.stop()
        rows = events(vad, padded(RATE), RATE)
        assert rows and all(r[5] == 0.0 for r in rows if r[0] == "inference_done")
        assert not turns(rows)
    finally:
        s.close()


def test_a_server_that_never_answers_counts_as_gone():
    def mute_app():
        async def ws(request):
            w = web.WebSocketResponse()
            await w.prepare(request)
            await w.send_str(
                json.dumps({"model": "mute", "sample_rate": 16000, "window": 512})
            )
            async for _ in w:  # reads, never answers
                pass
            return w

        app = web.Application()
        app.router.add_get("/{tail:.*}", ws)
        return app

    s = Server(app_factory=mute_app)
    try:
        vad = RemoteSileroVAD.load(s.url, reply_timeout=0.2)
        x = padded(RATE)
        rows = events(vad, x, RATE, pace=0.004)
        assert sum(r[0] == "inference_done" for r in rows) == len(x) // 512
        assert vad.windows_remote == 0 and vad.windows_local == len(x) // 512
        assert any(k == "start_of_speech" for k, _ in turns(rows))
    finally:
        s.close()


def test_update_options_reaches_open_streams(server):
    vad = RemoteSileroVAD.load(server.url)

    async def go():
        stream_rows = asyncio.create_task(_drive(vad, padded(RATE), RATE, pace=0.001))
        await asyncio.sleep(0.05)  # the stream is open and running
        vad.update_options(min_speech_duration=30.0)
        return await stream_rows

    assert not turns(asyncio.run(go()))


def test_capabilities_report_the_window(server):
    assert RemoteSileroVAD.load(server.url).capabilities.update_interval == 0.032
    assert isinstance(RemoteSileroVAD.load(server.url), agents.vad.VAD)


def test_a_keyed_server_refuses_a_call_without_the_key():
    s = Server(
        app_factory=lambda: srv.make_app(srv.OnnxBackend(None, "cpu"), api_key="k3y")
    )
    try:
        with pytest.raises(ConnectionError):
            RemoteSileroVAD.load(s.url, connect_timeout=2.0)
        with pytest.raises(ConnectionError):
            RemoteSileroVAD.load(
                s.url, headers={"Authorization": "Bearer nope"}, connect_timeout=2.0
            )
        vad = RemoteSileroVAD.load(s.url, headers={"Authorization": "Bearer k3y"})
        rows = events(vad, padded(RATE), RATE)
        assert vad.windows_remote > 0 and vad.windows_local == 0
        assert any(k == "start_of_speech" for k, _ in turns(rows))
    finally:
        s.close()
