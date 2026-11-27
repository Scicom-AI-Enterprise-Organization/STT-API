"""
Tests for `IsolatorVAD` (stt_api/livekit_plugin/voice_isolator/vad.py): the session's
VAD read off the voice isolator's connection, whose server scores the audio it sends.

What would fail silently, and is guarded here:

* **The marks.** `VoiceIsolator` says where each frame it returns stands in the
  server's audio. It moves its output against the server's when a reply is late
  (silence in front, the caller's own audio past the bound) and when a pause wins
  delay back (a piece dropped). A mark that is off pairs every window with the score
  of other audio, and nothing errors.
* **A server that does not score.** One started without `--vad`, or older than the
  request, sends no scores. The session must then be local silero, value for value,
  not an agent waiting `max_lag` for every window.
* **A connection lost mid-call.** `VoiceIsolator` passes the caller through from then
  on; the VAD must go on scoring that audio here.

The server is a stand-in that speaks the GPU server's protocol (vi/ws_worker.py in
Voice-Isolator), returns the audio it is sent and scores it with LiveKit's silero.
The GPU server's own scores are checked where torch lives
(Voice-Isolator's scripts/test_silero_stream.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import wave

import numpy as np
import pytest

rtc = pytest.importorskip("livekit.rtc", reason="livekit not installed")
pytest.importorskip("onnxruntime", reason="livekit extra not installed")
aiohttp = pytest.importorskip("aiohttp", reason="livekit extra not installed")
pytest.importorskip(
    "livekit.plugins.silero", reason="livekit-plugins-silero is the reference"
)

from aiohttp import web  # noqa: E402
from livekit import agents  # noqa: E402
from livekit.plugins import silero  # noqa: E402
from livekit.plugins.silero import onnx_model  # noqa: E402

from stt_api.livekit_plugin.voice_isolator import (  # noqa: E402
    IsolatorVAD,
    RemoteIsolator,
    VoiceIsolator,
)
from stt_api.livekit_plugin.voice_isolator.isolator import VAD_KEY  # noqa: E402

RATE = 16000
SPEECH_WAV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stt_api",
    "livekit_plugin",
    "dummy",
    "audio",
    "tawaran.wav",
)


def speech(rate: int = RATE, pad_s: float = 1.0) -> np.ndarray:
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
    z = np.zeros(int(pad_s * rate), np.float32)
    return np.concatenate([z, x, z])


@pytest.fixture
def scoring_server():
    """A stand-in for `vi.server_mp --vad`: every message's audio comes back as it
    was sent, and a call that asked with ?vad=1 also gets silero's p(speech) of
    each 512-sample window of what it was sent back. `scores`: answer the request at
    all (False: a server without --vad). `close_after`: hang up after that many
    messages. `hold`: seconds the scores come after their audio."""
    state = {"scores": True, "close_after": None, "hold": 0.0, "query": [], "p": []}
    session = onnx_model.new_inference_session(True)

    async def handler(request):
        state["query"].append(request.query_string)
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        loop = asyncio.get_running_loop()
        scoring = state["scores"] and request.query.get("vad") == "1"
        if scoring:
            await ws.send_str(json.dumps({"vad_ready": {"window": 512}}))
        model = onnx_model.OnnxModel(onnx_session=session, sample_rate=RATE)
        pend = np.zeros(0, np.int16)
        n = at = 0
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.BINARY:
                continue
            n += 1
            await ws.send_bytes(msg.data)
            if scoring:
                pend = np.concatenate([pend, np.frombuffer(msg.data, "<i2")])
                ps = []
                while len(pend) >= 512:
                    ps.append(float(model(pend[:512].astype(np.float32) / 32767.0)))
                    pend = pend[512:]
                if ps:
                    state["p"].extend(ps)
                    text = json.dumps({"vad": ps, "at": at})
                    at += len(ps)
                    if state["hold"]:
                        loop.call_later(
                            state["hold"],
                            lambda t=text: asyncio.ensure_future(ws.send_str(t)),
                        )
                    else:
                        await ws.send_str(text)
            if state["close_after"] and n >= state["close_after"]:
                await ws.close()
                break
        return ws

    loop = asyncio.new_event_loop()
    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = site._server.sockets[0].getsockname()[1]
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield f"ws://127.0.0.1:{port}/", state
    asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(2)
    loop.close()


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


def turns(rows) -> list[tuple[str, float]]:
    return [(r[0], r[2]) for r in rows if r[0] != "inference_done"]


def a_call(
    vi: VoiceIsolator,
    vad,
    x: np.ndarray,
    rate: int = RATE,
    ms: int = 50,
    streams: int = 1,
) -> tuple[list[list[tuple]], list[rtc.AudioFrame], list[float]]:
    """A call in real time, as an agent's audio loop runs it: every frame through the
    isolator, what comes back pushed to `streams` streams of `vad`. Returns each
    stream's events, the frames the isolator returned, and every window's delay."""
    n = rate * ms // 1000
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    got: list[rtc.AudioFrame] = []
    rows: list[list[tuple]] = [[] for _ in range(streams)]
    delays: list[float] = []

    async def go():
        ss = [vad.stream() for _ in range(streams)]

        async def drain(i):
            async for ev in ss[i]:
                rows[i].append(_row(ev))
                if i == 0 and ev.type == agents.vad.VADEventType.INFERENCE_DONE:
                    delays.append(ev.inference_duration)

        tasks = [asyncio.create_task(drain(i)) for i in range(streams)]
        t0 = time.perf_counter()
        for k in range(len(pcm) // n):
            g = vi._process(
                rtc.AudioFrame(pcm[k * n : (k + 1) * n].tobytes(), rate, 1, n)
            )
            got.append(g)
            for s in ss:
                s.push_frame(g)
            await asyncio.sleep(
                max(0.0, t0 + (k + 1) * ms / 1000 - time.perf_counter())
            )
        for s in ss:
            s.end_input()
        await asyncio.wait_for(asyncio.gather(*tasks), 10.0)
        for s in ss:
            await s.aclose()

    asyncio.run(go())
    vi._close()
    return rows, got, delays


def silero_on(frames: list[rtc.AudioFrame]) -> list[tuple]:
    """Upstream silero's events for the same frames."""

    async def go():
        s = silero.VAD.load().stream()
        rows: list[tuple] = []

        async def drain():
            async for ev in s:
                rows.append(_row(ev))

        task = asyncio.create_task(drain())
        for f in frames:
            s.push_frame(
                rtc.AudioFrame(bytes(f.data), f.sample_rate, 1, f.samples_per_channel)
            )
        s.end_input()
        await task
        await s.aclose()
        return rows

    return asyncio.run(go())


# ----------------------------------------------------------------------------- the marks


class _Scripted:
    """A backend whose stream returns a known signal late and in bursts: sample k of
    its output is `tone(k)`, quiet enough for `VoiceIsolator` to drop pieces of."""

    vad = True
    delay_s = 0.0
    down = False

    def __init__(self, stalls: dict[int, int]) -> None:
        self.stalls = stalls  # call number -> calls nothing comes back for, from it
        self.calls = 0
        self.sent = 0  # samples returned so far
        self.owed = 0  # samples taken and not yet returned
        self.until = 0

    @staticmethod
    def tone(k: np.ndarray) -> np.ndarray:
        return (100 + (k * 7) % 80).astype(np.float32) / 32768.0  # 0.003 .. 0.0055

    def open(self):
        return self

    def vad_p(self, i):
        return None

    def push(self, pcm: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.owed += len(pcm)
        if self.calls in self.stalls:
            self.until = self.calls + self.stalls[self.calls]
        if self.calls < self.until:
            return np.zeros(0, np.float32)
        out = self.tone(np.arange(self.sent, self.sent + self.owed))
        self.sent, self.owed = self.sent + self.owed, 0
        return out

    def close(self) -> None:
        pass


@pytest.mark.parametrize("rate_ms", [20, 50])
def test_every_frame_says_where_it_stands_in_the_servers_audio(rate_ms):
    # Replies held for a few frames at a time: silence goes in front, the bound is
    # passed and the caller's own audio (negative here) goes out, isolated audio is
    # skipped, and in the quiet that follows pieces are dropped to win delay back.
    # Through all of it, sample i of a frame marked `at` is sample at + i of what
    # the server sent, wherever the frame carries the server's audio at all.
    per = 50 // rate_ms  # the same stalls in seconds, whatever the frame
    backend = _Scripted(
        {5 * per: 1, 20 * per: 3 * per, 60 * per: per, 100 * per: 5 * per}
    )
    vi = VoiceIsolator(backend=backend, wait_ms=0, max_lag_ms=40.0)
    n = RATE * rate_ms // 1000
    raw = np.full(n, -3000, np.int16)  # the caller's own audio: never the tone
    checked = back = on = 0
    last = None
    for _ in range(int(12.0 * 1000 / rate_ms)):
        g = vi._process(rtc.AudioFrame(raw.tobytes(), RATE, 1, n))
        stream, at = g.userdata[VAD_KEY]
        assert stream is backend
        y = np.frombuffer(g.data, np.int16)
        k = np.arange(n) + int(at)
        want = np.rint(backend.tone(k) * 32768.0).astype(np.int16)
        isolated = (y != 0) & (y != -3000)
        assert at == int(at)
        assert np.array_equal(y[isolated], want[isolated]), "a mark is off"
        checked += int(isolated.sum())
        if last is not None:
            back += at < last + n  # silence went in front
            on += at > last + n  # a piece was dropped, or skipped
        last = at
    assert checked > 8 * RATE, "hardly any isolated audio was checked"
    assert back >= 2 and on >= 2, f"the output hardly moved: {back} back, {on} on"
    assert vi.unfiltered_ms > 0, "the bound was never passed"


def test_a_local_backend_marks_nothing():
    from pathlib import Path

    from stt_api.livekit_plugin.voice_isolator import LocalIsolator

    tiny = str(Path(__file__).parent / "fixtures" / "voice_isolator_tiny")
    vi = VoiceIsolator(backend=LocalIsolator(tiny))
    g = vi._process(rtc.AudioFrame(np.zeros(800, np.int16).tobytes(), RATE, 1, 800))
    assert VAD_KEY not in g.userdata
    vi._close()


# ----------------------------------------------------------------------------- the VAD


def test_the_servers_scores_cut_the_turns_local_silero_cuts(scoring_server):
    url, state = scoring_server
    x = np.concatenate([speech(), speech()])
    vad = IsolatorVAD.load()
    vi = VoiceIsolator(backend=RemoteIsolator(url, vad=True))
    (rows,), frames, delays = a_call(vi, vad, x)
    assert "vad=1" in state["query"][0]
    windows = len(frames) * 800 // 512
    assert sum(r[0] == "inference_done" for r in rows) == windows, "a window lost"
    st = vad.stats()
    assert st["windows_remote"] > 0.97 * windows, st
    assert st["windows"] == windows
    assert max(delays) < 0.1 + 0.03, f"a window waited {max(delays):.3f} s"
    assert np.percentile(delays, 50) < 0.01, "the scores do not come with the audio"
    ref, got = turns(silero_on(frames)), turns(rows)
    assert [k for k, _ in got] == [k for k, _ in ref]
    assert len([k for k, _ in got if k == "start_of_speech"]) == 2
    # the server's window ends up to one window before the agent's
    assert all(abs(a - b) <= 2 * 0.032 + 1e-6 for (_, a), (_, b) in zip(got, ref))
    # and the scores are the server's own: every one is a score it sent
    sent = np.array(state["p"])
    assert len(sent) >= st["windows_remote"]


def test_at_24k_too(scoring_server):
    url, _ = scoring_server
    x = speech(24000)
    vad = IsolatorVAD.load()
    vi = VoiceIsolator(backend=RemoteIsolator(url, vad=True))
    (rows,), frames, _ = a_call(vi, vad, x, rate=24000)
    windows = len(frames) * 1200 // 768
    assert sum(r[0] == "inference_done" for r in rows) == windows
    assert vad.windows_remote > 0.9 * windows
    ref, got = turns(silero_on(frames)), turns(rows)
    assert [k for k, _ in got] == [k for k, _ in ref]
    assert all(abs(a - b) <= 3 * 0.032 + 1e-6 for (_, a), (_, b) in zip(got, ref))


def test_a_server_that_does_not_score_leaves_local_silero(scoring_server):
    url, state = scoring_server
    state["scores"] = False  # started without --vad: the request is ignored
    x = speech()
    vad = IsolatorVAD.load()
    vi = VoiceIsolator(backend=RemoteIsolator(url, vad=True))
    (rows,), frames, delays = a_call(vi, vad, x)
    assert vad.windows_remote == 0 and vad.windows_late == 0
    assert max(delays) == 0.0, "a window waited for a score that never comes"
    assert rows == silero_on(frames), "it is not upstream silero, value for value"


def test_without_the_isolator_it_is_local_silero():
    # frames that never went through a VoiceIsolator carry no mark
    x = speech()
    n = 320
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    frames = [
        rtc.AudioFrame(pcm[k : k + n].tobytes(), RATE, 1, n)
        for k in range(0, len(pcm) - n + 1, n)
    ]
    vad = IsolatorVAD.load()

    async def go():
        s = vad.stream()
        rows = []

        async def drain():
            async for ev in s:
                rows.append(_row(ev))

        task = asyncio.create_task(drain())
        for f in frames:
            s.push_frame(f)
        s.end_input()
        await task
        await s.aclose()
        return rows

    assert asyncio.run(go()) == silero_on(frames)
    assert vad.windows_remote == 0


def test_a_connection_lost_mid_call_goes_on_locally(scoring_server):
    url, state = scoring_server
    x = np.concatenate([speech(), speech(), speech()])
    state["close_after"] = len(x) // 800 // 3  # a third of the way in
    vad = IsolatorVAD.load()
    vi = VoiceIsolator(backend=RemoteIsolator(url, vad=True))
    (rows,), frames, delays = a_call(vi, vad, x)
    windows = len(frames) * 800 // 512
    assert sum(r[0] == "inference_done" for r in rows) == windows, "a window lost"
    assert vad.windows_remote > windows // 5 and vad.windows_local > windows // 2
    assert max(delays) < 0.1 + 0.03
    assert len([k for k, _ in turns(rows) if k == "start_of_speech"]) == 3


def test_scores_that_come_late_are_made_here(scoring_server):
    url, state = scoring_server
    state["hold"] = 0.3  # every score 0.3 s after its audio: past max_lag
    x = speech()
    vad = IsolatorVAD.load(max_lag=0.1)
    vi = VoiceIsolator(backend=RemoteIsolator(url, vad=True))
    (rows,), frames, delays = a_call(vi, vad, x)
    windows = len(frames) * 800 // 512
    assert sum(r[0] == "inference_done" for r in rows) == windows
    assert vad.windows_late > 0.9 * windows
    assert max(delays) < 0.1 + 0.06, f"a window waited {max(delays):.3f} s"
    ref, got = turns(silero_on(frames)), turns(rows)
    assert got == ref, "late scores are local silero's, window for window"


def test_twin_streams_are_scored_once(scoring_server):
    url, _ = scoring_server
    x = speech()
    vad = IsolatorVAD.load()
    vi = VoiceIsolator(backend=RemoteIsolator(url, vad=True))
    (a, b), frames, _ = a_call(vi, vad, x, streams=2)
    windows = len(frames) * 800 // 512
    assert a == b
    assert vad.windows_remote + vad.windows_local == windows, "counted twice"
    assert not vad._links
