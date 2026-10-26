"""Tests for the voice isolator plugin (stt_api/livekit_plugin/voice_isolator).

The CPU path runs on a tiny model with random weights and the released architecture
(band LSTMs, a frame LSTM, a ring of keys and values, the enrollment encoder):
tests/fixtures/voice_isolator_tiny, written by Voice-Isolator's
scripts/tiny_fixture.py through the same export as the real models. The remote path
runs against a stand-in server that speaks the GPU server's protocol and halves
whatever it is sent.

What the LiveKit pipeline depends on: a frame back for every frame in, at the same
rate and length. What the model promises: the output does not depend on the frame
size (the context is the call's state, not the frame's) and calls share nothing.
The real models' equality with PyTorch is checked where torch lives, in
Voice-Isolator's export (124-134 dB on real calls).
"""

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

rtc = pytest.importorskip("livekit.rtc", reason="livekit not installed")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

from stt_api.livekit_plugin.voice_isolator import (  # noqa: E402
    LocalIsolator,
    RemoteIsolator,
    VoiceIsolator,
    isolate,
)

TINY = str(Path(__file__).parent / "fixtures" / "voice_isolator_tiny")
LSB = 1 / 32768


def speechlike(seconds: float, seed: int = 0, rate: int = 16000) -> np.ndarray:
    """Harmonic bursts with pauses and a changing level: the live rule has a
    definite loudest window to pick, as it would on speech."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    f0 = 140 + 40 * np.sin(2 * np.pi * 0.7 * t)
    x = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / rate) / k for k in range(1, 6))
    env = (np.sin(2 * np.pi * 1.3 * t) > -0.2) * (0.3 + 0.7 * (t / seconds))
    x = 0.2 * x * env + 0.01 * rng.standard_normal(len(t))
    return np.clip(np.rint(x * 32768), -32768, 32767).astype(np.float32) / 32768


def with_silences(seconds: float, seed: int = 0, rate: int = 16000) -> np.ndarray:
    """speechlike() with digital silence in its pauses, as the isolated output of a
    caller's pause is: where the plugin may win back delay."""
    x = speechlike(seconds, seed, rate)
    t = np.arange(len(x)) / rate
    return np.where(np.sin(2 * np.pi * 1.3 * t) > -0.2, x, 0.0).astype(np.float32)


def frames(x: np.ndarray, rate: int, ms: int):
    n = rate * ms // 1000
    for i in range(0, len(x) - n + 1, n):
        pcm = np.clip(np.rint(x[i : i + n] * 32768.0), -32768, 32767).astype(np.int16)
        yield rtc.AudioFrame(pcm.tobytes(), rate, 1, n)


def run(
    vi: VoiceIsolator,
    x: np.ndarray,
    rate: int = 16000,
    ms: int = 50,
    pause: float = 0.0,
    close: bool = True,
):
    out = []
    for f in frames(x, rate, ms):
        g = vi._process(f)
        assert g.samples_per_channel == f.samples_per_channel
        assert g.sample_rate == f.sample_rate and g.num_channels == 1
        out.append(
            np.frombuffer(g.data, np.int16, count=g.samples_per_channel) / 32768.0
        )
        if pause:
            time.sleep(pause)
    if close:
        vi._close()
    return np.concatenate(out)


@pytest.fixture(scope="module")
def local():
    return LocalIsolator(TINY)


class TestLocal:
    def test_frame_contract(self, local):
        x = speechlike(2.0)
        for ms in (10, 20, 50):
            assert len(run(VoiceIsolator(backend=local), x, ms=ms)) == len(x)

    def test_same_output_whatever_the_frame_size(self, local):
        # The point of carrying the context as state: 10 ms frames see what 50 ms
        # frames see. Only float rounding may differ.
        x = speechlike(6.0)
        a = run(VoiceIsolator(backend=local), x, ms=10)
        b = run(VoiceIsolator(backend=local), x, ms=20)
        c = run(VoiceIsolator(backend=local), x, ms=50)
        assert np.abs(a - c).max() <= 2 * LSB and np.abs(b - c).max() <= 2 * LSB

    def test_is_the_stream_ten_ms_later(self, local):
        x = speechlike(5.0, seed=1)
        y = run(VoiceIsolator(backend=local), x, ms=50)
        ref = isolate(TINY, x)[: len(y) - 160]
        ref = np.clip(np.rint(ref * 32768), -32768, 32767) / 32768
        assert np.abs(y[160 : 160 + len(ref)] - ref).max() <= 2 * LSB

    def test_reports_its_delay(self, local):
        vi = VoiceIsolator(backend=local)
        run(vi, speechlike(1.0), ms=50, close=False)
        assert vi.delay_ms == pytest.approx(10.0)
        vi._close()

    def test_calls_share_nothing(self, local):
        xa, xb = speechlike(4.0, seed=2), speechlike(4.0, seed=3)
        solo = run(VoiceIsolator(backend=local), xa, ms=20)
        va, vb = VoiceIsolator(backend=local), VoiceIsolator(backend=local)
        out = []
        for fa, fb in zip(frames(xa, 16000, 20), frames(xb, 16000, 20)):
            g = va._process(fa)
            vb._process(fb)
            out.append(np.frombuffer(g.data, np.int16) / 32768.0)
        va._close()
        vb._close()
        assert np.abs(np.concatenate(out) - solo).max() <= LSB

    def test_live_enrollment_switches_on_at_three_seconds(self, local):
        vi = VoiceIsolator(backend=local)
        run(vi, speechlike(2.5), ms=50, close=False)
        assert vi.enrolled_at is None
        run(vi, speechlike(1.0, seed=4), ms=50, close=False)
        assert vi.enrolled_at == 3.0

    def test_a_known_voice_from_the_first_frame(self):
        x = speechlike(2.0, seed=5)
        live = run(VoiceIsolator(backend=LocalIsolator(TINY)), x)
        known = run(
            VoiceIsolator(
                backend=LocalIsolator(TINY, enrollment=speechlike(3.0, seed=6))
            ),
            x,
        )
        assert np.abs(live - known).max() > 10 * LSB

    def test_other_rates_come_back_at_their_rate(self, local):
        x = speechlike(2.0, rate=48000)
        assert len(run(VoiceIsolator(backend=local), x, rate=48000, ms=50)) == len(x)

    def test_stereo_passes_through(self, local):
        pcm = (speechlike(0.05) * 32767).astype(np.int16)
        f = rtc.AudioFrame(np.repeat(pcm, 2).tobytes(), 16000, 2, len(pcm))
        assert VoiceIsolator(backend=local)._process(f) is f

    def test_a_missing_model_raises_at_once(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LocalIsolator(str(tmp_path))


@pytest.fixture
def halving_server():
    """A stand-in for the GPU server: same protocol, halves every sample. `delay`:
    every reply that much later; `stall` (k, s): stop for s seconds at message k,
    then answer what waited at once."""
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web

    state = {"enroll": [], "close_after": None, "delay": 0.0, "stall": None}

    async def handler(request):
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        loop = asyncio.get_running_loop()
        n = 0
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                a = np.frombuffer(msg.data, "<i2").astype(np.int32) // 2
                b = a.astype("<i2").tobytes()
                n += 1
                if state["stall"] and n == state["stall"][0]:
                    await asyncio.sleep(state["stall"][1])
                if state["delay"]:
                    loop.call_later(
                        state["delay"],
                        lambda b=b: asyncio.ensure_future(ws.send_bytes(b)),
                    )
                else:
                    await ws.send_bytes(b)
                if state["close_after"] and n >= state["close_after"]:
                    await ws.close()
                    break
            elif msg.type == aiohttp.WSMsgType.TEXT:
                state["enroll"].append(json.loads(msg.data)["enroll"])
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


def lag_at(y: np.ndarray, x: np.ndarray, seg: slice, max_lag: int) -> int:
    """The delay at which the output segment `seg` is exactly the halved input."""
    half = np.floor(x * 32768 / 2) / 32768
    errs = [
        np.abs(y[seg] - half[seg.start - d : seg.stop - d]).max()
        for d in range(max_lag)
    ]
    return int(np.argmin(errs))


class TestRemote:
    def test_round_trip(self, halving_server):
        # Replies come on the server's time. A frame that finds none is answered with
        # silence in front, so the delay may grow during a call; what must hold is
        # that what comes back is the halved input, in order, within the lag bound.
        url, _ = halving_server
        x = speechlike(2.0, seed=7)
        y = run(VoiceIsolator(backend=RemoteIsolator(url)), x, ms=20, pause=0.005)
        half = np.floor(x * 32768 / 2) / 32768
        for seg in (slice(16000, 20000), slice(len(y) - 4000, len(y))):
            d = lag_at(y, x, seg, 4000)
            err = np.abs(y[seg] - half[seg.start - d : seg.stop - d]).max()
            assert err <= 2 * LSB, f"output at {seg} is not the halved input at any lag"

    def test_reports_its_delay(self, halving_server):
        # delay_ms is the lag at which the output is the input at the end of the call,
        # plus the model's own 10 ms (the stand-in server answers without it)
        url, _ = halving_server
        x = speechlike(2.0, seed=10)
        vi = VoiceIsolator(backend=RemoteIsolator(url))
        y = run(vi, x, ms=20, pause=0.005, close=False)
        d = lag_at(y, x, slice(len(y) - 4000, len(y)), 4000)
        assert vi.delay_ms == pytest.approx(10.0 + d / 16.0)
        vi._close()

    def test_a_slow_server_costs_at_most_max_lag(self, halving_server):
        # the non-waiting plugin, replies 0.4 s late against a 0.25 s bound: the delay
        # stops at the bound and the call goes out unfiltered (the input itself, not
        # the halved input)
        url, state = halving_server
        state["delay"] = 0.4
        x = speechlike(2.0, seed=11)
        vi = VoiceIsolator(backend=RemoteIsolator(url), wait_ms=0, max_lag_ms=250.0)
        y = run(vi, x, ms=20, pause=0.02, close=False)
        assert vi.delay_ms == pytest.approx(260.0)
        assert vi.unfiltered_ms > 1000.0  # all but the first 0.65 s or so
        seg = slice(len(y) - 8000, len(y))
        assert np.abs(y[seg] - x[seg.start - 4000 : seg.stop - 4000]).max() <= LSB
        vi._close()

    def test_delay_from_a_stall_is_won_back_in_pauses(self, halving_server):
        # the non-waiting plugin: the delay grows through a stall and is won back later
        url, state = halving_server
        state["stall"] = (25, 0.2)  # half a second in, no reply for 0.2 s
        vi = VoiceIsolator(backend=RemoteIsolator(url), wait_ms=0, max_lag_ms=250.0)
        delays = []
        for f in frames(with_silences(6.0, seed=12), 16000, 20):
            vi._process(f)
            delays.append(vi.delay_ms)
            time.sleep(0.02)
        vi._close()
        assert max(delays) >= 150.0
        assert delays[-1] <= 60.0

    def test_each_frame_waits_for_its_own_reply(self, halving_server):
        # 50 ms frames, replies 5 ms after each: waiting answers a frame with its own
        # reply (only the model's 10 ms behind); not waiting, with the frame before's
        url, state = halving_server
        state["delay"] = 0.005
        x = speechlike(2.0, seed=13)
        waiting = VoiceIsolator(backend=RemoteIsolator(url))
        run(waiting, x, ms=50, pause=0.02, close=False)
        at_once = VoiceIsolator(
            backend=RemoteIsolator(url), wait_ms=0, max_lag_ms=250.0
        )
        run(at_once, x, ms=50, pause=0.02, close=False)
        assert waiting.delay_ms <= 20.0
        assert at_once.delay_ms >= 50.0
        waiting._close()
        at_once._close()

    def test_a_late_reply_goes_out_unfiltered_rather_than_late(self, halving_server):
        # replies 0.1 s late against the default 40 ms cap: the delay stops at the cap
        # and the late part of the call goes out as the input itself
        url, state = halving_server
        state["delay"] = 0.1
        x = speechlike(2.0, seed=14)
        vi = VoiceIsolator(backend=RemoteIsolator(url))
        run(vi, x, ms=50, pause=0.02, close=False)
        assert vi.delay_ms <= 50.0
        assert vi.unfiltered_ms > 500.0
        vi._close()

    def test_waiting_wins_back_the_delay_of_a_stall(self, halving_server):
        url, state = halving_server
        state["stall"] = (10, 0.2)  # half a second in, no reply for 0.2 s
        vi = VoiceIsolator(backend=RemoteIsolator(url))
        delays = []
        for f in frames(with_silences(12.0, seed=15), 16000, 50):
            vi._process(f)
            delays.append(vi.delay_ms)
            time.sleep(0.02)
        vi._close()
        assert max(delays) >= 40.0  # the stall built delay up to the cap
        assert delays[-1] <= 30.0  # probing in the pauses won it back

    def test_enrollment_is_sent_first(self, halving_server):
        url, state = halving_server
        clip = speechlike(3.0, seed=8)
        vi = VoiceIsolator(backend=RemoteIsolator(url, enrollment=clip))
        vi._process(next(frames(clip, 16000, 20)))
        time.sleep(0.1)
        vi._close()
        assert len(state["enroll"]) == 1

    def test_an_unreachable_server_raises_at_once(self):
        with pytest.raises(ConnectionError):
            VoiceIsolator(
                backend=RemoteIsolator("ws://127.0.0.1:9", connect_timeout=2.0)
            )

    def test_a_lost_server_passes_the_caller_through(self, halving_server):
        url, state = halving_server
        state["close_after"] = 5
        x = speechlike(1.0, seed=9)
        y = run(VoiceIsolator(backend=RemoteIsolator(url)), x, ms=20, pause=0.01)
        last = slice(len(y) - 3200, len(y))
        assert np.abs(y[last] - x[last]).max() <= LSB
