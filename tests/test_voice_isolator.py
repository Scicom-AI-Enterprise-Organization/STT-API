"""Tests for the voice isolator plugin (stt_api/livekit_plugin/voice_isolator).

The CPU path runs on a tiny model with random weights and the released architecture
(band LSTMs, a frame LSTM, a ring of keys and values, the enrollment encoder):
tests/fixtures/voice_isolator_tiny, written by Voice-Isolator's
scripts/tiny_fixture.py through the same export as the real models; the agent
model's presence gate on tests/fixtures/voice_isolator_tiny_gate, the same with a
presence head (its scale set so that its gate opens and closes on a real call). The remote path
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
TINY_GATE = str(Path(__file__).parent / "fixtures" / "voice_isolator_tiny_gate")
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


def steady(
    seconds: float, level: float, seed: int = 0, rate: int = 16000
) -> np.ndarray:
    """speechlike() at a steady level: no 3 s of it is much louder than any other."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    f0 = 140 + 40 * np.sin(2 * np.pi * 0.7 * t)
    x = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / rate) / k for k in range(1, 6))
    x = level * (
        x * (np.sin(2 * np.pi * 1.3 * t) > -0.2) + 0.05 * rng.standard_normal(len(t))
    )
    return np.clip(np.rint(x * 32768), -32768, 32767).astype(np.float32) / 32768


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

    def test_voicebot_starts_on_the_decoy(self):
        # Until the caller is enrolled the model is told to keep a voice that is in
        # no call, rather than nothing: another vector, another output.
        from stt_api.livekit_plugin.voice_isolator.onnx_stream import OnnxStream

        assert not np.allclose(
            OnnxStream(TINY).emb, OnnxStream(TINY, voicebot=True).emb
        )
        x = speechlike(2.0, seed=16)
        a = run(VoiceIsolator(backend=LocalIsolator(TINY)), x)
        b = run(VoiceIsolator(backend=LocalIsolator(TINY, voicebot=True)), x)
        assert np.abs(a - b).max() > 10 * LSB

    def test_voicebot_waits_for_a_voice_over_the_start(self):
        # One level throughout (the greeting's bed and nobody else): nothing is 6 dB
        # over the first 3 s, so the first enrollment waits for the 10 s cap; the
        # default rule takes the same audio at 3 s.
        x = steady(11.0, 0.05, seed=17)
        vb = VoiceIsolator(backend=LocalIsolator(TINY, voicebot=True))
        run(vb, x, close=False)
        assert vb.enrolled_at == 10.0
        vb._close()
        default = VoiceIsolator(backend=LocalIsolator(TINY))
        run(default, x, close=False)
        assert default.enrolled_at == 3.0
        default._close()

    def test_voicebot_enrolls_a_caller_who_comes_in_louder(self):
        # A bed for 4 s, then a voice 12 dB over it: enrolled within about a second
        # of the voice, at a half second, once a 3 s window holding it is 6 dB up.
        x = np.concatenate([steady(4.0, 0.05, seed=18), steady(4.0, 0.2, seed=19)])
        vi = VoiceIsolator(backend=LocalIsolator(TINY, voicebot=True))
        run(vi, x, close=False)
        assert vi.enrolled_at is not None and 4.5 <= vi.enrolled_at <= 6.0
        vi._close()

    def test_a_longer_rule_enrolls_a_caller_who_answers_late(self):
        # A bed for 17 s, then a voice 12 dB over it. The rule ends at 15.5 s, on
        # the bed; asked to run for 60 s it moves to the voice.
        from stt_api.livekit_plugin.voice_isolator.onnx_stream import OnnxStream

        x = np.concatenate([steady(17.0, 0.05, seed=20), steady(5.0, 0.2, seed=21)])
        short, long = OnnxStream(TINY), OnnxStream(TINY, loud_within=60.0)
        for i in range(0, len(x), 800):
            short.push(x[i : i + 800])
            long.push(x[i : i + 800])
        assert (short.loud_st[0] + 300) / 100 <= 15.5  # a window of the bed
        assert long.loud_st[0] / 100 >= 16.5  # one that holds the voice

    def test_an_arrival_ends_the_rule(self):
        # arrive_db: a voice 6 dB over the call's first 3 s is the caller arriving,
        # and the rule ends arrive_hold seconds later. Something louder after that
        # (here from 12 s) no longer moves the enrollment; without an arrival it does.
        from stt_api.livekit_plugin.voice_isolator.onnx_stream import OnnxStream

        x = np.concatenate(
            [
                steady(4.0, 0.03, seed=22),
                steady(8.0, 0.12, seed=23),
                steady(4.0, 0.5, seed=24),
            ]
        )
        held = OnnxStream(TINY, loud_within=60.0, arrive_db=6.0, arrive_hold=5.0)
        free = OnnxStream(TINY, loud_within=60.0)
        for i in range(0, len(x), 800):
            held.push(x[i : i + 800])
            free.push(x[i : i + 800])
        assert held.arrived_at is not None and 4.5 <= held.arrived_at <= 6.0
        assert (held.loud_st[0] + 300) / 100 <= 12.0  # the caller, not the shout
        assert free.arrived_at is None and free.loud_st[0] / 100 >= 11.5

    def test_a_model_names_its_own_rule(self, tmp_path):
        # config.json "serve": the agent model asks for the longer rule itself.
        import shutil

        from stt_api.livekit_plugin.voice_isolator.onnx_stream import OnnxStream

        d = tmp_path / "model"
        shutil.copytree(TINY, d)
        cfg = json.loads((d / "config.json").read_text())
        cfg["serve"] = {
            "enroll": "loud",
            "loud_within": 60,
            "arrive_db": 6,
            "arrive_hold": 15,
        }
        (d / "config.json").write_text(json.dumps(cfg))
        s = LocalIsolator(str(d)).open()
        assert (s.loud_until, s.arrive_db, s.arrive_hold) == (60.5, 6.0, 15.0)
        s = OnnxStream(str(d), loud_within=15.0, arrive_db=None)  # the caller's say
        assert (s.loud_until, s.arrive_db) == (15.5, None)
        s = OnnxStream(TINY)  # a model that asks for nothing: the first 15 s
        assert (s.loud_until, s.arrive_db) == (15.5, None)

    def test_a_model_names_another_rule_for_the_voicebot_mode(self, tmp_path):
        # "serve" "voicebot": what the model changes of its serving in that mode.
        # Here: no enrollment by default, the longer rule once a voicebot's caller
        # has been heard.
        import shutil

        from stt_api.livekit_plugin.voice_isolator.onnx_stream import OnnxStream

        d = tmp_path / "model"
        shutil.copytree(TINY, d)
        cfg = json.loads((d / "config.json").read_text())
        cfg["serve"] = {
            "enroll": "none",
            "voicebot": {"enroll": "loud", "loud_within": 60, "arrive_db": 6},
        }
        (d / "config.json").write_text(json.dumps(cfg))
        s = OnnxStream(str(d))
        assert not s.live and not s.clip
        s = OnnxStream(str(d), voicebot=True)
        assert s.clip and (s.loud_until, s.arrive_db) == (60.5, 6.0)
        x = np.concatenate([steady(4.0, 0.05, seed=29), steady(4.0, 0.2, seed=30)])
        for i in range(0, len(x), 800):
            s.push(x[i : i + 800])
        assert 4.5 <= s.enrolled_at <= 6.0 and not np.array_equal(s.emb, s.null)
        assert s.arrived_at is not None

    def test_a_model_served_with_no_enrollment(self, tmp_path):
        # "serve" "enroll": "none": no live rule, the learned no-enrollment vector
        # for the whole call; live_enroll=True still enrolls it. In the voicebot
        # mode it waits on the decoy as any model does, then drops to no enrollment.
        import shutil

        from stt_api.livekit_plugin.voice_isolator.onnx_stream import OnnxStream

        d = tmp_path / "model"
        shutil.copytree(TINY, d)
        cfg = json.loads((d / "config.json").read_text())
        cfg["serve"] = {"enroll": "none"}
        (d / "config.json").write_text(json.dumps(cfg))
        x = np.concatenate([steady(4.0, 0.05, seed=27), steady(4.0, 0.2, seed=28)])
        none, off = OnnxStream(str(d)), OnnxStream(TINY, live_enroll=False)
        loud, vb = (
            OnnxStream(str(d), live_enroll=True),
            OnnxStream(str(d), voicebot=True),
        )
        decoy = vb.emb.copy()
        outs = [[], []]
        for i in range(0, len(x), 800):
            outs[0].append(none.push(x[i : i + 800]))
            outs[1].append(off.push(x[i : i + 800]))
            loud.push(x[i : i + 800])
            vb.push(x[i : i + 800])
        assert none.enrolled_at is None and np.array_equal(none.emb, none.null)
        assert np.array_equal(np.concatenate(outs[0]), np.concatenate(outs[1]))
        assert loud.enrolled_at == 3.0
        assert not np.allclose(decoy, vb.null)
        assert 4.5 <= vb.enrolled_at <= 6.0 and np.array_equal(vb.emb, vb.null)
        assert not vb.waiting and not vb.live

    def test_voicebot_needs_the_live_rule(self):
        with pytest.raises(ValueError):
            LocalIsolator(TINY, voicebot=True, enrollment=speechlike(3.0))
        with pytest.raises(ValueError):
            LocalIsolator(TINY, voicebot=True, live_enroll=False)

    def test_the_decoy_ships_with_the_plugin(self):
        from stt_api.livekit_plugin.voice_isolator.onnx_stream import DECOY

        d = np.load(DECOY)
        assert d.dtype == np.int16 and d.shape == (64000,)  # 4 s at 16 kHz
        assert np.abs(d).max() == 16384  # half of full scale
        assert (DECOY.parent / "LICENSE.voicebot_decoy").is_file()

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


def closed_hops(y: np.ndarray, hop: int = 160) -> np.ndarray:
    n = len(y) // hop
    return np.all(y[: n * hop].reshape(n, hop) == 0, 1)


def logit(p) -> np.ndarray:
    p = np.asarray(p, np.float64)
    return np.log(p / (1 - p)).astype(np.float32)


class TestGate:
    """The agent model's presence gate: closed frames leave as digital silence."""

    def test_takes_the_models_own_setting(self):
        s = LocalIsolator(TINY_GATE).open()
        assert s.gate == {"theta": 0.6, "hold": 3, "theta_off": None, "keep_db": None}

    def test_off_and_other_settings(self):
        assert LocalIsolator(TINY_GATE, gate="off").open().gate is None
        assert LocalIsolator(TINY_GATE, gate="0.9:100:0.2:1").open().gate == {
            "theta": 0.9,
            "hold": 100,
            "theta_off": 0.2,
            "keep_db": 1.0,
        }

    def test_a_model_without_a_head_has_none(self, local):
        assert local.open().gate is None

    def test_a_gate_needs_a_head(self):
        with pytest.raises(ValueError, match="presence head"):
            LocalIsolator(TINY, gate="0.6:3")

    def test_it_opens_and_closes(self):
        x = speechlike(6.0, seed=5)
        closed = closed_hops(isolate(TINY_GATE, x))
        assert 0.02 < closed.mean() < 0.98
        assert not closed_hops(isolate(TINY_GATE, x, gate="off")).any()

    def test_a_gate_that_never_closes_is_the_model_itself(self):
        x = speechlike(4.0, seed=6)
        y, y0 = isolate(TINY_GATE, x, gate="0:3"), isolate(TINY_GATE, x, gate="off")
        assert np.abs(y[160:] - y0[160:]).max() <= 1e-6

    def test_a_gate_that_never_opens_is_silence(self):
        assert not np.any(isolate(TINY_GATE, speechlike(3.0), gate="1.01:3"))

    def test_same_output_whatever_the_frame_size(self):
        x = speechlike(6.0, seed=7)
        out = [
            run(VoiceIsolator(backend=LocalIsolator(TINY_GATE)), x, ms=ms)
            for ms in (10, 20, 50)
        ]
        assert closed_hops(out[2]).any()
        for y in out[:2]:
            assert (closed_hops(y) == closed_hops(out[2])).all()
            assert np.abs(y - out[2]).max() <= 2 * LSB

    def test_the_rule(self):
        # Opens at theta, stays open while the head is at theta_off or over and
        # `hold` frames after; hop t is weighted g[t-1] w^2[hop:] + g[t] w^2[:hop].
        from stt_api.livekit_plugin.voice_isolator.onnx_stream import OnnxStream

        s = OnnxStream(TINY_GATE, gate="0.5:2:0.3")
        p = [0.1, 0.6, 0.4, 0.2, 0.2, 0.2, 0.2, 0.6]  # open: 0 1 1 1 1 0 0 1
        y = s._gate_out(
            np.ones(8 * 160, np.float32), np.ones(8 * 160, np.float32), logit(p)
        )
        h = y.reshape(8, 160).mean(1)
        assert h[0] == 0 and h[6] == 0
        assert np.allclose(h[2:5], 1.0, atol=1e-6)
        assert h[1] == pytest.approx(0.5, abs=0.01) and h[5] == pytest.approx(
            0.5, abs=0.01
        )
        assert h[7] == pytest.approx(0.5, abs=0.01)
        # the state carries over to the next step: still open there
        y = s._gate_out(
            np.ones(160, np.float32), np.ones(160, np.float32), logit([0.2])
        )
        assert np.allclose(y, 1.0, atol=1e-6)

    def test_a_kept_frame_counts_as_the_caller(self):
        # keep_db: a frame the model kept within that many dB of the line is open
        # whatever the head says; one it turned down 20 dB is not.
        from stt_api.livekit_plugin.voice_isolator.onnx_stream import OnnxStream

        line = np.full(4 * 160, 0.1, np.float32)
        low = logit([0.01] * 4)
        s = OnnxStream(TINY_GATE, gate="0.99:0:0.99:1")
        kept = s._gate_out(line.copy(), line, low).reshape(4, 160).mean(1)
        assert np.allclose(kept[1:], 0.1, atol=1e-6)
        s = OnnxStream(TINY_GATE, gate="0.99:0:0.99:1")
        assert not np.any(s._gate_out(0.1 * line, line, low))

    def test_what_a_kept_frame_counts_for_while_the_decoy_holds(self, tmp_path):
        # The voicebot mode: until the first enrollment the model is told to keep
        # nobody. A model says what a kept frame counts for there ("serve"
        # "wait_keep"): as ever (the default), nothing, or only in runs; once a
        # caller is enrolled it counts as ever again.
        import shutil

        from stt_api.livekit_plugin.voice_isolator.onnx_stream import OnnxStream

        def model(wait_keep):
            d = tmp_path / f"wk{wait_keep}"
            if not d.exists():
                shutil.copytree(TINY_GATE, d)
                cfg = json.loads((d / "config.json").read_text())
                cfg["serve"] = dict(cfg.get("serve") or {}, wait_keep=wait_keep)
                (d / "config.json").write_text(json.dumps(cfg))
            return str(d)

        line = np.full(6 * 160, 0.1, np.float32)
        low = logit([0.01] * 6)
        g = dict(gate="0.99:0:0.99:1", voicebot=True)
        s = OnnxStream(TINY_GATE, **g)  # no word from the model: as ever
        assert s.waiting and s.wait_keep == 1
        assert np.allclose(s._gate_out(line.copy(), line, low)[160:], 0.1, atol=1e-6)
        s = OnnxStream(model(0), **g)  # nothing: only the head opens the gate
        assert not np.any(s._gate_out(line.copy(), line, low))
        s = OnnxStream(model(3), **g)  # from the third kept frame in a row
        h = s._gate_out(line.copy(), line, low).reshape(6, 160).mean(1)
        assert not np.any(h[:2]) and np.allclose(h[3:], 0.1, atol=1e-6)
        s = OnnxStream(model(0), **g)  # enrolled: a kept frame counts again
        x = np.concatenate([steady(4.0, 0.05, seed=25), steady(3.0, 0.2, seed=26)])
        for i in range(0, len(x), 800):
            s.push(x[i : i + 800])
        assert s.enrolled_at is not None and not s.waiting
        kept = s._gate_out(line[:640].copy(), line[:640], low[:4]).reshape(4, 160)
        assert np.allclose(kept.mean(1)[1:], 0.1, atol=1e-6)


@pytest.fixture
def halving_server():
    """A stand-in for the GPU server: same protocol, halves every sample. `delay`:
    every reply that much later; `stall` (k, s): stop for s seconds at message k,
    then answer what waited at once."""
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web

    state = {
        "enroll": [],
        "close_after": None,
        "delay": 0.0,
        "stall": None,
        "query": [],
    }

    async def handler(request):
        state["query"].append(request.query_string)
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

    def test_voicebot_rides_on_the_url(self, halving_server):
        # The server reads the mode from the call's URL when the call opens, so it
        # holds from the first chunk.
        url, state = halving_server
        x = speechlike(0.5, seed=20)
        run(VoiceIsolator(backend=RemoteIsolator(url, voicebot=True)), x, ms=20)
        run(VoiceIsolator(backend=RemoteIsolator(url)), x, ms=20)
        assert state["query"] == ["voicebot=1", ""]
        assert RemoteIsolator("ws://h:1", voicebot=True).url == "ws://h:1/?voicebot=1"
        assert (
            RemoteIsolator("ws://h:1/x?a=b&voicebot=1", voicebot=False).url
            == "ws://h:1/x?a=b&voicebot=0"
        )
        with pytest.raises(ValueError):
            RemoteIsolator(url, voicebot=True, enrollment=speechlike(3.0))

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
