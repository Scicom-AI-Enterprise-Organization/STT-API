"""
Silero VAD scored on a server, as a LiveKit Agents `agents.vad.VAD`: a drop-in for
`livekit.plugins.silero.VAD`.

The agent keeps everything but the model: the windowing, LiveKit's ExpFilter
(alpha 0.35, as upstream), the state machine, the speech buffer and the events are
upstream silero's, line for line. Each 32 ms window goes to the server over a
WebSocket, and its probability comes back. The server (`server.py`) batches the
windows of every call it holds into one model run.

One connection per audio stream, not per VAD stream: streams of one VAD that are
pushed the same frame objects share a connection, a model state and every window's
score. An `AgentSession` with a non-streaming STT pushes its own VAD stream and its
`StreamAdapter`'s the same frames, so without this every window of a call went to the
server twice. Each stream keeps its own filter, state machine, speech buffer and
events. A stream joins the older connection that has taken (or is one frame short of)
the frame it holds, on its first frames and now and then after; only the younger
moves.

Failure: an unreachable server raises in `load()`, so a bad URL fails the worker at
prewarm, not the first call. A server lost mid-call, or one that has not answered a
window for `reply_timeout` seconds, fails over to LiveKit's own silero on this CPU
(`fallback=True`, the default) for the windows still in flight and every window
after, while the connection is retried in the background. With `fallback=False` those
windows score 0.0: the agent hears nobody until the server is back.

Delay: a window the server has not answered `max_lag` seconds (0.2) after it was sent
is scored here at once, and the server's late reply for it is dropped. The connection
stays up, so the server keeps the call's state and its replies take over again once
they are back within `max_lag`: a slow server is late, not lost. It is lost when it
sends nothing for `reply_timeout`, or when a reply it owes is 2 s old. Sends go through
a queue, so a stalled connection never holds up the audio, and a stream does not wait
for its connection to open: the first windows queue behind the handshake. The CPU
model takes over warmed on the second of audio before the window: started cold it is
up to 0.94 off in p(speech), warmed on 1 s at most 0.04.

`compensate_delay=True` takes the round trip off `min_silence_duration`, in whole
windows, so the end of a turn is reported when local silero would have reported it
rather than one round trip later. The price: a pause shorter by that much ends the
turn. Off by default.

What it is doing: `RemoteSileroVAD.stats()` (windows scored by the server, here, late;
round trip percentiles), logged every `log_every` seconds while there is traffic.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import os
import threading
import time
import weakref
from dataclasses import dataclass

import numpy as np
from livekit import agents, rtc
from livekit.agents import utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ..clearline.vad import _FAST_RATES, _Resampler

logger = logging.getLogger(__name__)

__all__ = ["RemoteSileroVAD", "RemoteSileroVADStream"]

SAMPLE_RATE = 16000
WINDOW = 512  # silero at 16 kHz: 32 ms
WINDOW_DURATION = WINDOW / SAMPLE_RATE
_BACKOFF = (0.5, 1.0, 2.0, 5.0)  # seconds between reconnects, the last repeating
_WARMUP_WINDOWS = 32  # 1 s of audio run through the CPU model before it takes over
_MAX_BEHIND = 2.0  # seconds: a server owing a reply this old is behind for good
_PROBATION = 8  # frames after it starts on each of which a stream looks for a connection to share
_RECHECK = 8  # and, while it is alone on its connection, every so many frames after
_LOOK_BACK = 16  # how many of its last frames it looks for an older link's last among
_DELAY_ALPHA = (
    0.05  # the weight of one window in the delay `compensate_delay` takes off
)


@dataclass
class _VADOptions:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    deactivation_threshold: float


def _local_weights_sha256() -> str | None:
    """sha256 of the silero graph livekit-plugins-silero ships here, if it is installed."""
    try:
        import importlib.resources

        res = (
            importlib.resources.files("livekit.plugins.silero.resources")
            / "silero_vad.onnx"
        )
        return hashlib.sha256(res.read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001 - only used for a warning
        return None


def _handshake(url: str, headers: dict[str, str], timeout: float) -> dict:
    """Connect once, read the server's hello, close. Runs its own loop on a thread:
    `load()` is synchronous and may be called from inside a running loop."""
    import aiohttp

    async def go() -> dict:
        async with aiohttp.ClientSession() as s:
            ws = await asyncio.wait_for(s.ws_connect(url, headers=headers), timeout)
            async with ws:
                msg = await ws.receive(timeout=timeout)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    raise ConnectionError(
                        f"expected the server's hello, got {msg.type}"
                    )
                return json.loads(msg.data)

    box: dict = {}

    def run() -> None:
        try:
            box["hello"] = asyncio.run(go())
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    t = threading.Thread(target=run, name="remote-silero-handshake", daemon=True)
    t.start()
    t.join(timeout + 1.0)
    if "hello" not in box:
        exc = box.get("error", TimeoutError("no hello"))
        # The type, not repr: a handshake error's repr carries our request headers.
        raise ConnectionError(
            f"silero VAD server at {url} is unusable: {type(exc).__name__}"
        ) from (exc if isinstance(exc, BaseException) else None)
    return box["hello"]


class RemoteSileroVAD(agents.vad.VAD):
    """
    Silero VAD on a GPU server, a drop-in for `livekit.plugins.silero.VAD`.

        from stt_api.livekit_plugin import RemoteSileroVAD

        def prewarm(proc):
            proc.userdata["vad"] = RemoteSileroVAD.load("ws://gpu-host:8766")

    `load()` connects once to check the server and blocks: call it in `prewarm`.
    """

    @classmethod
    def load(
        cls,
        url: str | None = None,
        *,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.55,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
        headers: dict[str, str] | None = None,
        connect_timeout: float = 5.0,
        reply_timeout: float = 0.5,
        max_lag: float = 0.2,
        fallback: bool = True,
        check_server: bool = True,
        compensate_delay: bool = False,
        log_every: float = 60.0,
    ) -> RemoteSileroVAD:
        """
        Args:
            url: the server, `ws://host:port`. Default `SILERO_VAD_URL`.
            min_speech_duration .. deactivation_threshold: as `silero.VAD.load`, same
                defaults (deactivation defaults to max(activation - 0.15, 0.01)).
            headers: sent with every WebSocket handshake (an auth header, say).
                Without an `Authorization` among them, `SILERO_VAD_API_KEY` is sent
                as `Authorization: Bearer <key>` when it is set.
            connect_timeout: seconds for a connection to open.
            reply_timeout: no reply at all for this long counts as the server lost.
            max_lag: a window unanswered this long is scored on this CPU at once.
            fallback: score windows with LiveKit's silero on this CPU while the server
                is unreachable. Needs `livekit-plugins-silero`. False scores them 0.0.
            check_server: connect once now and raise if the server is unusable.
            compensate_delay: take the round trip off `min_silence_duration`, in whole
                windows, so a turn's end is reported when local silero would have.
            log_every: seconds between two log lines of `stats()` while windows are
                being scored. 0: never.
        """
        url = url or os.environ.get("SILERO_VAD_URL")
        if not url:
            raise ValueError(
                "RemoteSileroVAD needs url= or SILERO_VAD_URL (ws://host:port)"
            )
        if is_given(deactivation_threshold) and deactivation_threshold <= 0:
            raise ValueError("deactivation_threshold must be greater than 0")
        headers = dict(headers or {})
        key = os.environ.get("SILERO_VAD_API_KEY")
        if key and not any(k.lower() == "authorization" for k in headers):
            headers["Authorization"] = f"Bearer {key}"
        session = None
        if fallback:
            try:
                from livekit.plugins.silero import onnx_model
            except ImportError as exc:
                raise ImportError(
                    "fallback=True scores windows locally while the server is down and "
                    "needs livekit-plugins-silero; install it or pass fallback=False"
                ) from exc
            session = onnx_model.new_inference_session(True)
        hello: dict = {}
        if check_server:
            hello = _handshake(url, headers, connect_timeout)
            if hello.get("sample_rate") != SAMPLE_RATE or hello.get("window") != WINDOW:
                raise ConnectionError(
                    f"silero VAD server at {url} serves {hello}, not 512-sample windows "
                    "at 16 kHz"
                )
            local = _local_weights_sha256()
            if local and hello.get("weights_sha256") not in (None, local):
                logger.warning(
                    "silero VAD server at %s runs other silero weights (%s, sha256 %s) "
                    "than livekit-plugins-silero here (sha256 %s): its probabilities, "
                    "and the fallback's, differ",
                    url,
                    hello.get("model"),
                    str(hello.get("weights_sha256"))[:12],
                    local[:12],
                )
        opts = _VADOptions(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            deactivation_threshold=deactivation_threshold
            if is_given(deactivation_threshold)
            else max(activation_threshold - 0.15, 0.01),
        )
        vad = cls(
            url=url,
            headers=headers,
            opts=opts,
            connect_timeout=connect_timeout,
            reply_timeout=reply_timeout,
            max_lag=max_lag,
            fallback_session=session,
            compensate_delay=compensate_delay,
            log_every=log_every,
        )
        vad._server_model = hello.get("model", "silero")
        return vad

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        opts: _VADOptions,
        connect_timeout: float,
        reply_timeout: float,
        max_lag: float = 0.2,
        fallback_session=None,
        compensate_delay: bool = False,
        log_every: float = 60.0,
    ) -> None:
        super().__init__(
            capabilities=agents.vad.VADCapabilities(update_interval=WINDOW_DURATION)
        )
        self._url, self._headers, self._opts = url, headers, opts
        self._connect_timeout, self._reply_timeout = connect_timeout, reply_timeout
        self._max_lag = max_lag
        self._fallback_session = fallback_session
        self._compensate_delay = compensate_delay
        self._streams = weakref.WeakSet[RemoteSileroVADStream]()
        self._links: list[_Link] = []  # the connections open, oldest first
        # id(frame) -> (the oldest link that took it, which of its frames it was), for
        # the frames in the links' rings (they are alive, so no id repeats): one
        # lookup tells a stream that a link is hearing its audio, however many
        # unrelated calls this VAD carries
        self._taken: dict[int, tuple[_Link, int]] = {}
        # streams running. With one, nothing can be shared: its link keeps no
        # record for a twin to find it by.
        self._n_live = 0
        self._born = 0
        self._server_model = "silero"
        # counters, for tests and dashboards: windows the server scored, and windows
        # scored here because it could not. A window is counted once, however many
        # streams share its connection.
        self.windows_remote = 0
        self.windows_local = 0
        # of windows_local, those scored here because the server was late, not lost
        self.windows_late = 0
        self.connects = (
            0  # connections opened, the first of each stream and every reconnect
        )
        self.outages = 0  # times a connection was given up on
        self._round_trips: collections.deque[float] = collections.deque(maxlen=4096)
        self._log_every = log_every
        self._logged_at = time.perf_counter()
        self._log_at = self._logged_at + log_every if log_every else float("inf")
        self._logged = (0, 0, 0)

    @property
    def model(self) -> str:
        return self._server_model

    @property
    def provider(self) -> str:
        return "remote-silero"

    def stream(self) -> RemoteSileroVADStream:
        s = RemoteSileroVADStream(self, self._opts)
        self._streams.add(s)
        # counted from here, not from its task's first step: a twin made in the same
        # breath must find the first frame the other took
        self._n_live += 1
        s._task.add_done_callback(self._stream_done)
        return s

    def _stream_done(self, _task) -> None:
        self._n_live -= 1

    def stats(self) -> dict:
        """What this VAD has done since it was loaded, and the round trip of the last
        4,096 windows the server scored. `local_share` is the number to watch: every
        window scored here is CPU the server was meant to save."""
        r = sorted(self._round_trips)

        def pct(q: float) -> float | None:
            return round(1e3 * r[min(len(r) - 1, int(q * len(r)))], 2) if r else None

        n = self.windows_remote + self.windows_local
        return {
            "windows": n,
            "windows_remote": self.windows_remote,
            "windows_local": self.windows_local,
            "windows_late": self.windows_late,
            "local_share": round(self.windows_local / n, 5) if n else None,
            "late_share": round(self.windows_late / n, 5) if n else None,
            "round_trip_ms_p50": pct(0.5),
            "round_trip_ms_p95": pct(0.95),
            "round_trip_ms_p99": pct(0.99),
            "streams": len(self._streams),
            "connections": len(self._links),
            "connects": self.connects,
            "outages": self.outages,
        }

    def _maybe_log(self, now: float) -> None:
        if now < self._log_at:
            return
        self._log_at = now + self._log_every
        was = self._logged
        self._logged = (self.windows_remote, self.windows_local, self.windows_late)
        remote, local, late = (a - b for a, b in zip(self._logged, was))
        span, self._logged_at = now - self._logged_at, now
        if not remote + local:
            return
        s = self.stats()
        logger.info(
            "silero VAD server at %s: %d windows in %.0f s, %.2f%% scored here "
            "(%.2f%% late), round trip p50 %s ms p99 %s ms, %d streams on %d "
            "connections",
            self._url,
            remote + local,
            span,
            100.0 * local / (remote + local),
            100.0 * late / (remote + local),
            s["round_trip_ms_p50"],
            s["round_trip_ms_p99"],
            s["streams"],
            s["connections"],
        )

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        # every stream shares this options object, as upstream's do
        for k, v in (
            ("min_speech_duration", min_speech_duration),
            ("min_silence_duration", min_silence_duration),
            ("prefix_padding_duration", prefix_padding_duration),
            ("max_buffered_speech", max_buffered_speech),
            ("activation_threshold", activation_threshold),
            ("deactivation_threshold", deactivation_threshold),
        ):
            if is_given(v):
                setattr(self._opts, k, v)
        for s in self._streams:
            s._resize_buffer()

    @property
    def min_silence_duration(self) -> float | None:
        return self._opts.min_silence_duration

    def _new_link(self, frame: rtc.AudioFrame) -> _Link:
        return _Link(self, frame)


class _Link:
    """Frames to probabilities, once, for every stream of a VAD that is pushed the same
    frames: the pending input, the resampler, the connection and what it still owes,
    and the CPU model that stands in for it. It remembers the last frames it took (by
    identity) so that another stream of the same VAD can tell it is being fed the same
    audio, and join."""

    RING = 128

    def __init__(self, vad: RemoteSileroVAD, frame: rtc.AudioFrame) -> None:
        vad._born += 1
        self.born = vad._born
        self._vad = vad
        self.input_rate = frame.sample_rate  # the rate frames arrive at
        self._lk_resampler: rtc.AudioResampler | None = None  # rates numpy does not do
        if frame.sample_rate in _FAST_RATES:
            self.rate = frame.sample_rate  # the rate the streams on it run at
        else:  # an odd rate: LiveKit's resampler to 16 kHz, and the streams run there
            self.rate = SAMPLE_RATE
            self._lk_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                quality=rtc.AudioResamplerQuality.QUICK,
            )
        self._rs = _Resampler(self.rate, WINDOW)
        self.need = self._rs.need  # input samples per window
        self._pend = np.empty(self.need + self.rate, np.int16)
        self._n = 0

        self.count = 0  # frames taken
        self.ring: list[rtc.AudioFrame | None] = [None] * self.RING
        self.subs: list[RemoteSileroVADStream] = []
        self.seq = 0  # windows made
        self.delivered = 0  # windows scored and handed to the streams
        # the last windows scored, each with the frame that completed it: a stream
        # that joins at a frame this link took a moment ago gets the windows completed
        # since, so two streams started on the same frame emit the same events
        # whichever task ran first
        self.recent: collections.deque[tuple] = collections.deque(maxlen=64)
        self.delay = 0.0  # how long after it was made a window's score comes, smoothed
        self._waiters: list[tuple[int, asyncio.Future]] = []

        # the connection: windows sent and not yet answered, oldest first, as
        # (input samples, event frame, time sent, 16 kHz int16 window, seq, frame no.)
        self._ws = None
        self._session = None
        self._gen = 0  # bumped on every connection, so a dead one's replies are dropped
        self._inflight: collections.deque = collections.deque()
        # the oldest windows in _inflight were scored here for being late: their
        # replies are still owed, and are dropped when they come
        self._answered_here = 0
        self._lag_timer: asyncio.TimerHandle | None = None
        # the last second of windows scored, to warm the CPU model on
        self._history: collections.deque = collections.deque(maxlen=_WARMUP_WINDOWS)
        self._local_in_step = False  # the CPU model scored the window before this one
        self._recv_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._down_since: float | None = None
        self._last_reply = 0.0
        self._local = None  # LiveKit's OnnxModel, made on the first failover
        self._closing = False
        self._outgoing: asyncio.Queue | None = None
        self._open_task: asyncio.Task | None = None
        self._start()

    def _start(self) -> None:
        # this connection's sends. The queue exists before the connection does: the
        # first windows queue behind the handshake rather than hold the audio up.
        self._outgoing = asyncio.Queue()
        self._open_task = asyncio.create_task(self._open())

    # ------------------------------------------------------------------ frames
    def last(self) -> rtc.AudioFrame | None:
        return self.ring[(self.count - 1) % self.RING] if self.count else None

    def first_from(self, k: int) -> int:
        """The first window completed by frame `k` or a later one."""
        for row in self.recent:
            if row[0] >= k:
                return row[1]
        for row in self._inflight:
            if row[5] >= k:
                return row[4]
        return self.seq

    def take(self, frame: rtc.AudioFrame) -> None:
        """A new frame: every window it completes is scored once, for every stream on
        this link."""
        n = self.count
        slot = n % self.RING
        vad = self._vad
        if vad._n_live > 1 or vad._taken:
            taken = vad._taken
            old = self.ring[slot]
            if old is not None:
                # leaving the ring. A younger link that took the same frames drops
                # an older one's entry a moment early: no stream looks that far back.
                taken.pop(id(old), None)
            hit = taken.get(id(frame))
            if hit is None or hit[0].born >= self.born or hit[0]._closing:
                taken[id(frame)] = (self, n)  # an older link's stays
        self.ring[slot] = frame
        self.count = n + 1
        data = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            data = data.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        if self._lk_resampler is not None:
            for g in self._lk_resampler.push(
                rtc.AudioFrame(data.tobytes(), frame.sample_rate, 1, len(data))
            ):
                self._samples(np.frombuffer(g.data, dtype=np.int16))
        else:
            self._samples(data)

    def _samples(self, data: np.ndarray) -> None:
        n = self._n + len(data)
        if n > len(self._pend):
            self._pend = np.concatenate((self._pend[: self._n], np.empty(n, np.int16)))
        self._pend[self._n : n] = data
        self._n = n
        need = self.need
        while self._n >= need:
            win = self._pend[:need].copy()
            rest = self._n - need
            if rest:
                self._pend[:rest] = self._pend[need : self._n]
            self._n = rest
            if self.rate == SAMPLE_RATE:
                x16 = win
            else:
                x16 = np.clip(np.rint(self._rs(win) * 32767.0), -32768, 32767).astype(
                    np.int16
                )
            out = rtc.AudioFrame(
                data=win.tobytes(),
                sample_rate=self.rate,
                num_channels=1,
                samples_per_channel=need,
            )
            self._score(win, out, x16)

    def reset(self) -> None:
        """A flush of the only stream on it: the model's state and the pending input."""
        if self._outgoing is not None:
            self._outgoing.put_nowait('{"reset": true}')
        if self._local is not None:
            self._local.reset()
        self._history.clear()
        self._local_in_step = False
        self._n = 0

    # ------------------------------------------------------------------ scores
    def _score(self, win, out, x16) -> None:
        now = time.perf_counter()
        seq, at = self.seq, self.count - 1
        self.seq += 1
        if self._ws is not None and self._inflight:
            # liveness, not lag: a backlog the server is working through (frames
            # pushed in a burst) is not an outage, until it is _MAX_BEHIND old
            oldest = self._inflight[0][2]
            if now - max(oldest, self._last_reply) > self._vad._reply_timeout:
                self._fail(TimeoutError("no reply within reply_timeout"))
            elif now - max(oldest, self._up_since) > _MAX_BEHIND:
                self._fail(TimeoutError("the server is too far behind"))
        if self._outgoing is None:
            self._score_here(win, out, x16, 0.0, seq, at)
            return
        self._inflight.append((win, out, now, x16, seq, at))
        self._arm_lag_timer()
        self._outgoing.put_nowait(x16.tobytes())

    def _deliver(self, p: float, win, out, delay: float, seq: int, at: int) -> None:
        vad = self._vad
        if vad._compensate_delay:
            self.delay += _DELAY_ALPHA * (delay - self.delay)
        self.delivered = seq + 1
        subs = self.subs
        if vad._n_live == 1 and len(subs) == 1:  # no stream to share with
            s = subs[0]
            if seq >= s._from_seq:
                s._on_window(p, win, out, delay)
        else:
            self.recent.append((at, seq, p, win, out, delay))
            for s in tuple(subs):
                if seq >= s._from_seq:
                    try:
                        s._on_window(p, win, out, delay)
                    except Exception:  # one stream's trouble is not the others'
                        logger.exception("remote silero: a stream failed on a window")
        if self._waiters:
            self._wake()

    def _wake(self) -> None:
        keep = []
        for seq, fut in self._waiters:
            if self.delivered >= seq:
                if not fut.done():
                    fut.set_result(None)
            else:
                keep.append((seq, fut))
        self._waiters = keep

    async def wait_delivered(self, seq: int) -> None:
        """Until the first `seq` windows have been scored: the windows already sent
        come back, or are given up on at `reply_timeout` and scored here."""
        if self.delivered >= seq:
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append((seq, fut))
        try:
            await asyncio.wait_for(fut, self._vad._reply_timeout)
        except asyncio.TimeoutError:
            self._fail(TimeoutError("no reply within reply_timeout"))

    def _score_here(
        self, win, out, x16: np.ndarray, delay: float, seq: int, at: int
    ) -> None:
        """Score one window on this CPU and emit it, `delay` s after it was sent."""
        self._vad.windows_local += 1
        if self._vad._fallback_session is None:
            p = 0.0  # fail closed: hold
        else:
            if self._local is None:
                from livekit.plugins.silero import onnx_model

                self._local = onnx_model.OnnxModel(
                    onnx_session=self._vad._fallback_session, sample_rate=SAMPLE_RATE
                )
            if not self._local_in_step:
                self._local.reset()
                for earlier in self._history:
                    self._local(earlier.astype(np.float32) / 32767.0)
                self._local_in_step = True
            p = self._local(x16.astype(np.float32) / 32767.0)
        self._history.append(x16)
        self._deliver(p, win, out, delay, seq, at)

    def _arm_lag_timer(self) -> None:
        """Wake when the oldest window still owed a score turns `max_lag` old."""
        if self._lag_timer is not None or len(self._inflight) <= self._answered_here:
            return
        sent = self._inflight[self._answered_here][2]
        self._lag_timer = asyncio.get_running_loop().call_later(
            max(0.0, sent + self._vad._max_lag - time.perf_counter()), self._on_lag
        )

    def _on_lag(self) -> None:
        self._lag_timer = None
        now = time.perf_counter()
        while len(self._inflight) > self._answered_here:
            win, out, sent, x16, seq, at = self._inflight[self._answered_here]
            if now - sent < self._vad._max_lag:
                break
            self._vad.windows_late += 1
            self._answered_here += 1
            self._score_here(win, out, x16, now - sent, seq, at)
        self._arm_lag_timer()

    def _cancel_lag_timer(self) -> None:
        if self._lag_timer is not None:
            self._lag_timer.cancel()
            self._lag_timer = None

    # ------------------------------------------------------------------ the connection
    _up_since = 0.0

    async def _open(self) -> None:
        await self._connect(first=True)

    async def _connect(self, first: bool = False) -> bool:
        import aiohttp

        gen = self._gen
        session = aiohttp.ClientSession()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(self._vad._url, headers=self._vad._headers),
                self._vad._connect_timeout,
            )
            hello = await ws.receive(timeout=self._vad._connect_timeout)
            if hello.type != aiohttp.WSMsgType.TEXT:
                raise ConnectionError(f"expected the server's hello, got {hello.type}")
        except asyncio.CancelledError:
            await session.close()
            raise
        except Exception as exc:  # noqa: BLE001
            await session.close()
            if gen == self._gen and (first or self._down_since is None):
                self._fail(exc)
            return False
        if gen != self._gen or self._closing:
            # given up on while it was opening (the windows queued for it were scored
            # here): the reconnect loop opens the next one
            await self._close(ws, session)
            return False
        self._session, self._ws = session, ws
        self._gen += 1
        self._vad.connects += 1
        self._up_since = self._last_reply = time.perf_counter()
        if not first:
            self._outgoing = asyncio.Queue()
            if self._history:
                # the server starts this call from zero: warm it on the last second,
                # its replies dropped as a late window's are
                sent = time.perf_counter()
                for x16 in self._history:
                    self._inflight.append((None, None, sent, x16, -1, -1))
                self._answered_here += len(self._history)
                self._outgoing.put_nowait(
                    b"".join(x16.tobytes() for x16 in self._history)
                )
        self._recv_task = asyncio.create_task(self._recv(ws, self._gen))
        self._send_task = asyncio.create_task(self._send(ws, self._outgoing, self._gen))
        if self._down_since is not None:
            logger.info(
                "silero VAD server at %s is back after %.1f s",
                self._vad._url,
                time.perf_counter() - self._down_since,
            )
            self._down_since = None
        return True

    async def _send(self, ws, outgoing: asyncio.Queue, gen: int) -> None:
        """One connection's sends, in order, off the audio's path."""
        try:
            while True:
                message = await outgoing.get()
                if isinstance(message, str):
                    await ws.send_str(message)
                else:
                    await ws.send_bytes(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the windows are scored by _fail
            if gen == self._gen:
                self._fail(exc)

    async def _recv(self, ws, gen: int) -> None:
        import aiohttp

        vad = self._vad
        try:
            async for msg in ws:
                if gen != self._gen:
                    return
                if msg.type == aiohttp.WSMsgType.BINARY:
                    t = self._last_reply = time.perf_counter()
                    for p in np.frombuffer(msg.data, dtype="<f4"):
                        if not self._inflight:
                            break
                        win, out, sent, x16, seq, at = self._inflight.popleft()
                        if self._answered_here:
                            self._answered_here -= 1
                            continue
                        vad.windows_remote += 1
                        vad._round_trips.append(t - sent)
                        self._history.append(x16)
                        self._local_in_step = False
                        self._deliver(float(p), win, out, t - sent, seq, at)
                    self._arm_lag_timer()
                    if t >= vad._log_at:
                        vad._maybe_log(t)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            if gen == self._gen:
                self._fail(exc)
            return
        if gen == self._gen and not self._closing:
            self._fail(ConnectionError("the server closed the connection"))

    def _fail(self, exc: BaseException) -> None:
        """The server is gone (or too slow): score what it still owes here, in order,
        and keep reconnecting in the background."""
        if self._closing:
            return
        ws, session = self._ws, self._session
        self._ws = self._session = None
        self._gen += 1
        self._stop_sending()
        if self._down_since is None:
            self._down_since = time.perf_counter()
            self._vad.outages += 1
            logger.warning(
                "silero VAD server at %s lost (%s): %s until it is back",
                self._vad._url,
                type(exc).__name__,
                "scoring on this CPU"
                if self._vad._fallback_session is not None
                else "every window scores 0.0",
            )
        self._cancel_lag_timer()
        for _ in range(self._answered_here):
            self._inflight.popleft()
        self._answered_here = 0
        now = time.perf_counter()
        while self._inflight:
            win, out, sent, x16, seq, at = self._inflight.popleft()
            self._score_here(win, out, x16, now - sent, seq, at)
        if ws is not None:
            asyncio.ensure_future(self._close(ws, session))
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        k = 0
        while not self._closing and self._ws is None:
            await asyncio.sleep(_BACKOFF[min(k, len(_BACKOFF) - 1)])
            k += 1
            if not self._closing and await self._connect():
                return

    def _stop_sending(self) -> None:
        self._outgoing = None
        if self._send_task is not None:
            self._send_task.cancel()
            self._send_task = None

    @staticmethod
    async def _close(ws, session) -> None:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
        await session.close()

    async def aclose(self) -> None:
        """The last stream has left."""
        self._closing = True
        ws, session = self._ws, self._session
        self._ws = self._session = None
        self._gen += 1
        self._cancel_lag_timer()
        self._stop_sending()
        for t in (self._recv_task, self._reconnect_task, self._open_task):
            if t is not None and t is not asyncio.current_task():
                t.cancel()
        for _, fut in self._waiters:
            if not fut.done():
                fut.set_result(None)
        self._waiters = []
        taken = self._vad._taken
        for f in self.ring:
            if f is not None and taken.get(id(f), (None,))[0] is self:
                del taken[id(f)]
        self.ring = [None] * self.RING
        if ws is not None:
            await self._close(ws, session)


class RemoteSileroVADStream(agents.vad.VADStream):
    def __init__(self, vad: RemoteSileroVAD, opts: _VADOptions) -> None:
        super().__init__(vad)
        self._vad: RemoteSileroVAD = vad
        self._opts = opts
        self._exp_filter = utils.ExpFilter(alpha=0.35)

        self._input_rate = 0  # the rate frames arrive at; 0 until the first frame
        self._rate = (
            0  # the rate the stream runs at (16 kHz for rates numpy does not do)
        )
        self._speech_buffer: np.ndarray | None = None
        self._speech_buffer_index = 0
        self._speech_buffer_max_reached = False
        self._prefix_padding_samples = 0

        # the link this stream is on, how many of its frames this stream has been
        # through, and the first of its windows this stream is owed
        self._link: _Link | None = None
        self._pos = 0
        self._from_seq = 0
        self._since = (
            0  # frames since this stream started, counted while it is not alone
        )
        self._recent: collections.deque[rtc.AudioFrame] = collections.deque(
            maxlen=_LOOK_BACK
        )  # the last of them
        self._reset_public_state()

    def _reset_public_state(self) -> None:
        self._pub_speaking = False
        self._pub_speech_duration = 0.0
        self._pub_silence_duration = 0.0
        self._pub_current_sample = 0
        self._pub_timestamp = 0.0
        self._speech_threshold_duration = 0.0
        self._silence_threshold_duration = 0.0

    def update_options(self, **kw) -> None:  # the VAD's options object is shared
        self._vad.update_options(**kw)

    def _resize_buffer(self) -> None:
        if not self._rate or self._speech_buffer is None:
            return
        self._prefix_padding_samples = int(
            self._opts.prefix_padding_duration * self._rate
        )
        self._speech_buffer.resize(
            int(self._opts.max_buffered_speech * self._rate)
            + self._prefix_padding_samples,
            refcheck=False,
        )
        self._speech_buffer_index = min(
            self._speech_buffer_index, len(self._speech_buffer)
        )
        self._speech_buffer_max_reached = False

    # ------------------------------------------------------------------ the loop
    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        vad = self._vad
        try:
            async for f in self._input_ch:
                if isinstance(f, self._FlushSentinel):
                    await self._flush()
                    continue
                if not isinstance(f, rtc.AudioFrame):
                    continue
                if self._input_rate and f.sample_rate != self._input_rate:
                    logger.error("a frame with another sample rate was already pushed")
                    continue
                e = self._link
                found = None
                if vad._n_live > 1:  # another stream may be hearing this audio
                    self._recent.append(f)
                    n = self._since = self._since + 1
                    if n <= _PROBATION or not n % _RECHECK:
                        found = self._find(f)
                        if found is not None and e is not None and e.delivered < e.seq:
                            # moving to an older link: this one's windows come back
                            # first
                            await e.wait_delivered(e.seq)
                            found = self._find(f)
                if found is None and e is not None and self._pos == e.count:
                    self._pos += 1  # the usual frame: this stream's link takes it
                    e.take(f)
                else:
                    self._feed(f, found)
            # end of input: the windows already sent still get their events
            if self._link is not None:
                await self._link.wait_delivered(self._link.seq)
        finally:
            gone = self._leave()
            if gone is not None:
                await gone.aclose()

    def _find(self, frame: rtc.AudioFrame) -> tuple[bool, _Link, int] | None:
        """An OLDER link of this VAD that is hearing the same frames, looked for by a
        stream alone on its link (or with none yet) on each of its first frames and
        now and then after. Only ever the younger moves, so a stream that has been
        running keeps its model state and its windows. Two ways to find one:
          it has taken this very frame already (this stream is behind it): (True,
            link, which of its frames this one was);
          its last frame is one this stream has been through (this stream is ahead,
            as when the other's task is the slower of the two): (False, link, where
            in `_recent` that frame is)."""
        e = self._link
        if not (
            (e is None or len(e.subs) == 1) and len(self._vad._links) > (e is not None)
        ):
            return None
        taken = self._vad._taken
        hit = taken.get(id(frame))
        if hit is not None:
            o, k = hit
            if (
                o is not e
                and not o._closing
                and o.ring[k % o.RING] is frame
                and o.input_rate == frame.sample_rate
                and (e is None or o.born < e.born)
            ):
                return True, o, k
        r = self._recent
        for back in range(2, min(len(r), _LOOK_BACK) + 1):
            f = r[-back]
            hit = taken.get(id(f))
            if hit is None:
                continue
            o, k = hit
            if (
                o is not e
                and not o._closing
                and k == o.count - 1
                and o.ring[k % o.RING] is f
                and o.input_rate == frame.sample_rate
                and (e is None or o.born < e.born)
            ):
                return False, o, len(r) - back
        return None

    def _feed(self, frame: rtc.AudioFrame, found) -> None:
        e = self._link
        if found is not None:
            behind, o, k = found
            if behind:
                first = o.first_from(k)
                self._join(o, k + 1, first)
                for _at, seq, p, win, out, delay in tuple(o.recent):
                    if seq >= first:  # completed before this stream was here
                        self._on_window(p, win, out, delay)
                return
            for f in list(self._recent)[k + 1 : -1]:
                o.take(f)  # the frames it has not had, in order
            self._join(o, o.count, o.seq)
            e = o
        if e is None:
            e = self._own_link(frame)
        if self._pos < e.count:
            # the link took this frame already, for another stream; the window it
            # may have completed comes here with that stream's
            k = self._pos
            if k >= e.count - e.RING and e.ring[k % e.RING] is not frame:
                # not the same audio after all: this stream goes its own way from here
                e = self._own_link(frame)
            else:
                self._pos += 1
                return
        self._pos += 1
        e.take(frame)

    def _own_link(self, frame: rtc.AudioFrame) -> _Link:
        e = self._vad._new_link(frame)
        self._vad._links.append(e)
        self._join(e, 0, 0)
        return e

    def _join(self, e: _Link, pos: int, from_seq: int) -> None:
        gone = self._leave()
        if gone is not None:
            asyncio.ensure_future(gone.aclose())
        e.subs.append(self)
        self._link, self._pos, self._from_seq = e, pos, from_seq
        if not self._rate:
            self._input_rate, self._rate = e.input_rate, e.rate
            self._prefix_padding_samples = int(
                self._opts.prefix_padding_duration * self._rate
            )
            self._speech_buffer = np.empty(
                int(self._opts.max_buffered_speech * self._rate)
                + self._prefix_padding_samples,
                dtype=np.int16,
            )

    def _leave(self) -> _Link | None:
        """Off its link; returns the link if that left it empty, to be closed."""
        e, self._link = self._link, None
        if e is None:
            return None
        if self in e.subs:
            e.subs.remove(self)
        if e.subs:
            return None
        if e in self._vad._links:
            self._vad._links.remove(e)
        return e

    async def _flush(self) -> None:
        # upstream resets everything between two windows: let the windows already
        # sent come back and run through the old state first. The model's state is
        # reset when this stream is alone on its link; a twin's flush leaves the
        # audio they share as it is.
        e = self._link
        if e is not None:
            await e.wait_delivered(e.seq)
            if len(e.subs) == 1:
                e.reset()
        self._exp_filter = utils.ExpFilter(alpha=0.35)
        self._speech_buffer_index = 0
        self._speech_buffer_max_reached = False
        if self._speech_buffer is not None:
            self._speech_buffer.fill(0)
        self._reset_public_state()

    # ------------------------------------------------------------------ upstream's state machine
    def _on_window(
        self, p: float, win: np.ndarray, frame: rtc.AudioFrame, duration: float
    ) -> None:
        """One window's probability: upstream silero's bookkeeping, events and state
        machine, line for line (as `clearline.vad.ClearlineVADStream._on_window`)."""
        opts = self._opts
        assert self._speech_buffer is not None
        p = self._exp_filter.apply(exp=1.0, sample=p)
        self._pub_current_sample += WINDOW
        self._pub_timestamp += WINDOW_DURATION

        available_space = len(self._speech_buffer) - self._speech_buffer_index
        to_copy_buffer = min(len(win), available_space)
        if to_copy_buffer > 0:
            self._speech_buffer[
                self._speech_buffer_index : self._speech_buffer_index + to_copy_buffer
            ] = win[:to_copy_buffer]
            self._speech_buffer_index += to_copy_buffer
        elif not self._speech_buffer_max_reached:
            self._speech_buffer_max_reached = True
            logger.warning(
                "max_buffered_speech reached, ignoring further data for the current speech input"
            )

        if self._pub_speaking:
            self._pub_speech_duration += WINDOW_DURATION
        else:
            self._pub_silence_duration += WINDOW_DURATION

        self._event_ch.send_nowait(
            agents.vad.VADEvent(
                type=agents.vad.VADEventType.INFERENCE_DONE,
                samples_index=self._pub_current_sample,
                timestamp=self._pub_timestamp,
                silence_duration=self._pub_silence_duration,
                speech_duration=self._pub_speech_duration,
                probability=p,
                inference_duration=duration,
                frames=[frame],
                speaking=self._pub_speaking,
                raw_accumulated_silence=self._silence_threshold_duration,
                raw_accumulated_speech=self._speech_threshold_duration,
            )
        )

        if p >= opts.activation_threshold or (
            self._pub_speaking and p > opts.deactivation_threshold
        ):
            self._speech_threshold_duration += WINDOW_DURATION
            self._silence_threshold_duration = 0.0
            if (
                not self._pub_speaking
                and self._speech_threshold_duration >= opts.min_speech_duration
            ):
                self._pub_speaking = True
                self._pub_silence_duration = 0.0
                self._pub_speech_duration = self._speech_threshold_duration
                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.START_OF_SPEECH,
                        samples_index=self._pub_current_sample,
                        timestamp=self._pub_timestamp,
                        silence_duration=self._pub_silence_duration,
                        speech_duration=self._pub_speech_duration,
                        frames=[self._copy_speech_buffer()],
                        speaking=True,
                    )
                )
        else:
            self._silence_threshold_duration += WINDOW_DURATION
            self._speech_threshold_duration = 0.0
            if not self._pub_speaking:
                self._reset_write_cursor()
            min_silence = opts.min_silence_duration
            if self._vad._compensate_delay and self._link is not None:
                # this window's score is `delay` late: count that many whole windows
                # less, so the event is no earlier than local silero's
                late = min(self._link.delay, self._vad._max_lag)
                min_silence -= int(late / WINDOW_DURATION) * WINDOW_DURATION
            if self._pub_speaking and self._silence_threshold_duration >= min_silence:
                self._pub_speaking = False
                self._pub_silence_duration = self._silence_threshold_duration
                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.END_OF_SPEECH,
                        samples_index=self._pub_current_sample,
                        timestamp=self._pub_timestamp,
                        silence_duration=self._pub_silence_duration,
                        speech_duration=max(
                            0.0,
                            self._pub_speech_duration
                            - self._silence_threshold_duration,
                        ),
                        frames=[self._copy_speech_buffer()],
                        speaking=False,
                    )
                )
                self._pub_speech_duration = 0.0
                self._reset_write_cursor()

    def _reset_write_cursor(self) -> None:
        assert self._speech_buffer is not None
        if self._speech_buffer_index <= self._prefix_padding_samples:
            return
        padding_data = self._speech_buffer[
            self._speech_buffer_index
            - self._prefix_padding_samples : self._speech_buffer_index
        ]
        self._speech_buffer_max_reached = False
        self._speech_buffer[: self._prefix_padding_samples] = padding_data
        self._speech_buffer_index = self._prefix_padding_samples

    def _copy_speech_buffer(self) -> rtc.AudioFrame:
        assert self._speech_buffer is not None
        return rtc.AudioFrame(
            sample_rate=self._rate,
            num_channels=1,
            samples_per_channel=self._speech_buffer_index,
            data=self._speech_buffer[: self._speech_buffer_index].tobytes(),
        )
