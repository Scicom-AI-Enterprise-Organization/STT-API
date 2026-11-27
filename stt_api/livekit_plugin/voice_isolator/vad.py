"""
The session's VAD from the voice isolator's own connection.

    backend = RemoteIsolator(url, vad=True)       # the server runs --vad
    vad = IsolatorVAD.load()                      # in prewarm, as silero.VAD.load()

    session = AgentSession(vad=vad, ...)
    await session.start(..., room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            sample_rate=16000,
            noise_cancellation=lambda params: VoiceIsolator(backend=backend),
        ),
    ))

A call that runs `RemoteIsolator` already has its audio on a GPU server. Asked with
`vad=True`, that server also scores the audio it sends back with Silero VAD and sends
p(speech) of every 512-sample window on the same connection. `VoiceIsolator` marks
each frame it returns with where the frame stands in that audio
(`frame.userdata[VAD_KEY]`), and this VAD reads the windows' scores by the mark: no
model run in the agent, no second connection, no second upload. Everything after the
score is `RemoteSileroVAD`'s, so upstream silero's: the filter, the state machine,
the speech buffer and the events, and one scoring for the streams of a session that
are pushed the same frames.

What is scored is the server's window, not the agent's. The server cuts the audio it
sends into windows from its first sample. The agent's windows start at the first
frame the VAD is pushed, and `VoiceIsolator` moves its output against the server's by
a few milliseconds when a reply is late (silence put in front) or a pause wins delay
back (a quiet piece dropped). So each of the agent's windows takes the score of the
last server window that ends inside it: the same audio up to 32 ms earlier. Turns
start and end where local silero puts them, within a window or two.

A window scores on this CPU, with LiveKit's silero warmed on the second before
(`RemoteSileroVAD`'s fallback), when: its frame carries no mark (the server does not
run `--vad`, the connection was lost and the caller is passed through, the frames
did not come from a `VoiceIsolator`, or the rate is not 8 / 16 / 24 / 32 / 48 kHz);
or its score has not come `max_lag` seconds after the window was complete. So a
session whose isolator cannot score is local silero, not a silent agent.
`stats()["local_share"]` says how often.
"""

from __future__ import annotations

import asyncio
import logging
import time

from livekit import rtc
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ..remote_silero.client import (
    SAMPLE_RATE,
    WINDOW,
    RemoteSileroVAD,
    _Link,
    _VADOptions,
)
from .isolator import VAD_KEY

logger = logging.getLogger(__name__)

__all__ = ["IsolatorVAD"]


class IsolatorVAD(RemoteSileroVAD):
    """Silero VAD scored by the voice isolator's server on the audio it returns: a
    drop-in for `livekit.plugins.silero.VAD` in a session that runs
    `VoiceIsolator(RemoteIsolator(url, vad=True))`."""

    @classmethod
    def load(  # type: ignore[override]
        cls,
        *,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.55,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
        max_lag: float = 0.1,
        compensate_delay: bool = False,
        log_every: float = 60.0,
    ) -> IsolatorVAD:
        """
        Args:
            min_speech_duration .. deactivation_threshold: as `silero.VAD.load`, same
                defaults (deactivation defaults to max(activation - 0.15, 0.01)).
            max_lag: a window whose score has not come this long after the window was
                complete is scored on this CPU. The score follows the audio it is of
                by about a millisecond, so this only runs out when the connection
                stalls after the audio came.
            compensate_delay, log_every: as `RemoteSileroVAD.load`.
        """
        if is_given(deactivation_threshold) and deactivation_threshold <= 0:
            raise ValueError("deactivation_threshold must be greater than 0")
        try:
            from livekit.plugins.silero import onnx_model
        except ImportError as exc:
            raise ImportError(
                "IsolatorVAD scores on this CPU whatever the isolator's server does "
                "not, and needs livekit-plugins-silero for it"
            ) from exc
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
        return cls(
            url="the voice isolator's connection",
            headers={},
            opts=opts,
            connect_timeout=0.0,
            reply_timeout=max(0.5, 2 * max_lag),
            max_lag=max_lag,
            fallback_session=onnx_model.new_inference_session(True),
            compensate_delay=compensate_delay,
            log_every=log_every,
        )

    @property
    def provider(self) -> str:
        return "voice-isolator"

    def _new_link(self, frame: rtc.AudioFrame) -> _Link:
        return _IsolatorLink(self, frame)


class _IsolatorLink(_Link):
    """`_Link` with the isolator's connection in place of its own: a window's score is
    looked up by where the window ends in the server's audio."""

    def _start(self) -> None:
        self._iso = None  # the call's _RemoteStream, from its frames' marks
        self._loop = asyncio.get_running_loop()
        self._taken = 0  # input samples taken, all frames
        self._frame_at = 0  # of them, those before the frame being taken
        # where that frame's first sample is in the server's audio (16 kHz samples);
        # None: the frame has no mark
        self._frame_src: float | None = None
        self._told = False  # a wake-up from the I/O thread is on its way

    # ------------------------------------------------------------------ frames
    def take(self, frame: rtc.AudioFrame) -> None:
        tag = getattr(frame, "userdata", {}).get(VAD_KEY)
        self._frame_at = self._taken
        if tag is None or self._lk_resampler is not None:
            self._frame_src = None
        else:
            iso, self._frame_src = tag
            if iso is not self._iso:
                if self._iso is not None:
                    self._iso.vad_unlisten(self._tell)
                self._iso = iso
                iso.vad_listen(self._tell)
        super().take(frame)

    def _samples(self, data) -> None:
        self._taken += len(data)
        super()._samples(data)

    def reset(self) -> None:
        # the server's state is the call's audio, not this stream's to reset: a flush
        # resets what is local
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
        iso, i = self._iso, -1
        if (
            self._frame_src is not None
            and iso is not None
            and iso.vad_ready is not None
            and not iso.down
        ):
            # the window ends at the last sample taken that is not still pending
            end = self._frame_src + (
                self._taken - self._n - self._frame_at
            ) * SAMPLE_RATE / (self.rate)
            i = int(end // WINDOW) - 1  # the last server window that ends by then
            if i < iso.vad_count - iso.VAD_RING:
                i = -1  # long gone
        if i < 0:
            self._here(now)  # in order: what is waiting first
            self._score_here(win, out, x16, 0.0, seq, at)
            return
        self._inflight.append((win, out, now, x16, seq, at, i))
        self._pump()

    def _pump(self) -> None:
        """Every waiting window whose score has come, in order."""
        iso, vad = self._iso, self._vad
        t = time.perf_counter()
        while self._inflight and self._inflight[0][6] < iso.vad_count:
            win, out, sent, x16, seq, at, i = self._inflight.popleft()
            p = iso.vad_p(i)
            if p is None:  # it came and went while this window waited
                self._score_here(win, out, x16, t - sent, seq, at)
                continue
            vad.windows_remote += 1
            vad._round_trips.append(t - sent)
            self._history.append(x16)
            self._local_in_step = False
            self._deliver(p, win, out, t - sent, seq, at)
        self._cancel_lag_timer()
        self._arm_lag_timer()
        vad._maybe_log(t)

    def _here(self, now: float) -> None:
        """Every waiting window, scored on this CPU."""
        self._cancel_lag_timer()
        while self._inflight:
            win, out, sent, x16, seq, at, _ = self._inflight.popleft()
            self._score_here(win, out, x16, now - sent, seq, at)

    def _on_lag(self) -> None:
        self._lag_timer = None
        now = time.perf_counter()
        while self._inflight:
            win, out, sent, x16, seq, at, _ = self._inflight[0]
            if now - sent < self._vad._max_lag:
                break
            self._inflight.popleft()
            self._vad.windows_late += 1
            self._score_here(win, out, x16, now - sent, seq, at)
        self._arm_lag_timer()

    def _tell(self) -> None:
        # on the isolator's I/O thread: more scores have come, or the connection went
        if not self._told and not self._closing:
            self._told = True
            try:
                self._loop.call_soon_threadsafe(self._woken)
            except RuntimeError:  # the agent's loop has closed
                pass

    def _woken(self) -> None:
        self._told = False
        if self._closing or self._iso is None:
            return
        if self._iso.down:
            self._here(time.perf_counter())
        elif self._inflight:
            self._pump()

    def _fail(self, exc: BaseException) -> None:
        if not self._closing:  # a wait for the scores ran out: they are scored here
            self._here(time.perf_counter())

    async def aclose(self) -> None:
        if self._iso is not None:
            self._iso.vad_unlisten(self._tell)
        await super().aclose()
