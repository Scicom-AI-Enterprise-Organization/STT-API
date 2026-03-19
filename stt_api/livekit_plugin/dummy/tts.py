from __future__ import annotations

import logging
import uuid

from livekit.agents import tts
from livekit.agents.tts.tts import AudioEmitter
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("dummy-tts")

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
SILENCE_DURATION_S = 1.0


class TTS(tts.TTS):
    """Dummy TTS plugin that returns silent audio without calling any external API."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        num_channels: int = NUM_CHANNELS,
        silence_duration: float = SILENCE_DURATION_S,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._silence_duration = silence_duration

    @property
    def model(self) -> str:
        return "dummy"

    @property
    def provider(self) -> str:
        return "dummy"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> DummyChunkedStream:
        return DummyChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            silence_duration=self._silence_duration,
        )


class DummyChunkedStream(tts.ChunkedStream):
    """Dummy TTS stream that emits silent PCM audio."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        sample_rate: int,
        num_channels: int,
        silence_duration: float,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._silence_duration = silence_duration

    async def _run(self, output_emitter: AudioEmitter) -> None:
        request_id = str(uuid.uuid4())
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            mime_type="audio/pcm",
        )

        # Generate silent PCM audio (16-bit samples = 2 bytes per sample)
        num_samples = int(self._sample_rate * self._silence_duration) * self._num_channels
        silent_audio = b"\x00\x00" * num_samples

        output_emitter.push(silent_audio)
        output_emitter.flush()

        logger.debug("dummy TTS synthesized silence for: %s", self._input_text)
