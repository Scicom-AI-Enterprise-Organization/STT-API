from __future__ import annotations

import logging
import os
import uuid

import soundfile as sf

from livekit.agents import tts
from livekit.agents.tts.tts import AudioEmitter
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("dummy-tts")

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
AUDIO_FILE = os.path.join(os.path.dirname(__file__), "audio", "tawaran.wav")

_cached_audio: bytes | None = None


def _load_audio() -> bytes:
    global _cached_audio
    if _cached_audio is None:
        data, sr = sf.read(AUDIO_FILE, dtype="int16")
        if len(data.shape) > 1:
            data = data[:, 0]
        _cached_audio = data.tobytes()
        logger.info("loaded TTS audio: %s (%d samples, %dHz)", AUDIO_FILE, len(data), sr)
    return _cached_audio


class TTS(tts.TTS):
    """Dummy TTS that plays back a pre-recorded audio file.

    Outputs real human speech so that echo-based load tests produce audio
    that silero VAD can detect, triggering the full pipeline.
    """

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        num_channels: int = NUM_CHANNELS,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

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
        )


class DummyChunkedStream(tts.ChunkedStream):
    """Dummy TTS stream that emits pre-recorded speech audio."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        sample_rate: int,
        num_channels: int,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._sample_rate = sample_rate
        self._num_channels = num_channels

    async def _run(self, output_emitter: AudioEmitter) -> None:
        request_id = str(uuid.uuid4())
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            mime_type="audio/pcm",
        )

        audio_data = _load_audio()
        output_emitter.push(audio_data)
        output_emitter.flush()

        logger.debug("dummy TTS played tawaran.mp3 for: %s", self._input_text)
