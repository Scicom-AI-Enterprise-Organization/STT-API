"""Dummy TTS plugin — emits a short silence. Does nothing."""

from __future__ import annotations

import numpy as np

from livekit.agents import tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

SAMPLE_RATE = 24000
NUM_CHANNELS = 1


class _DummyStream(tts.ChunkedStream):
    def __init__(self, *, tts_instance: TTS, text: str) -> None:
        super().__init__(
            tts=tts_instance, input_text=text, conn_options=DEFAULT_API_CONNECT_OPTIONS
        )

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.start_segment()
        silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.int16)
        output_emitter.push(silence.tobytes())
        output_emitter.end_segment()


class TTS(tts.TTS):
    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
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
    ) -> _DummyStream:
        return _DummyStream(tts_instance=self, text=text)
