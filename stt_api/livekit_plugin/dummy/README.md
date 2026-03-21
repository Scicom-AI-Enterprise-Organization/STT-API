# Dummy plugins for LiveKit Agents

No-op STT, LLM, and TTS plugins that avoid external API calls, making them safe for load testing (e.g. `lk perf agent-load-test`) without incurring costs from Groq, OpenAI, or other paid services.

- **STT** returns canned transcriptions (cycles through 5 predefined sentences)
- **LLM** returns canned responses (cycles through 5 predefined replies)
- **TTS** plays back a pre-recorded audio file (`audio/tawaran.mp3`) so that Silero VAD can detect real speech and trigger the full pipeline

1. Import dummy plugins

```python
from stt_api.livekit_plugin.dummy import STT, LLM, TTS
```

2. Replace paid STT/LLM/TTS with dummy plugins in AgentSession

```python
session = AgentSession(
        stt=STT(),
        llm=LLM(),
        tts=TTS(),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
```

3. Run load test without worrying about API costs

```bash
lk perf agent-load-test \
  --url wss://livekit-server.example.com \
  --api-key <key> \
  --api-secret <secret> \
  --agent-name adha-agent \
  --rooms 50 \
  --duration 3m
```