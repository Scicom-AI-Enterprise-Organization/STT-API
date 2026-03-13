# VAD plugin for LiveKit Agents

1. Export VAD_WORKERS to control the number of parallel ONNX inference workers (default: 4)

```
import os
os.environ["VAD_WORKERS"] = "4"
```

2. Import VAD from the vad plugin

```python
from stt_api.livekit_plugin.vad import VAD
```

3. Load the VAD model in your prewarm function and add it to AgentSession

```python
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = VAD.load()


session = AgentSession(
        stt=groq.STT(model="whisper-large-v3-turbo", language="en"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="ash"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
```
