# Turn detector plugin for LiveKit Agents

1. Export livekit remote url which call vllm engine api 

```
import os                                   
os.environ["LIVEKIT_REMOTE_EOT_URL"] = "<vllm engine api>"
```

2. Import MultilingualModel from turn detector plugin

```python
from stt_api.livekit_plugin.turn_detector import MultilingualModel
```

3. Add MultilingualModel as turn detection in AgentSession

```python
session = AgentSession(
        stt=groq.STT(model="whisper-large-v3-turbo", language="en"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="ash"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
```