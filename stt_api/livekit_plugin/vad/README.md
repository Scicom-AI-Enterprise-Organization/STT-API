# VAD plugin for LiveKit Agents

LiveKit VAD plugin powered by [FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD) (Deep FSMN architecture). Single-process, ~0.06ms per frame at 10ms frame shift.

## Usage

1. Import VAD from the plugin

```python
from stt_api.livekit_plugin.vad import VAD
```

2. Load the model in your prewarm function and add it to AgentSession

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

## Profiling

To run profiling on load (200 sample inferences + live tracking per stream):

```python
proc.userdata["vad"] = VAD.load(profiling=True)
```

## Configuration

```python
VAD.load(
    min_speech_duration=0.06,      # seconds to confirm speech
    min_silence_duration=0.4,      # seconds of silence to end speech
    prefix_padding_duration=0.08,  # padding before speech start
    max_buffered_speech=60.0,      # max speech buffer (seconds)
    activation_threshold=0.5,      # speech probability threshold
    sample_rate=16000,             # only 16kHz supported
    use_gpu=False,                 # GPU inference
    profiling=False,               # enable profiling
)
```
