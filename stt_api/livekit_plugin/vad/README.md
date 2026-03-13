# VAD plugin for LiveKit Agents

Drop-in replacement for `livekit.plugins.silero.VAD` with concurrency-limited inference to prevent CPU oversubscription.

## What it does

- Caps concurrent ONNX inferences via `ThreadPoolExecutor` + `asyncio.Semaphore`
- Sets `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` to limit internal ONNX threading
- Configurable via environment variables

## Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `VAD_WORKERS` | `4` | Max concurrent VAD inferences |
| `OMP_NUM_THREADS` | `1` | OpenMP threads per inference |
| `OPENBLAS_NUM_THREADS` | `1` | OpenBLAS threads per inference |
| `MKL_NUM_THREADS` | `1` | Intel MKL threads per inference |

For a 16-CPU machine, recommended: `VAD_WORKERS=8` (leaves 8 cores for STT/TTS/event loop).

## Usage

1. Import VAD from the custom plugin instead of silero

```python
from stt_api.livekit_plugin.vad import VAD
```

2. Load and use as drop-in replacement for silero VAD

```python
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = VAD.load()
```

3. Add VAD to AgentSession

```python
session = AgentSession(
    stt=groq.STT(model="whisper-large-v3-turbo", language="en"),
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=openai.TTS(model="gpt-4o-mini-tts", voice="ash"),
    turn_detection=MultilingualModel(),
    vad=ctx.proc.userdata["vad"],
)
```
