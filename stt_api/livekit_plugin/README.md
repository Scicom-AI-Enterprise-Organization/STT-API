# LiveKit plugins

Plugins for [LiveKit Agents](https://docs.livekit.io/agents/), installable on
their own — the extra pulls the LiveKit runtime and nothing else from this repo's
stack. No torch, no fastapi, no server-side diarization models.

| Plugin | What it does | Docs |
|---|---|---|
| `WhisperSTT` `DropBlankSTT` | Whisper STT whose blank transcripts cannot interrupt the agent | [whisper_stt](whisper_stt/README.md) |
| `PulseVAD` | Frame-level voice activity, 2,118 parameters, 0.097 ms a window | [pulse_vad](pulse_vad/README.md) |
| `SemanticVAD` | End of turn from audio, without waiting for the transcript | [semantic_vad](semantic_vad/README.md) |
| `MultilingualModel` | End of turn from text, against a vLLM endpoint or locally | [turn_detector](turn_detector/README.md) |
| `GTCRN` | Self-hosted noise cancellation, in-process, no licence key | [noise_cancellation](noise_cancellation/README.md) |
| `DummySTT` `DummyLLM` `DummyTTS` | Canned responses for load tests, zero API cost | [dummy](dummy/README.md) |

Adding a plugin to this package? It is covered by the extra the moment its
dependencies are listed there and its entry point is in `__init__.py` — no new
extra, no separate distribution.

## Install

```bash
uv pip install "stt-api[scicom-livekit-plugin] @ git+https://github.com/Scicom-AI-Enterprise-Organization/STT-API.git"
```

Pin a commit or tag when it matters — a bare URL tracks `main`:

```bash
uv pip install "stt-api[scicom-livekit-plugin] @ git+https://github.com/Scicom-AI-Enterprise-Organization/STT-API.git@<sha-or-tag>"
```

As a dependency of an agent project, in its `pyproject.toml`:

```toml
[project]
dependencies = [
    "stt-api[scicom-livekit-plugin] @ git+https://github.com/Scicom-AI-Enterprise-Organization/STT-API.git",
]
```

The repo is private, so the machine doing the install needs a credential for it:
`gh auth setup-git` on a laptop, or a token in the URL for CI and Docker builds
(`git+https://x-access-token:${GITHUB_TOKEN}@github.com/...`). Keep the token out
of the image — pass it as a build secret.

## Use

```python
from stt_api.livekit_plugin import GTCRN, PulseVAD, SemanticVAD, WhisperSTT
```

Names resolve on first use, so importing one plugin does not load the others'
dependencies. The per-plugin READMEs above carry the real usage — a sketch of
several together:

```python
from livekit.agents import AgentSession
from livekit.agents.voice import room_io

from stt_api.livekit_plugin import GTCRN, PulseVAD, ScicomEoT, SemanticVAD, WhisperSTT

def prewarm(proc):
    proc.userdata["vad"] = PulseVAD.load()

session = AgentSession(
    stt=WhisperSTT(),                             # reads STT_URL / STT_API / STT_MODEL
    llm=..., tts=...,
    vad=ctx.proc.userdata["vad"],
    turn_detection=SemanticVAD(backend=ScicomEoT()),
)

await session.start(
    agent=MyAgent(),
    room=ctx.room,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            sample_rate=16000,                   # the model's rate: skips resampling
            noise_cancellation=lambda params: GTCRN(),
        ),
    ),
)
```

## What the extra installs

`livekit-agents[turn-detector]`, plus `numpy`, `onnxruntime`, `aiohttp`,
`transformers`, `huggingface_hub` and `soundfile` — enough for every plugin in
the table. `stt-api[server]` — torch, torchaudio, fastapi, librosa, silero-vad
and the rest — is a separate extra and is never pulled in by this one, and
neither is `[benchmark]`, which is what the shootout and figure tooling inside
these packages needs. `tests/test_packaging.py` holds that line.

The plugin modules are still shipped inside the `stt_api` package, alongside the
server's modules. They are inert files: nothing under `stt_api.livekit_plugin`
imports the server side, so the unused modules cost disk and nothing else.

`livekit` is kept as an alias for `scicom-livekit-plugin`; both install the same thing.
