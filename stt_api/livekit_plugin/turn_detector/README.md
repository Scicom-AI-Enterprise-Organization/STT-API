# Turn detector plugin for LiveKit Agents

This plugin introduces end-of-turn detection for LiveKit Agents using a custom open-weight model to determine when a user has finished speaking.

Traditional voice agents use VAD (voice activity detection) for end-of-turn detection. However, VAD models lack language understanding, often causing false positives where the agent interrupts the user before they finish speaking.

By leveraging a language model specifically trained for this task, this plugin offers a more accurate and robust method for detecting end-of-turns.

See [https://docs.livekit.io/agents/build/turns/turn-detector/](https://docs.livekit.io/agents/build/turns/turn-detector/) for more information.

## Installation

```bash
pip install livekit-plugins-turn-detector
```

## Usage

### Multilingual model

We've trained a multilingual model that supports the following languages: `English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Indonesian, Russian, Turkish, Hindi`

By default, the model uses a remote vLLM inference server for predictions. The remote endpoint can be configured via the `LIVEKIT_REMOTE_EOT_URL` environment variable. When no remote URL is configured, the model falls back to local inference.

```python
from livekit.plugins.turn_detector.multilingual import MultilingualModel

session = AgentSession(
    ...
    turn_detection=MultilingualModel(),
)
```

### Usage with RealtimeModel

The turn detector can be used even with speech-to-speech models such as OpenAI's Realtime API. You'll need to provide a separate STT to ensure our model has access to the text content.

```python
session = AgentSession(
    ...
    stt=deepgram.STT(model="nova-3", language="multi"),
    llm=openai.realtime.RealtimeModel(),
    turn_detection=MultilingualModel(),
)
```

## Configuration

### Remote inference (default)

The plugin sends requests to a vLLM-compatible `/v1/completions` endpoint. Set the base URL via the environment variable:

```bash
export LIVEKIT_REMOTE_EOT_URL="https://your-vllm-endpoint.example.com"
```

The served model is `livekit/turn-detector`. Remote inference has a 2-second timeout per request.

### Local inference (fallback)

If `LIVEKIT_REMOTE_EOT_URL` is unset or empty, the plugin falls back to local inference using the HuggingFace model `livekit/turn-detector` (revision `v0.4.1-intl`). In this mode, model files are required — download them before first use:

```bash
python my_agent.py download-files
```

Model files are downloaded to and loaded from the location specified by the `HF_HUB_CACHE` environment variable. If not set, this defaults to `$HF_HOME/hub` (typically `~/.cache/huggingface/hub`).

## License

The plugin source code is licensed under the Apache-2.0 license.

The end-of-turn model is licensed under the [LiveKit Model License](https://huggingface.co/livekit/turn-detector/blob/main/LICENSE).