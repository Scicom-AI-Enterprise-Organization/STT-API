# The GPU backend of the voice isolator plugin (RemoteIsolator): the release,
# Scicom-intl/voice-isolator-m-wavlm (M + a causal WavLM-Large branch, 555M
# parameters), served by the model repository's own WebSocket server. One GPU
# carries 64 live calls with 20 ms chunks at 29 ms median / 39 ms p99 round trip
# on the same host, once every call has enrolled; more calls fall behind.
#
# The repository is private: build with a token that can read it, as a secret,
# so it is never written into a layer.
#
#   docker build --secret id=hf_token,env=HF_TOKEN -t voice-isolator-gpu \
#       -f stt_api/livekit_plugin/voice_isolator/Dockerfile .
#   docker run --gpus all -p 8765:8765 voice-isolator-gpu
#
# Agents then use RemoteIsolator("ws://<host>:8765") or VOICE_ISOLATOR_URL.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir numpy soundfile safetensors huggingface_hub websockets uvloop

RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('Scicom-intl/voice-isolator-m-wavlm', local_dir='/model')"

WORKDIR /model
EXPOSE 8765

# --workers: processes carrying the WebSockets (one Python process tops out near
# 6,400 messages a second); --enroll-loud: the live enrollment rule, the same one
# LocalIsolator runs; --chunk-hops 2: 20 ms steps (the WavLM branch needs 20 ms
# multiples).
CMD ["python", "-m", "vi.server_mp", "--ckpt", "/model", "--port", "8765", \
     "--workers", "8", "--enroll-loud", "--chunk-hops", "2", "--max-streams", "128"]
