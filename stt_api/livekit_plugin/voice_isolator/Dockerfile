# The GPU backend of the voice isolator plugin (RemoteIsolator): the release,
# Scicom-intl/voice-isolator (M + a causal WavLM-Large branch, 555M
# parameters), served by the model repository's own WebSocket server. Measured
# through LiveKit agents (benchmark/): one GPU carried 64 live calls in LiveKit's
# 50 ms frames, the agents hearing them 100 ms late at the median, 110 ms at p99,
# nothing passed unfiltered, a step of all 64 taking 10 ms of every 40 ms.
#
#   docker build -t voice-isolator-gpu -f stt_api/livekit_plugin/voice_isolator/Dockerfile .
#   docker run --gpus all -p 8765:8765 voice-isolator-gpu
#
# Agents then use RemoteIsolator("ws://<host>:8765") or VOICE_ISOLATOR_URL.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir numpy soundfile safetensors huggingface_hub websockets uvloop

RUN python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('Scicom-intl/voice-isolator', local_dir='/model')"

WORKDIR /model
EXPOSE 8765

# --workers: processes carrying the WebSockets (one Python process tops out near
# 6,400 messages a second); --enroll-loud: the live enrollment rule, the same one
# LocalIsolator runs; --chunk-hops 4: 40 ms steps, for LiveKit's 50 ms frames (the
# WavLM branch needs 20 ms multiples). With agents set to 20 ms frames
# (AudioInputOptions(frame_size_ms=20)), --chunk-hops 2 halves the delay (50 ms
# median at 32 calls) for fewer calls per GPU.
CMD ["python", "-m", "vi.server_mp", "--ckpt", "/model", "--port", "8765", \
     "--workers", "8", "--enroll-loud", "--chunk-hops", "4", "--max-streams", "128"]
