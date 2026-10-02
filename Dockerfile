FROM python:3.11-slim

RUN apt update && \
    apt install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

# The server stack is an extra: base dependencies are empty so that an agent
# installing stt-api[scicom-livekit-plugin] never pulls torch and friends.
RUN uv sync --extra server --frozen -v || uv sync --extra server -v

COPY . .
