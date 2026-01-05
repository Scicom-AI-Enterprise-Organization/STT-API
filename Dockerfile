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

RUN uv sync --frozen -v || uv sync -v

COPY . .
