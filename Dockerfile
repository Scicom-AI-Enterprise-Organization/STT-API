FROM python:3.11-slim

RUN apt update && \
    apt install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

# PyTorch CPU will be installed automatically via uv
RUN uv sync --frozen -v || uv sync -v

COPY . .
