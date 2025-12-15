FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ffmpeg \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# deps 먼저 설치 (캐시 최대 활용)
COPY requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

# app
WORKDIR /app/stt
COPY app.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
