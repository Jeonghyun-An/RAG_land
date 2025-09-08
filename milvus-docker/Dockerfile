# milvus-docker/Dockerfile
# CUDA 런타임(12.1, cuDNN 포함)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# 1) 공통 런타임(빠짐없이 설치)
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl \
      libglib2.0-0 libgl1 libsm6 libxext6; \
    rm -rf /var/lib/apt/lists/*

# # 2) (옵션) Tesseract + 한국어
# RUN --mount=type=cache,target=/var/cache/apt \
#     --mount=type=cache,target=/var/lib/apt/lists \
#     set -eux; \
#     if [ "${INSTALL_TESSERACT}" = "1" ]; then \
#       apt-get update; \
#       apt-get install -y --no-install-recommends \
#         tesseract-ocr tesseract-ocr-kor; \
#       rm -rf /var/lib/apt/lists/*; \
#     fi

# 작업 디렉토리
WORKDIR /app

# requirements.txt 복사 (context 기준)
COPY milvus-docker/requirements.txt requirements.txt

# 일부 패키지 사전 설치로 충돌 완화(선택)
RUN pip install --no-cache-dir "numpy>=1.24,<2.0"

# pip 캐시 사용(빌드킷 필수) → 재빌드 속도↑
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY app/ ./app

# FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
