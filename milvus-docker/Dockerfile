FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=60

ARG INSTALL_OCR_STACK=1
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl \
      libglib2.0-0 libgl1 libsm6 libxext6 \
      poppler-utils; \
    if [ "$INSTALL_OCR_STACK" = "1" ]; then \
      apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-kor \
        ghostscript qpdf; \
    fi; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY milvus-docker/requirements.txt requirements.txt

# pip 캐시 사용 + 바이너리 우선
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip setuptools wheel && \
    python -m pip install --prefer-binary "numpy>=1.24,<2.0" && \
    python -m pip install --prefer-binary -r requirements.txt

# CUBRID 휠: 외부 URL 대신 로컬 벤더링 권장(속도/안정성↑)
# 외부 URL을 계속 쓰려면, 아래 두 줄로 교체하세요.
ARG CUBRID_WHEEL_URL="https://ftp.cubrid.org/CUBRID_Drivers/Python_Driver/11.3.0/Linux/cubrid_python-11.3.0.51-cp310-cp310-linux_x86_64.whl"
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --prefer-binary "${CUBRID_WHEEL_URL}"

# COPY third_party/cubrid/cubrid_python-11.3.0.51-cp310-cp310-linux_x86_64.whl /tmp/cubrid.whl
# RUN --mount=type=cache,target=/root/.cache/pip \
#     python -m pip install /tmp/cubrid.whl && rm -f /tmp/cubrid.whl

COPY app/ ./app

CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port","8000"]
