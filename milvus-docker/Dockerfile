# CUDA 런타임(12.1, cuDNN 포함) + PyTorch 사전탑재
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=60

# ========== [추가] HWP 변환 지원 토글 ==========
ARG INSTALL_OCR_STACK=1
ARG INSTALL_HWP_SUPPORT=1

# ---- 시스템 패키지 설치 ----
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget \
      libglib2.0-0 libgl1 libsm6 libxext6 \
      poppler-utils \
      fonts-nanum fonts-nanum-coding fonts-nanum-extra; \
    if [ "$INSTALL_OCR_STACK" = "1" ]; then \
      apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-kor \
        ghostscript qpdf; \
    fi; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 복사하여 레이어 캐시 최대로 활용
COPY milvus-docker/requirements.txt requirements.txt

# ---- pip 캐시(빌드킷) 적극 사용 + 휠 우선 ----
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip setuptools wheel && \
    python -m pip install --prefer-binary "numpy>=1.24,<2.0" && \
    python -m pip install --prefer-binary -r requirements.txt

# ========== [추가] pyhwp 설치 (HWP 변환용) ==========
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_HWP_SUPPORT" = "1" ]; then \
      python -m pip install --prefer-binary --pre pyhwp; \
    fi

# ---- CUBRID 파이썬 드라이버 (cp310) : URL 설치 + pip 캐시 활용 ----
ARG CUBRID_WHEEL_URL="https://ftp.cubrid.org/CUBRID_Drivers/Python_Driver/11.3.0/Linux/cubrid_python-11.3.0.51-cp310-cp310-linux_x86_64.whl"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --prefer-binary "${CUBRID_WHEEL_URL}"

# 앱 코드는 마지막에 복사 → 코드 변경만 있을 때 의존성 레이어 재사용
COPY app/ ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8000"]