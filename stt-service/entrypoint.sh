#!/bin/bash
# stt-service/entrypoint.sh

set -e

echo "========================================"
echo "🎤 Starting STT Service"
echo "========================================"
echo "Model: ${STT_MODEL_SIZE:-medium}"
echo "Device: ${STT_DEVICE:-cuda}"
echo "Compute: ${STT_COMPUTE_TYPE:-float16}"
echo "========================================"

# CUDA 확인
if [ "${STT_DEVICE}" = "cuda" ]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA Device Info:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "nvidia-smi not found, but CUDA device requested"
    fi
fi

# 모델 디렉토리 생성
mkdir -p ${HF_HOME:-/models}

# Uvicorn 시작
exec python3 -m uvicorn app:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --log-level info \
    --no-access-log