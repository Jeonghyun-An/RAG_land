# app/config.py
import os

def _as_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y"}

# 실행 환경
IS_DOCKER = _as_bool(os.getenv("IS_DOCKER"), False)

# MinIO
MINIO_HOST = "minio:9000" if IS_DOCKER else os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = _as_bool(os.getenv("MINIO_SECURE"), False)
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-docs")

# Milvus
MILVUS_HOST = "milvus" if IS_DOCKER else os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

# HF 모델
MODEL_ID = os.getenv("MODEL_ID", "saltlux/Ko-Llama3-Luxia-8B")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")  # 없으면 빈 문자열
