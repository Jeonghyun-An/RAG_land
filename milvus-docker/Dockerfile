# milvus-docker/Dockerfile

FROM python:3.10-slim

# 시스템 패키지 (tesseract + 한국어, cv 런타임 용)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-kor \
    libglib2.0-0 libgl1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 (context 기준)
COPY milvus-docker/requirements.txt ./requirements.txt

# pip 최신화 + numpy 먼저(의존 충돌 방지)
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir "numpy>=1.23,<2.0"

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 모댈 관련 의존성 설치
# COPY milvus-docker/requirements-model.txt ./requirements-model.txt
# RUN pip install --no-cache-dir -r requirements-model.txt

# main.py가 있는 app 폴더의 내용만 복사 (디렉토리 X)
COPY app/ ./app


# FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
