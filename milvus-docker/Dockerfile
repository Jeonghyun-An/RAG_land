# milvus-docker/Dockerfile

FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 (context 기준)
COPY milvus-docker/requirements.txt ./requirements.txt

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 모댈 관련 의존성 설치
# COPY milvus-docker/requirements-model.txt ./requirements-model.txt
# RUN pip install --no-cache-dir -r requirements-model.txt

# main.py가 있는 app 폴더의 내용만 복사 (디렉토리 X)
COPY app/ ./app


# FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
