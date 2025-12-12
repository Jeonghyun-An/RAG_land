# app/main.py
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    try:
        Image.ANTIALIAS = Image.LANCZOS
    except Exception:
        from PIL import Image as _I
        Image.ANTIALIAS = getattr(_I, "BICUBIC", None)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.llama_router import router as llama_router
from app.api.db_router import router as db_router
from app.api.java_router import router as java_router  # 운영용
from app.api.dev_router import router as dev_router    # 개발용
from app.api.stt_router import router as stt_router  # STT 라우터
from app.services.embedding_model import get_embedding_model
get_embedding_model()  # 앱 시작 시 미리 로드

API_BASE = "/llama"

app = FastAPI(
    title="Nuclear RAG API",
    description="원자력 문서 RAG 시스템 - 운영/개발 모드 분리",
    version="2.0.0",
    docs_url=f"{API_BASE}/docs",
    openapi_url=f"{API_BASE}/openapi.json",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(llama_router, prefix=API_BASE)
app.include_router(db_router, prefix=API_BASE)
app.include_router(java_router, prefix=API_BASE)  # /llama/java - 운영용
app.include_router(dev_router, prefix=API_BASE)   # /llama/dev - 개발용
app.include_router(stt_router, prefix=API_BASE)  # stt 라우터

@app.get(f"{API_BASE}/healthz")
def healthz():
    return {
        "status": "ok",
        "routers": {
            "llama": "LLM 및 RAG 쿼리",
            "db": "DB 조회",
            "java": "Java 연동 (운영)",
            "dev": "개발/테스트 (로컬)",
            "stt": "음성-텍스트 변환 (STT)"
        }
    }

@app.get("/")
def root():
    return {
        "message": "Nuclear RAG API",
        "docs": f"{API_BASE}/docs",
        "production_endpoint": f"{API_BASE}/java",
        "development_endpoint": f"{API_BASE}/dev",
        "stt_endpoint": f"{API_BASE}/stt"
    }