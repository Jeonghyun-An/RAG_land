# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.llama_router import router as llama_router
from app.api.db_router import router as db_router
from app.api.java_router import router as java_router 

API_BASE = "/llama"

app = FastAPI(
    title="RAG API",
    docs_url=f"{API_BASE}/docs",
    openapi_url=f"{API_BASE}/openapi.json",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(llama_router, prefix=API_BASE)
app.include_router(db_router, prefix=API_BASE)
app.include_router(java_router, prefix=API_BASE) 

@app.get(f"{API_BASE}/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/")  # 선택
def root():
    return {"message": "ok", "docs": f"{API_BASE}/docs"}
