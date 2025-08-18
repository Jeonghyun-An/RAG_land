# app/api/llama_router.py
from __future__ import annotations

import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
from pydantic import BaseModel

from app.services.chunker import chunk_text
from app.services.file_parser import parse_pdf
from app.services.llama_model import load_model, generate_answer
from app.services.milvus_store import MilvusStore
from app.services.minio_store import MinIOStore

router = APIRouter(tags=["llama"])

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- Schemas ----------
class GenerateReq(BaseModel):
    prompt: str
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"

class AskReq(BaseModel):
    question: str
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    top_k: int = 3

class UploadResp(BaseModel):
    filename: str
    minio_object: str
    indexed: str  # "background"
    job_id: Optional[str] = None

class AskResp(BaseModel):
    answer: str
    used_chunks: int

# ---------- Helpers ----------
def index_pdf_to_milvus(file_path: str) -> None:
    """PDF → text → chunks → Milvus"""
    text = parse_pdf(file_path)
    if not text:
        raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다.")
    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError("Chunking 결과가 비었습니다.")

    MilvusStore.wait_for_milvus()
    db = MilvusStore()
    db.add_texts(chunks)

# ---------- Routes ----------
@router.get("/test")
def test():
    return {"status": "LLaMA router is working"}

@router.post("/generate")
def generate(body: GenerateReq):
    try:
        model, tokenizer = load_model(body.model_name)
        result = generate_answer(body.prompt, model, tokenizer)
        return {"response": result}
    except Exception as e:
        raise HTTPException(500, f"모델 응답 생성 실패: {e}")

@router.post("/upload", response_model=UploadResp)
async def upload_document(
    background_tasks: BackgroundTasks,           # ✅ Optional/기본값 쓰지 말 것
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF 파일만 업로드 가능합니다.")

    minio = MinIOStore()

    try:
        # 1) 로컬 저장
        safe_name = os.path.basename(file.filename)
        local_path = os.path.join(UPLOAD_DIR, safe_name)
        content = await file.read()
        with open(local_path, "wb") as f:
            f.write(content)

        # 2) MinIO 업로드 (유니크 오브젝트명)
        object_name = f"uploaded/{uuid.uuid4().hex}_{safe_name}"
        minio.upload(local_path, object_name=object_name, content_type="application/pdf")

        # 3) Milvus 인덱싱: 항상 백그라운드
        job_id = uuid.uuid4().hex
        background_tasks.add_task(index_pdf_to_milvus, local_path)

        return UploadResp(
            filename=safe_name,
            minio_object=object_name,
            indexed="background",
            job_id=job_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"업로드 처리 중 오류: {e}")

@router.post("/ask", response_model=AskResp)
def ask_question(body: AskReq):
    try:
        MilvusStore.wait_for_milvus()
        db = MilvusStore()
        retrieved: List[str] = db.search(body.question, top_k=body.top_k)

        if not retrieved:
            raise HTTPException(404, "관련 문서를 찾지 못했습니다. 먼저 문서를 업로드/인덱싱 해주세요.")

        context = "\n\n".join(retrieved)
        prompt = f"""아래 문서를 참고해서 질문에 정확히 답하세요. 근거가 없으면 모른다고 답합니다.

[문서 발췌]
{context}

[질문]
{body.question}

[답변]
"""
        model, tokenizer = load_model(body.model_name)
        answer = generate_answer(prompt, model, tokenizer)
        return AskResp(answer=answer, used_chunks=len(retrieved))

    except HTTPException:
        raise
    except RuntimeError as milvus_error:
        raise HTTPException(503, f"Milvus 연결 대기/검색 실패: {milvus_error}")
    except Exception as e:
        raise HTTPException(500, f"질의 처리 중 오류: {e}")

# ---------- MinIO Utilities ----------
@router.get("/files")
def list_files(
    prefix: str = Query("", description="prefix로 필터링 (예: 'uploaded/')"),
):
    try:
        return {"files": MinIOStore().list_files(prefix=prefix)}
    except Exception as e:
        raise HTTPException(500, f"MinIO 파일 조회 실패: {e}")

@router.get("/file/{object_name}")
def get_file_url(
    object_name: str,
    method: str = Query("GET", pattern="^(GET|PUT)$"),
    minutes: int = Query(60, ge=1, le=7*24*60, description="URL 만료 시간(분)"),
    download_name: Optional[str] = Query(
        None, description="다운로드 파일명 지정 시 Content-Disposition 헤더 세팅"
    ),
):
    try:
        from datetime import timedelta
        minio = MinIOStore()

        response_headers = None
        if download_name:
            response_headers = {
                "response-content-disposition": f'attachment; filename="{download_name}"'
            }

        url = minio.presigned_url(
            object_name=object_name,
            method=method,
            expires=timedelta(minutes=minutes),
            response_headers=response_headers,
        )
        return {"url": url}
    except Exception as e:
        raise HTTPException(500, f"파일 사전서명 URL 생성 실패: {e}")

@router.delete("/file/{object_name}")
def delete_file(object_name: str):
    try:
        minio = MinIOStore()
        if not minio.exists(object_name):
            raise HTTPException(404, "파일이 존재하지 않습니다.")
        minio.delete(object_name)
        return {"status": "ok", "deleted": object_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"파일 삭제 실패: {e}")
