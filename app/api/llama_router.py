# app/api/llama_router.py
from __future__ import annotations

import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
from pydantic import BaseModel

from app.services.chunker import chunk_text
from app.services.file_parser import parse_pdf
from app.services.llama_model import generate_answer_unified
from app.services.milvus_store import MilvusStore
from app.services.minio_store import MinIOStore
from app.services.pdf_converter import convert_to_pdf, ConvertError
from app.services.chunker import smart_chunk_pages
from app.services.milvus_store_v2 import MilvusStoreV2
from app.services.embedding_model import get_embedding_model, embed
from app.services.reranker import rerank



router = APIRouter(tags=["llama"])

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- Schemas ----------
class GenerateReq(BaseModel):
    prompt: str
    model_name: str = "llama-1b"

class AskReq(BaseModel):
    question: str
    model_name: str = "llama-1b"
    top_k: int = 3

class UploadResp(BaseModel):
    filename: str
    minio_object: str
    indexed: str  # "background"
    job_id: Optional[str] = None

class AskResp(BaseModel):
    answer: str
    used_chunks: int
    # (선택) 출처 제공
    sources: Optional[List[dict]] = None


# ---------- Helpers ----------
def index_pdf_to_milvus(file_path: str) -> None:
    print(f"[INDEX] start: {file_path}")
    # 1) 페이지별 텍스트
    pages = parse_pdf(file_path, by_page=True)
    if not pages:
        raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다.")

    # 2) 임베딩 토크나이저로 토큰 기준 청킹
    model = get_embedding_model()
    tokenizer = getattr(model, "tokenizer", None)
    encode = tokenizer.encode if tokenizer and hasattr(tokenizer, "encode") else (lambda s: s.split())

    chunks = smart_chunk_pages(pages, encode)
    if not chunks:
        raise RuntimeError("Chunking 결과가 비었습니다.")

    # 3) v2 컬렉션에 메타 포함 저장
    store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())
    doc_id = os.path.basename(file_path)
    store.insert(doc_id, chunks, embed_fn=embed)
    print(f"[INDEX] done: {file_path} (chunks={len(chunks)})")

# ---------- Routes ----------
@router.get("/test")
def test():
    return {"status": "LLaMA router is working"}

@router.post("/generate")
def generate(body: GenerateReq):
    try:
        result = generate_answer_unified(body.prompt, body.model_name)
        return {"response": result}
    except Exception as e:
        raise HTTPException(500, f"모델 응답 생성 실패: {e}")


@router.post("/upload", response_model=UploadResp)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # 1) 로컬 저장
    safe_name = os.path.basename(file.filename)
    local_path = os.path.join(UPLOAD_DIR, safe_name)
    content = await file.read()
    with open(local_path, "wb") as f:
        f.write(content)

    # 2) 비-PDF면 PDF로 변환
    try:
        pdf_path = convert_to_pdf(local_path)   # 이미 PDF면 그대로 반환
    except ConvertError as e:
        raise HTTPException(400, f"파일 변환 실패: {e}")
    except Exception as e:
        raise HTTPException(500, f"파일 변환 중 예외: {e}")

    # 3) MinIO 업로드 (원본 + 변환본 둘 다 올리고 싶으면 선택)
    minio = MinIOStore()
    object_pdf = f"uploaded/{uuid.uuid4().hex}_{os.path.basename(pdf_path)}"
    minio.upload(pdf_path, object_name=object_pdf, content_type="application/pdf")

    # 4) 인덱싱(백그라운드)
    job_id = uuid.uuid4().hex
    background_tasks.add_task(index_pdf_to_milvus, pdf_path)

    return UploadResp(
        filename=safe_name,
        minio_object=object_pdf,
        indexed="background",
        job_id=job_id,
    )

@router.post("/ask", response_model=AskResp)
def ask_question(body: AskReq):
    try:
        # 1) 초기 넉넉히 검색 (예: 20)
        model = get_embedding_model()
        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())
        cands = store.search(body.question, embed_fn=embed, topk=max(20, body.top_k*5))
        if not cands:
            raise HTTPException(404, "관련 문서를 찾지 못했습니다. 먼저 문서를 업로드/인덱싱 해주세요.")

        # 2) 리랭크 후 상위 k
        topk = rerank(body.question, cands, top_k=body.top_k)

        # 3) 임계값 컷오프(리랭커 스코어 기준, 필요시 조정)
        THRESH = float(os.getenv("RAG_SCORE_THRESHOLD", "0.2"))
        if "re_score" in topk[0] and topk[0]["re_score"] < THRESH:
            return AskResp(answer="문서에서 답을 확실히 찾기 어렵습니다.", used_chunks=0, sources=[])

        # 4) 컨텍스트 + 출처 만들기
        context = ""
        sources = []
        for i, c in enumerate(topk, 1):
            context += f"[{i}] (doc:{c['doc_id']} p.{c['page']} {c['section']})\n{c['chunk']}\n\n"
            sources.append({"id": i, "doc_id": c["doc_id"], "page": c["page"], "section": c["section"]})

        prompt = f"""아래 문서를 참고해서 질문에 정확히 답하세요. 근거가 없으면 모른다고 답합니다.

[문서 발췌]
{context}

[질문]
{body.question}

[형식]
- 한 문장 핵심 답
- 필요하면 근거 출처 번호로 인용: [1]
"""
        answer = generate_answer_unified(prompt, body.model_name)
        return AskResp(answer=answer, used_chunks=len(topk), sources=sources)

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
