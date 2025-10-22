# app/api/java_router.py
"""
Java 시스템 연동 라우터 (운영용)
- 서버 파일시스템 사용
- DB 완전 연동
- 실제 운영 환경 전용
"""
from __future__ import annotations

import os
import hashlib
import hmac
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
import httpx

from app.services.db_connector import DBConnector
from app.services.pdf_converter import convert_to_pdf, ConvertError
from app.services import job_state

router = APIRouter(prefix="/java", tags=["java-production"])

# 환경변수
SHARED_SECRET = os.getenv("JAVA_SHARED_SECRET", "")
SERVER_BASE_PATH = os.getenv("SERVER_BASE_PATH", "/mnt/shared")


# ==================== Schemas ====================
class ConvertAndIndexRequest(BaseModel):
    """자바 → AI 트리거 요청"""
    data_id: str
    path: str  # 서버 파일시스템 상대 경로
    file_id: str
    webhook_url: Optional[str] = None
    ocr_manual_required: bool = False
    reindex_required_yn: bool = False


class ConvertAndIndexResponse(BaseModel):
    """즉시 응답"""
    status: str
    job_id: str
    data_id: str
    message: str


class WebhookPayload(BaseModel):
    """AI → 자바 콜백 페이로드"""
    job_id: str
    data_id: str
    status: str
    converted: bool = False
    metrics: Optional[Dict[str, Any]] = None
    timestamps: Optional[Dict[str, str]] = None
    message: str = ""
    chunk_count: Optional[int] = None


class StatusResponse(BaseModel):
    """상태 조회 응답"""
    data_id: str
    rag_index_status: str
    parse_yn: Optional[str] = None
    chunk_count: Optional[int] = None
    parse_start_dt: Optional[str] = None
    parse_end_dt: Optional[str] = None
    milvus_doc_id: Optional[str] = None


# ==================== Helper Functions ====================
def verify_internal_token(token: Optional[str]) -> bool:
    if not token:
        return False
    return token == SHARED_SECRET


def generate_hmac_signature(payload: str, secret: str) -> str:
    return hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


async def send_webhook(url: str, payload: WebhookPayload, secret: str):
    try:
        payload_json = payload.model_dump_json()
        signature = generate_hmac_signature(payload_json, secret)
        
        headers = {
            'Content-Type': 'application/json',
            'X-Webhook-Signature': signature
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, content=payload_json, headers=headers)
            response.raise_for_status()
            print(f"[PROD-WEBHOOK] ✅ Sent to {url}")
    except Exception as e:
        print(f"[PROD-WEBHOOK] ❌ Failed: {e}")


# ==================== Background Task ====================
# app/api/java_router.py

async def process_convert_and_index_prod(
    job_id: str,
    data_id: str,
    path: str,
    file_id: str,
    webhook_url: Optional[str],
    ocr_manual_required: bool,
    reindex_required_yn: bool
):
    """운영용 백그라운드 처리 - 단순 청킹 적용"""
    db = DBConnector()
    start_time = datetime.utcnow()
    
    job_state.start(job_id, data_id=data_id, file_id=file_id)
    db.mark_ocr_start(data_id)
    
    try:
        # ========== Step 1: 서버 파일 로드 ==========
        job_state.update(job_id, status="uploaded", step="Loading file from server")
        
        full_path = os.path.join(SERVER_BASE_PATH, path, file_id)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"서버 파일 없음: {full_path}")
        
        print(f"[PROD] Using server file: {full_path}")
        
        # ========== Step 2: PDF 변환 (필요시) ==========
        job_state.update(job_id, status="parsing", step="Converting to PDF if needed")
        
        is_already_pdf = file_id.lower().endswith('.pdf')
        converted_pdf_path = full_path
        
        if not is_already_pdf:
            # ... PDF 변환 로직 (기존 코드 유지) ...
            pass
        
        # ========== Step 3: 단순 청킹 & 인덱싱 (NEW) ==========
        job_state.update(job_id, status="processing", step="Simple chunking for proofreading")
        
        # 3-1) PDF 텍스트 추출
        from app.services.file_parser import parse_pdf
        
        print(f"[PROD-CHUNK] Extracting text from: {converted_pdf_path}")
        pages_std = parse_pdf(converted_pdf_path, by_page=True)
        
        if not pages_std:
            raise RuntimeError("텍스트 추출 실패")
        
        print(f"[PROD-CHUNK] Extracted {len(pages_std)} pages")
        
        # 3-2) 단순 청킹
        from app.services.simple_proofreading_chunker import simple_chunk_by_paragraph
        from app.services.embedding_model import get_embedding_model
        
        embed_model = get_embedding_model()
        encoder_fn = embed_model.tokenizer.encode
        
        print(f"[PROD-CHUNK] Chunking with simple proofreading chunker (paragraph-based)")
        
        chunks = simple_chunk_by_paragraph(
            pages_std,
            encoder_fn,
            target_tokens=400  # 조정 가능
        )
        
        if not chunks:
            raise RuntimeError("청킹 실패: 청크가 생성되지 않음")
        
        print(f"[PROD-CHUNK] Created {len(chunks)} chunks")
        
        # 3-3) 임베딩
        job_state.update(job_id, status="embedding", step=f"Embedding {len(chunks)} chunks")
        
        from app.services.embedding_model import embed
        
        chunk_texts = []
        chunk_metas = []
        
        for chunk_text, chunk_meta in chunks:
            # META 라인 제거하고 본문만
            clean_text = chunk_text
            if clean_text.startswith("META:"):
                nl_pos = clean_text.find("\n")
                clean_text = clean_text[nl_pos + 1:] if nl_pos != -1 else ""
            
            chunk_texts.append(clean_text.strip())
            chunk_metas.append(chunk_meta)
        
        print(f"[PROD-CHUNK] Embedding {len(chunk_texts)} chunks...")
        embeddings = embed(chunk_texts)
        
        # 3-4) Milvus 저장
        job_state.update(job_id, status="indexing", step="Indexing to Milvus")
        
        from app.services.milvus_store_v2 import MilvusStoreV2
        
        mvs = MilvusStoreV2()
        collection_name = os.getenv("MILVUS_COLLECTION_NAME", "nuclear_rag")
        
        # 기존 문서 삭제 (재인덱싱 시)
        if reindex_required_yn:
            print(f"[PROD-CHUNK] Deleting existing doc: {data_id}")
            mvs.delete_by_doc_id(collection_name, data_id)
        
        # 청크 삽입
        print(f"[PROD-CHUNK] Inserting {len(chunks)} chunks to Milvus")
        
        for i, (emb, text, meta) in enumerate(zip(embeddings, chunk_texts, chunk_metas)):
            mvs.insert_one(
                collection_name,
                doc_id=data_id,
                chunk_id=f"{data_id}_chunk_{i}",
                chunk_index=i,
                text=text,
                embedding=emb.tolist() if hasattr(emb, 'tolist') else emb,
                page=meta.get('page', 1),
                pages=meta.get('pages', [meta.get('page', 1)]),
                metadata={
                    "type": meta.get('type', 'proofreading_chunk'),
                    "token_count": meta.get('token_count', 0),
                    "char_count": meta.get('char_count', 0),
                    "file_id": file_id,
                    "data_id": data_id
                }
            )
        
        print(f"[PROD-CHUNK] ✅ Successfully indexed {len(chunks)} chunks")
        
        # ========== Step 4: 결과 조회 및 DB 업데이트 ==========
        pages = len(pages_std)
        chunk_count = len(chunks)
        
        print(f"[PROD] ✅ Indexing completed: {pages} pages, {chunk_count} chunks")
        
        # DB 업데이트: RAG 인덱싱 완료
        db.update_rag_completed(
            data_id,
            chunks=chunk_count,
            doc_id=data_id
        )
        
        # Job 상태 업데이트
        job_state.complete(
            job_id,
            pages=pages,
            chunks=chunk_count
        )
        
        # ========== Step 5: 완료 & Webhook ==========
        end_time = datetime.utcnow()
        
        if webhook_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="done",
                converted=not is_already_pdf,
                metrics={"pages": pages, "chunks": chunk_count},
                chunk_count=chunk_count,
                timestamps={
                    "start": start_time.isoformat(), 
                    "end": end_time.isoformat()
                },
                message="converted and indexed with simple proofreading chunker" if not is_already_pdf else "indexed with simple proofreading chunker"
            )
            await send_webhook(webhook_url, payload, SHARED_SECRET)
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        db.update_parse_status(data_id, rag_status="error")
        print(f"[PROD] ❌ Error: {e}")
        
        if webhook_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="error",
                message=str(e)
            )
            await send_webhook(webhook_url, payload, SHARED_SECRET)
        
        raise

# ==================== Routes ====================
@router.post("/convert-and-index", response_model=ConvertAndIndexResponse)
async def convert_and_index(
    request: ConvertAndIndexRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: Optional[str] = Header(None)
):
    """
    자바 → AI 트리거 API (운영용)
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    # 중복 체크
    db = DBConnector()
    existing = db.get_file_by_id(request.data_id)
    
    if existing and existing.get('rag_index_status') == 'done':
        if not request.reindex_required_yn:
            return ConvertAndIndexResponse(
                status="already_done",
                job_id="",
                data_id=request.data_id,
                message="Already indexed"
            )
    
    job_id = str(uuid.uuid4())[:8]
    
    background_tasks.add_task(
        process_convert_and_index_prod,
        job_id=job_id,
        data_id=request.data_id,
        path=request.path,
        file_id=request.file_id,
        webhook_url=request.webhook_url,
        ocr_manual_required=request.ocr_manual_required,
        reindex_required_yn=request.reindex_required_yn
    )
    
    return ConvertAndIndexResponse(
        status="accepted",
        job_id=job_id,
        data_id=request.data_id,
        message="processing"
    )


@router.get("/status/{data_id}", response_model=StatusResponse)
def get_status(data_id: str, x_internal_token: Optional[str] = Header(None)):
    """상태 조회 API (운영용)"""
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    db = DBConnector()
    meta = db.get_file_by_id(data_id)
    
    if not meta:
        raise HTTPException(404, f"data_id {data_id} not found")
    
    return StatusResponse(
        data_id=data_id,
        rag_index_status=meta.get('rag_index_status', 'unknown'),
        parse_yn=meta.get('parse_yn'),
        chunk_count=meta.get('chunk_count'),
        parse_start_dt=str(meta.get('parse_start_dt')) if meta.get('parse_start_dt') else None,
        parse_end_dt=str(meta.get('parse_end_dt')) if meta.get('parse_end_dt') else None,
        milvus_doc_id=meta.get('milvus_doc_id')
    )


@router.get("/health")
def health_check():
    """헬스 체크"""
    return {"status": "ok", "service": "java-router-production"}