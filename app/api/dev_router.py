# app/api/dev_router.py
"""
개발/테스트용 라우터
- 로컬 디렉토리 파일 사용
- DB 업데이트 없음 (개발 모드)
- 단순 청커(simple_proofreading_chunker) 사용
- Webhook 페이로드만 전달
"""
from __future__ import annotations

import os
import hashlib
import hmac
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Header, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
import httpx

from app.services.minio_store import MinIOStore
from app.services.pdf_converter import convert_to_pdf, ConvertError
from app.services import job_state

router = APIRouter(prefix="/dev", tags=["development"])

# 환경변수
DEV_SECRET = os.getenv("DEV_SECRET", "devSecret2025")
LOCAL_STAGING_PATH = os.getenv("LOCAL_STAGING_PATH", "/tmp/remote_staging")


# ==================== Schemas ====================
class DevConvertRequest(BaseModel):
    data_id: str
    path: str = ""
    file_id: str
    callback_url: Optional[str] = None  # callback_url로 변경


class DevConvertResponse(BaseModel):
    status: str
    job_id: str
    data_id: str
    message: str


class DevWebhookPayload(BaseModel):
    job_id: str
    data_id: str
    status: str
    converted: bool = False
    metrics: Optional[Dict[str, Any]] = None
    timestamps: Optional[Dict[str, str]] = None
    message: str = ""
    pdf_key_minio: Optional[str] = None
    chunk_count: Optional[int] = None


class DevStatusResponse(BaseModel):
    data_id: str
    rag_index_status: str
    parse_yn: Optional[str] = None
    chunk_count: Optional[int] = None
    parse_start_dt: Optional[str] = None
    parse_end_dt: Optional[str] = None
    milvus_doc_id: Optional[str] = None


# ==================== Helper Functions ====================
def verify_dev_token(token: Optional[str]) -> bool:
    if not token:
        return False
    return token == DEV_SECRET


def generate_hmac_signature(payload: str, secret: str) -> str:
    return hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


async def send_dev_webhook(url: str, payload: DevWebhookPayload, secret: str):
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
            print(f"[DEV-WEBHOOK] Sent to {url}")
    except Exception as e:
        print(f"[DEV-WEBHOOK] Failed: {e}")


def generate_minio_pdf_key(data_id: str) -> str:
    return f"dev/pdfs/{data_id}.pdf"


# ==================== Background Task ====================
async def process_dev_convert_and_index(
    job_id: str,
    data_id: str,
    file_id: str,
    callback_url: Optional[str]
):
    """
    개발용 백그라운드 처리
    - 단순 청커(simple_proofreading_chunker) 사용
    - DB 업데이트 없음
    - Webhook으로 결과 전달
    """
    store = MinIOStore()
    start_time = datetime.utcnow()
    
    job_state.start(job_id, data_id=data_id, file_id=file_id)
    
    try:
        # ========== Step 1: 로컬 파일 로드 ==========
        job_state.update(job_id, status="uploaded", step="Loading local file")
        
        staging_dir = Path(LOCAL_STAGING_PATH)
        staging_dir.mkdir(parents=True, exist_ok=True)
        file_path = staging_dir / file_id
        
        if not file_path.exists():
            raise FileNotFoundError(f"로컬 파일 없음: {file_path}")
        
        print(f"[DEV] Using local file: {file_path}")
        
        # ========== Step 2: PDF 변환 (필요시) ==========
        job_state.update(job_id, status="parsing", step="Converting to PDF if needed")
        
        is_already_pdf = file_id.lower().endswith('.pdf')
        pdf_key: Optional[str] = None
        converted_pdf_path = str(file_path)
        
        if not is_already_pdf:
            try:
                # PDF 변환
                converted_pdf_path = convert_to_pdf(str(file_path))
                
                # MinIO 업로드
                with open(converted_pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                pdf_key = generate_minio_pdf_key(data_id)
                store.upload_bytes(
                    data=pdf_bytes,
                    object_name=pdf_key,
                    content_type="application/pdf",
                    length=len(pdf_bytes)
                )
                
                print(f"[DEV] PDF converted and uploaded: {pdf_key}")
                    
            except ConvertError as ce:
                raise RuntimeError(f"PDF 변환 실패: {ce}")
            except Exception as e:
                print(f"[DEV] Upload error: {e}")
                raise RuntimeError(f"MinIO 업로드 실패: {e}")
        
        # PDF 변환 완료 웹훅
        if callback_url:
            await send_dev_webhook(
                callback_url,
                DevWebhookPayload(
                    job_id=job_id,
                    data_id=data_id,
                    status="pdf_converted",
                    converted=not is_already_pdf,
                    metrics={"converted": not is_already_pdf},
                    timestamps={"start": start_time.isoformat()},
                    pdf_key_minio=pdf_key
                ),
                DEV_SECRET
            )
        
        # ========== Step 3: 단순 청킹 & 인덱싱 ==========
        job_state.update(job_id, status="processing", step="Simple chunking for development")
        
        # 3-1) PDF 텍스트 추출
        from app.services.file_parser import parse_pdf
        
        print(f"[DEV-CHUNK] Extracting text from: {converted_pdf_path}")
        pages_std = parse_pdf(converted_pdf_path, by_page=True)
        
        if not pages_std:
            raise RuntimeError("텍스트 추출 실패")
        
        print(f"[DEV-CHUNK] Extracted {len(pages_std)} pages")
        
        # 3-2) 단순 청킹
        from app.services.chunkers.simple_proofreading_chunker import simple_chunk_by_paragraph
        from app.services.embedding_model import get_embedding_model
        
        embed_model = get_embedding_model()
        encoder_fn = embed_model.tokenizer.encode
        
        print(f"[DEV-CHUNK] Chunking with simple proofreading chunker (paragraph-based)")
        
        chunks = simple_chunk_by_paragraph(
            pages_std,
            encoder_fn,
            target_tokens=400  # 조정 가능
        )
        
        if not chunks:
            raise RuntimeError("청킹 실패: 청크가 생성되지 않음")
        
        print(f"[DEV-CHUNK] Created {len(chunks)} chunks")
        
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
        
        print(f"[DEV-CHUNK] Embedding {len(chunk_texts)} chunks...")
        embeddings = embed(chunk_texts)
        
        # 3-4) Milvus 저장
        job_state.update(job_id, status="indexing", step="Indexing to Milvus")
        
        from app.services.milvus_store_v2 import MilvusStoreV2
        
        mvs = MilvusStoreV2()
        collection_name = os.getenv("MILVUS_COLLECTION_NAME", "rag_chunks_v2")
        
        # 기존 문서 삭제 (개발 모드에서 재테스트 시)
        print(f"[DEV-CHUNK] Deleting existing doc (if any): {data_id}")
        mvs.delete_by_doc_id(collection_name, data_id)
        
        # 청크 삽입
        print(f"[DEV-CHUNK] Inserting {len(chunks)} chunks to Milvus")
        
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
                    "type": meta.get('type', 'simple_dev_chunk'),
                    "token_count": meta.get('token_count', 0),
                    "char_count": meta.get('char_count', 0),
                    "file_id": file_id,
                    "data_id": data_id
                }
            )
        
        print(f"[DEV-CHUNK] Successfully indexed {len(chunks)} chunks")
        
        # ========== Step 4: 결과 조회 및 웹훅 전송 ==========
        pages = len(pages_std)
        chunk_count = len(chunks)
        
        print(f"[DEV] Completed: {pages} pages, {chunk_count} chunks")
        
        # Job 상태 업데이트 (DB 업데이트는 하지 않음)
        job_state.complete(
            job_id,
            pages=pages,
            chunks=chunk_count
        )
        
        if callback_url:
            end_time = datetime.utcnow()
            payload = DevWebhookPayload(
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
                message="Development mode: converted and indexed (no DB update)",
                pdf_key_minio=pdf_key
            )
            await send_dev_webhook(callback_url, payload, DEV_SECRET)
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        print(f"[DEV] Error: {e}")
        
        if callback_url:
            payload = DevWebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="error",
                message=str(e)
            )
            await send_dev_webhook(callback_url, payload, DEV_SECRET)
        
        raise


# ==================== Routes ====================
@router.post("/convert-and-index", response_model=DevConvertResponse)
async def dev_convert_and_index(
    request: DevConvertRequest,
    background_tasks: BackgroundTasks,
    x_dev_token: Optional[str] = Header(None)
):
    """개발용 트리거 API - 단순 청커 사용, DB 업데이트 없음"""
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized")
    
    staging_dir = Path(LOCAL_STAGING_PATH)
    file_path = staging_dir / request.file_id
    
    if not file_path.exists():
        raise HTTPException(404, f"Local file not found: {file_path}")
    
    job_id = str(uuid.uuid4())[:8]
    
    background_tasks.add_task(
        process_dev_convert_and_index,
        job_id=job_id,
        data_id=request.data_id,
        file_id=request.file_id,
        callback_url=request.callback_url
    )
    
    return DevConvertResponse(
        status="accepted",
        job_id=job_id,
        data_id=request.data_id,
        message="Development mode processing (simple chunker, no DB)"
    )


@router.get("/status/{data_id}", response_model=DevStatusResponse)
def dev_get_status(data_id: str, x_dev_token: Optional[str] = Header(None)):
    """상태 조회 (개발 모드에서는 job_state만 조회)"""
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized")
    
    # job_state에서 조회 (DB 조회 없음)
    state = job_state.get(data_id)
    
    if not state:
        raise HTTPException(404, f"Job not found: {data_id}")
    
    return DevStatusResponse(
        data_id=data_id,
        rag_index_status=state.get('status', 'unknown'),
        parse_yn=None,  # 개발 모드에서는 DB 없음
        chunk_count=state.get('chunks'),
        parse_start_dt=state.get('created_at'),
        parse_end_dt=state.get('completed_at'),
        milvus_doc_id=data_id
    )


@router.get("/health")
def dev_health_check():
    """헬스 체크"""
    return {
        "status": "ok", 
        "service": "dev-router (simple chunker, no DB)",
        "mode": "development"
    }