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
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from fastapi import APIRouter, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
import httpx

from app.services.db_connector import DBConnector
from app.services.minio_store import MinIOStore
from app.services.milvus_store_v2 import MilvusStoreV2
from app.services.embedding_model import get_embedding_model
from app.services.pdf_converter import convert_to_pdf, ConvertError
from app.services.file_parser import parse_pdf_blocks
from app.services import job_state

router = APIRouter(prefix="/java", tags=["java-production"])

# 환경변수
SHARED_SECRET = os.getenv("JAVA_SHARED_SECRET", "landsoftSecret2025!Nuclear")
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


async def send_webhook(url: str, payload: WebhookPayload, secret: str, max_retries: int = 3):
    payload_json = payload.model_dump_json()
    signature = generate_hmac_signature(payload_json, secret)
    
    headers = {
        "Content-Type": "application/json",
        "X-RAG-Signature": signature
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, content=payload_json, headers=headers)
                if response.status_code < 500:
                    print(f"[WEBHOOK] Sent to {url}, status={response.status_code}")
                    return
                else:
                    print(f"[WEBHOOK] Attempt {attempt+1} failed with {response.status_code}")
            except Exception as e:
                print(f"[WEBHOOK] Attempt {attempt+1} error: {e}")
            
            if attempt < max_retries - 1:
                import asyncio
                await asyncio.sleep(2 ** attempt)


def _simple_chunk_fallback(pages_text: List[Tuple[int, str]], 
                           encoder_fn, 
                           target_tokens: int = 400) -> List[Tuple[str, Dict]]:
    """최종 폴백 청커"""
    chunks = []
    chunk_id = 0
    
    for page_no, text in pages_text:
        if not text or not text.strip():
            continue
            
        paragraphs = text.split('\n\n')
        current_text = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = len(encoder_fn(para))
            
            if current_tokens + para_tokens <= target_tokens:
                current_text += para + "\n\n"
                current_tokens += para_tokens
            else:
                if current_text.strip():
                    chunks.append((
                        current_text.strip(),
                        {'page': page_no, 'section': f'page_{page_no}_chunk_{chunk_id}', 'token_count': current_tokens}
                    ))
                    chunk_id += 1
                current_text = para + "\n\n"
                current_tokens = para_tokens
        
        if current_text.strip():
            chunks.append((
                current_text.strip(),
                {'page': page_no, 'section': f'page_{page_no}_chunk_{chunk_id}', 'token_count': current_tokens}
            ))
            chunk_id += 1
    
    return chunks


# ==================== Background Task ====================
async def process_convert_and_index_prod(
    job_id: str,
    data_id: str,
    path: str,
    file_id: str,
    webhook_url: Optional[str],
    ocr_manual_required: bool,
    reindex_required_yn: bool
):
    """
    운영용 백그라운드 처리
    - 서버 파일시스템 사용
    - DB 완전 업데이트
    """
    db = DBConnector()
    start_time = datetime.utcnow()
    
    job_state.start(job_id, data_id, file_id)
    db.mark_ocr_start(data_id)
    
    try:
        # ========== Step 1: 서버 파일 로드 ==========
        job_state.update(job_id, status="uploaded", step="Loading file from server")
        
        full_path = os.path.join(SERVER_BASE_PATH, path, file_id)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"서버 파일 없음: {full_path}")
        
        print(f"[PROD] Using server file: {full_path}")
        
        # ========== Step 2: PDF 변환 ==========
        job_state.update(job_id, status="parsing", step="Converting to PDF")
        
        is_already_pdf = file_id.lower().endswith('.pdf')
        converted_pdf_path = full_path
        
        if not is_already_pdf:
            try:
                pdf_bytes = convert_to_pdf(full_path)
                converted_name = os.path.splitext(file_id)[0] + '.pdf'
                
                # 서버 파일시스템에 저장
                converted_pdf_path = os.path.join(SERVER_BASE_PATH, path, converted_name)
                with open(converted_pdf_path, 'wb') as f:
                    f.write(pdf_bytes)
                
                # DB 업데이트: 변환된 파일 경로
                folder_only = path
                prefix = "COMMON/oskData/"
                if folder_only.startswith(prefix):
                    folder_only = folder_only[len(prefix):]
                db.update_converted_file_path(data_id, folder_only, converted_name)
                print(f"[PROD] PDF saved: {converted_pdf_path}")
                    
            except ConvertError as ce:
                raise RuntimeError(f"PDF 변환 실패: {ce}")
        
        # ========== Step 3: OCR 추출 ==========
        job_state.update(job_id, status="parsing", step="Extracting text with OCR")
        
        pages_text, pages_blocks = parse_pdf_blocks(converted_pdf_path)
        
        if not pages_text:
            raise RuntimeError("OCR 결과가 비어있습니다")
        
        # DB에 OCR 결과 저장
        db.mark_ocr_success(data_id)
        for page_no, text in pages_text:
            if text.strip():
                db.insert_ocr_result(data_id, page_no, text)
        
        # PDF+OCR 완료 웹훅
        if webhook_url:
            await send_webhook(
                webhook_url,
                WebhookPayload(
                    job_id=job_id,
                    data_id=data_id,
                    status="pdf_ocr_done",
                    converted=not is_already_pdf,
                    metrics={"pages": len(pages_text)},
                    timestamps={"start": start_time.isoformat()}
                ),
                SHARED_SECRET
            )
        
        # ========== Step 4: 청킹 ==========
        job_state.update(job_id, status="chunking", step="Chunking text")
        
        layout_map = {}
        if pages_blocks:
            for page_no, blocks in pages_blocks:
                layout_map[page_no] = blocks
        
        model = get_embedding_model()
        from sentence_transformers import SentenceTransformer
        encoder_model = SentenceTransformer(
            model.model_name_or_path if hasattr(model, 'model_name_or_path') 
            else 'sentence-transformers/all-MiniLM-L6-v2'
        )
        encoder_fn = encoder_model.tokenizer.encode
        
        chunks = None
        
        # 1순위: 원자력 법령/매뉴얼 청커
        try:
            from app.services.law_chunker import NuclearLegalChunker
            law_chunker = NuclearLegalChunker(encoder_fn=encoder_fn, target_tokens=400, overlap_tokens=100)
            chunks = law_chunker.chunk_pages(pages_text, layout_map)
            if chunks:
                print(f"[CHUNK] Nuclear legal chunker: {len(chunks)} chunks")
        except Exception as e:
            print(f"[CHUNK] Nuclear legal chunker failed: {e}")
        
        # 2순위: 레이아웃 인지 청커
        if not chunks:
            try:
                from app.services.layout_chunker import LayoutAwareChunker
                layout_chunker = LayoutAwareChunker(encoder_fn=encoder_fn, target_tokens=400, overlap_tokens=100)
                chunks = layout_chunker.chunk_pages(pages_text, layout_map)
                if chunks:
                    print(f"[CHUNK] Layout-aware chunker: {len(chunks)} chunks")
            except Exception as e:
                print(f"[CHUNK] Layout-aware chunker failed: {e}")
        
        # 3순위: 스마트 청커 플러스
        if not chunks:
            try:
                from app.services.chunker import smart_chunk_pages_plus
                chunks = smart_chunk_pages_plus(pages_text, encoder_fn, target_tokens=400, overlap_tokens=100, layout_blocks=layout_map)
                if chunks:
                    print(f"[CHUNK] Smart chunker plus: {len(chunks)} chunks")
            except Exception as e:
                print(f"[CHUNK] Smart chunker plus failed: {e}")
        
        # 4순위: 기본 스마트 청커
        if not chunks:
            try:
                from app.services.chunker import smart_chunk_pages
                chunks = smart_chunk_pages(pages_text, encoder_fn, target_tokens=400, overlap_tokens=100, layout_blocks=layout_map)
                if chunks:
                    print(f"[CHUNK] Basic smart chunker: {len(chunks)} chunks")
            except Exception as e:
                print(f"[CHUNK] Basic smart chunker failed: {e}")
        
        # 최종 폴백
        if not chunks:
            print("[CHUNK] Using fallback chunker")
            chunks = _simple_chunk_fallback(pages_text, encoder_fn, target_tokens=400)
        
        if not chunks:
            raise RuntimeError("청킹 결과가 비어있습니다")
        
        # ========== Step 5: 임베딩 & Milvus 적재 ==========
        job_state.update(job_id, status="embedding", step="Generating embeddings")
        
        from app.services.embedding_model import embed
        
        chunk_texts = [chunk[0] for chunk in chunks]
        embeddings = embed(chunk_texts)
        
        job_state.update(job_id, status="indexing", step="Indexing to Milvus")
        
        milvus = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())
        
        chunk_ids = [f"{data_id}_{i}" for i in range(len(chunks))]
        doc_ids = [data_id] * len(chunks)
        sections = []
        
        for chunk_text, chunk_meta in chunks:
            section_info = chunk_meta.get('section', '')
            if len(section_info) > 512:
                section_info = section_info[:512]
            sections.append(section_info)
        
        milvus.upsert(
            chunk_ids=chunk_ids,
            doc_ids=doc_ids,
            sections=sections,
            embeddings=embeddings,
            chunks=chunk_texts
        )
        
        # ========== Step 6: DB 업데이트 ==========
        job_state.update(job_id, status="cleanup", step="Updating database")
        
        end_time = datetime.utcnow()
        
        db.update_rag_completed(
            data_id,
            chunks=len(chunks),
            doc_id=data_id
        )
        
        # ========== Step 7: 완료 & Webhook ==========
        job_state.complete(job_id, pages=len(pages_text), chunks=len(chunks))
        
        if webhook_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="done",
                converted=not is_already_pdf,
                metrics={"pages": len(pages_text), "chunks": len(chunks)},
                chunk_count=len(chunks),
                timestamps={"start": start_time.isoformat(), "end": end_time.isoformat()},
                message="converted and indexed" if not is_already_pdf else "indexed"
            )
            await send_webhook(webhook_url, payload, SHARED_SECRET)
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        db.update_parse_status(data_id, rag_status="error")
        
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