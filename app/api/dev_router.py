# app/api/dev_router.py
"""
개발/테스트용 라우터
- 로컬 디렉토리 파일 사용
- DB 업데이트 없음
- Webhook 페이로드만 전달
- 빠른 테스트 및 디버깅용
"""
from __future__ import annotations

import os
import hashlib
import hmac
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from fastapi import APIRouter, HTTPException, Header, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
import httpx

from app.services.minio_store import MinIOStore
from app.services.milvus_store_v2 import MilvusStoreV2
from app.services.embedding_model import get_embedding_model
from app.services.pdf_converter import convert_to_pdf, ConvertError
from app.services.file_parser import parse_pdf_blocks
from app.services import job_state

router = APIRouter(prefix="/dev", tags=["development"])

# 환경변수
DEV_SECRET = os.getenv("DEV_SECRET", "devSecret2025")
LOCAL_STAGING_PATH = os.getenv("LOCAL_STAGING_PATH", "/tmp/remote_staging")


# ==================== Schemas ====================
class DevConvertRequest(BaseModel):
    """개발용 트리거 요청"""
    data_id: str
    file_id: str  # 로컬 스테이징 디렉토리 내 파일명
    webhook_url: Optional[str] = None


class DevConvertResponse(BaseModel):
    """즉시 응답"""
    status: str
    job_id: str
    data_id: str
    message: str
    local_file_path: str


class DevWebhookPayload(BaseModel):
    """개발용 Webhook 페이로드"""
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
    """개발용 상태 응답 (메모리 기반)"""
    data_id: str
    status: str
    step: str
    pages: Optional[int] = None
    chunks: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error: Optional[str] = None


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


async def send_dev_webhook(url: str, payload: DevWebhookPayload, secret: str, max_retries: int = 3):
    payload_json = payload.model_dump_json()
    signature = generate_hmac_signature(payload_json, secret)
    
    headers = {
        "Content-Type": "application/json",
        "X-Dev-Signature": signature
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, content=payload_json, headers=headers)
                if response.status_code < 500:
                    print(f"[DEV-WEBHOOK] Sent to {url}, status={response.status_code}")
                    return
                else:
                    print(f"[DEV-WEBHOOK] Attempt {attempt+1} failed with {response.status_code}")
            except Exception as e:
                print(f"[DEV-WEBHOOK] Attempt {attempt+1} error: {e}")
            
            if attempt < max_retries - 1:
                import asyncio
                await asyncio.sleep(2 ** attempt)


def generate_minio_pdf_key(data_id: str) -> str:
    now = datetime.utcnow()
    return f"dev_converted/{now.year}/{now.month:02d}/{data_id}.pdf"


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
async def process_dev_convert_and_index(
    job_id: str,
    data_id: str,
    file_id: str,
    webhook_url: Optional[str]
):
    """
    개발용 백그라운드 처리
    - 로컬 파일 사용
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
        
        # ========== Step 2: PDF 변환 ==========
        job_state.update(job_id, status="parsing", step="Converting to PDF")
        
        is_already_pdf = file_id.lower().endswith('.pdf')
        pdf_key: Optional[str] = None
        converted_pdf_path = str(file_path)
        
        if not is_already_pdf:
            try:
                pdf_bytes = convert_to_pdf(str(file_path))
                
                # MinIO에 저장
                pdf_key = generate_minio_pdf_key(data_id)
                store.put_bytes(pdf_key, pdf_bytes)
                
                # 로컬 임시 파일로도 저장 (OCR 처리용)
                temp_pdf = staging_dir / f"{data_id}_converted.pdf"
                with open(temp_pdf, 'wb') as f:
                    f.write(pdf_bytes)
                converted_pdf_path = str(temp_pdf)
                print(f"[DEV] PDF saved to MinIO: {pdf_key}")
                    
            except ConvertError as ce:
                raise RuntimeError(f"PDF 변환 실패: {ce}")
        
        # ========== Step 3: OCR 추출 ==========
        job_state.update(job_id, status="parsing", step="Extracting text with OCR")
        
        pages_text, pages_blocks = parse_pdf_blocks(converted_pdf_path)
        
        if not pages_text:
            raise RuntimeError("OCR 결과가 비어있습니다")
        
        print(f"[DEV] OCR extracted {len(pages_text)} pages")
        
        # PDF+OCR 완료 웹훅
        if webhook_url:
            await send_dev_webhook(
                webhook_url,
                DevWebhookPayload(
                    job_id=job_id,
                    data_id=data_id,
                    status="pdf_ocr_done",
                    converted=not is_already_pdf,
                    metrics={"pages": len(pages_text)},
                    timestamps={"start": start_time.isoformat()},
                    pdf_key_minio=pdf_key
                ),
                DEV_SECRET
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
                print(f"[DEV-CHUNK] Nuclear legal chunker: {len(chunks)} chunks")
        except Exception as e:
            print(f"[DEV-CHUNK] Nuclear legal chunker failed: {e}")
        
        # 2순위: 레이아웃 인지 청커
        if not chunks:
            try:
                from app.services.layout_chunker import LayoutAwareChunker
                layout_chunker = LayoutAwareChunker(encoder_fn=encoder_fn, target_tokens=400, overlap_tokens=100)
                chunks = layout_chunker.chunk_pages(pages_text, layout_map)
                if chunks:
                    print(f"[DEV-CHUNK] Layout-aware chunker: {len(chunks)} chunks")
            except Exception as e:
                print(f"[DEV-CHUNK] Layout-aware chunker failed: {e}")
        
        # 3순위: 스마트 청커 플러스
        if not chunks:
            try:
                from app.services.chunker import smart_chunk_pages_plus
                chunks = smart_chunk_pages_plus(pages_text, encoder_fn, target_tokens=400, overlap_tokens=100, layout_blocks=layout_map)
                if chunks:
                    print(f"[DEV-CHUNK] Smart chunker plus: {len(chunks)} chunks")
            except Exception as e:
                print(f"[DEV-CHUNK] Smart chunker plus failed: {e}")
        
        # 4순위: 기본 스마트 청커
        if not chunks:
            try:
                from app.services.chunker import smart_chunk_pages
                chunks = smart_chunk_pages(pages_text, encoder_fn, target_tokens=400, overlap_tokens=100, layout_blocks=layout_map)
                if chunks:
                    print(f"[DEV-CHUNK] Basic smart chunker: {len(chunks)} chunks")
            except Exception as e:
                print(f"[DEV-CHUNK] Basic smart chunker failed: {e}")
        
        # 최종 폴백
        if not chunks:
            print("[DEV-CHUNK] Using fallback chunker")
            chunks = _simple_chunk_fallback(pages_text, encoder_fn, target_tokens=400)
        
        if not chunks:
            raise RuntimeError("청킹 결과가 비어있습니다")
        
        # 청킹 완료 웹훅
        if webhook_url:
            await send_dev_webhook(
                webhook_url,
                DevWebhookPayload(
                    job_id=job_id,
                    data_id=data_id,
                    status="chunking_done",
                    metrics={"chunks": len(chunks)},
                    chunk_count=len(chunks),
                    timestamps={"start": start_time.isoformat()}
                ),
                DEV_SECRET
            )
        
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
        
        print(f"[DEV] Indexed {len(chunks)} chunks to Milvus")
        
        # ========== Step 6: 완료 (DB 업데이트 없음) ==========
        job_state.complete(job_id, pages=len(pages_text), chunks=len(chunks))
        
        if webhook_url:
            end_time = datetime.utcnow()
            payload = DevWebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="done",
                converted=not is_already_pdf,
                metrics={"pages": len(pages_text), "chunks": len(chunks)},
                chunk_count=len(chunks),
                timestamps={"start": start_time.isoformat(), "end": end_time.isoformat()},
                message="Development mode: converted and indexed (no DB update)",
                pdf_key_minio=pdf_key
            )
            await send_dev_webhook(webhook_url, payload, DEV_SECRET)
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        print(f"[DEV] Error: {e}")
        
        if webhook_url:
            payload = DevWebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="error",
                message=str(e)
            )
            await send_dev_webhook(webhook_url, payload, DEV_SECRET)
        
        raise


# ==================== Routes ====================
@router.post("/convert-and-index", response_model=DevConvertResponse)
async def dev_convert_and_index(
    request: DevConvertRequest,
    background_tasks: BackgroundTasks,
    x_dev_token: Optional[str] = Header(None)
):
    """
    개발용 트리거 API
    - 로컬 스테이징 디렉토리 파일 사용
    - DB 업데이트 없음
    """
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized - Invalid dev token")
    
    # 로컬 파일 존재 확인
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
        webhook_url=request.webhook_url
    )
    
    return DevConvertResponse(
        status="accepted",
        job_id=job_id,
        data_id=request.data_id,
        message="Development mode processing (no DB update)",
        local_file_path=str(file_path)
    )


@router.get("/status/{data_id}", response_model=DevStatusResponse)
def dev_get_status(data_id: str, x_dev_token: Optional[str] = Header(None)):
    """
    개발용 상태 조회 (메모리 기반)
    """
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized - Invalid dev token")
    
    state = job_state.get(data_id)
    
    if not state:
        raise HTTPException(404, f"Job not found: {data_id}")
    
    return DevStatusResponse(
        data_id=data_id,
        status=state.get('status', 'unknown'),
        step=state.get('step', ''),
        pages=state.get('pages'),
        chunks=state.get('chunks'),
        start_time=state.get('start_time'),
        end_time=state.get('end_time'),
        error=state.get('error')
    )


@router.get("/jobs")
def dev_list_jobs(x_dev_token: Optional[str] = Header(None)):
    """
    개발용: 모든 Job 목록 조회
    """
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized - Invalid dev token")
    
    # list_jobs 함수 사용
    jobs = job_state.list_jobs(status=None, limit=100)
    
    return {"jobs": jobs, "count": len(jobs)}


@router.delete("/jobs/{job_id}")
def dev_clear_job(job_id: str, x_dev_token: Optional[str] = Header(None)):
    """
    개발용: Job 상태 삭제
    """
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized - Invalid dev token")
    
    job_state.clear(job_id)
    return {"message": f"Job {job_id} cleared"}


@router.delete("/jobs")
def dev_clear_all_jobs(x_dev_token: Optional[str] = Header(None)):
    """
    개발용: 모든 Job 상태 삭제
    """
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized - Invalid dev token")
    
    job_state.clear_all()
    return {"message": "All jobs cleared"}


@router.get("/health")
def dev_health_check():
    """헬스 체크"""
    staging_dir = Path(LOCAL_STAGING_PATH)
    return {
        "status": "ok",
        "service": "dev-router",
        "local_staging_path": str(staging_dir),
        "staging_dir_exists": staging_dir.exists(),
        "staging_dir_writable": os.access(staging_dir, os.W_OK) if staging_dir.exists() else False
    }


@router.post("/upload-test-file")
async def dev_upload_test_file(
    file: UploadFile = File(...),
    x_dev_token: Optional[str] = Header(None)
):
    """
    개발용: 테스트 파일 업로드 (로컬 스테이징에 저장)
    """
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized - Invalid dev token")
    
    staging_dir = Path(LOCAL_STAGING_PATH)
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일명 안전하게 처리
    file_id = os.path.basename(file.filename or "upload.bin")
    file_path = staging_dir / file_id
    
    # 파일 내용 읽기
    content = await file.read()
    
    # 파일 저장
    with open(file_path, 'wb') as f:
        f.write(content)
    
    return {
        "message": "Test file uploaded",
        "file_id": file_id,
        "file_path": str(file_path),
        "size": len(content)
    }