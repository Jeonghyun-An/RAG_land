# app/api/dev_router.py
"""
개발/테스트용 라우터
- 로컬 디렉토리 파일 사용
- DB 업데이트 없음
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
    webhook_url: Optional[str] = None
    ocr_manual_required: bool = False
    reindex_required_yn: bool = False


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
            print(f"[DEV-WEBHOOK] ✅ Sent to {url}")
    except Exception as e:
        print(f"[DEV-WEBHOOK] ❌ Failed: {e}")


def generate_minio_pdf_key(data_id: str) -> str:
    return f"dev/pdfs/{data_id}.pdf"


# ==================== Background Task ====================
async def process_dev_convert_and_index(
    job_id: str,
    data_id: str,
    file_id: str,
    webhook_url: Optional[str]
):
    """
    개발용 백그라운드 처리
    - llama_router의 index_pdf_to_milvus() 재사용
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
                
                print(f"[DEV] ✅ PDF converted and uploaded: {pdf_key}")
                    
            except ConvertError as ce:
                raise RuntimeError(f"PDF 변환 실패: {ce}")
            except Exception as e:
                print(f"[DEV] ❌ Upload error: {e}")
                raise RuntimeError(f"MinIO 업로드 실패: {e}")
        
        # PDF 변환 완료 웹훅
        if webhook_url:
            await send_dev_webhook(
                webhook_url,
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
        
        # ========== Step 3: 고도화된 청킹 & 인덱싱 (llama_router 재사용) ==========
        job_state.update(job_id, status="processing", step="Using advanced indexing pipeline")
        
        # 🔥 핵심: OCR 모드를 'never'로 설정 (변환된 PDF는 텍스트 포함)
        original_ocr_mode = os.environ.get("OCR_MODE")
        try:
            if not is_already_pdf:
                os.environ["OCR_MODE"] = "never"  # 변환된 PDF는 OCR 불필요
            
            # llama_router의 고도화된 인덱싱 파이프라인 호출
            from app.api.llama_router import index_pdf_to_milvus
            
            index_pdf_to_milvus(
                job_id=job_id,
                file_path=converted_pdf_path,
                minio_object=pdf_key,  # MinIO 경로도 전달
                uploaded=True,
                remove_local=False,  # 개발 모드에서는 로컬 파일 유지
                doc_id=data_id
            )
            
        finally:
            # 환경변수 복원
            if original_ocr_mode is not None:
                os.environ["OCR_MODE"] = original_ocr_mode
            elif "OCR_MODE" in os.environ:
                del os.environ["OCR_MODE"]
        
        # ========== Step 4: 결과 조회 및 웹훅 전송 ==========
        state = job_state.get(job_id) or {}
        chunks = state.get('chunks', 0)
        pages = state.get('pages', 0)
        
        print(f"[DEV] ✅ Completed: {pages} pages, {chunks} chunks")
        
        if webhook_url:
            end_time = datetime.utcnow()
            payload = DevWebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="done",
                converted=not is_already_pdf,
                metrics={"pages": pages, "chunks": chunks},
                chunk_count=chunks,
                timestamps={
                    "start": start_time.isoformat(), 
                    "end": end_time.isoformat()
                },
                message="Development mode: converted and indexed (no DB update)",
                pdf_key_minio=pdf_key
            )
            await send_dev_webhook(webhook_url, payload, DEV_SECRET)
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        print(f"[DEV] ❌ Error: {e}")
        
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
    """개발용 트리거 API"""
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
        webhook_url=request.webhook_url
    )
    
    return DevConvertResponse(
        status="accepted",
        job_id=job_id,
        data_id=request.data_id,
        message="Development mode processing"
    )


@router.get("/status/{data_id}", response_model=DevStatusResponse)
def dev_get_status(data_id: str, x_dev_token: Optional[str] = Header(None)):
    """상태 조회"""
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized")
    
    state = job_state.get(data_id)
    
    if not state:
        raise HTTPException(404, f"Job not found: {data_id}")
    
    return DevStatusResponse(
        data_id=data_id,
        rag_index_status=state.get('status', 'unknown'),
        parse_yn=state.get('parse_yn'),
        chunk_count=state.get('chunks'),
        parse_start_dt=state.get('created_at'),
        parse_end_dt=state.get('updated_at'),
        milvus_doc_id=data_id
    )


@router.get("/jobs")
def dev_list_jobs(x_dev_token: Optional[str] = Header(None)):
    """모든 Job 목록"""
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized")
    
    jobs = job_state.list_jobs(status=None, limit=100)
    return {"jobs": jobs, "count": len(jobs)}


@router.delete("/jobs/{job_id}")
def dev_clear_job(job_id: str, x_dev_token: Optional[str] = Header(None)):
    """Job 상태 삭제"""
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized")
    
    job_state.clear(job_id)
    return {"message": f"Job {job_id} cleared"}


@router.delete("/jobs")
def dev_clear_all_jobs(x_dev_token: Optional[str] = Header(None)):
    """모든 Job 삭제"""
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized")
    
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
    """테스트 파일 업로드"""
    if not verify_dev_token(x_dev_token):
        raise HTTPException(401, "Unauthorized")
    
    staging_dir = Path(LOCAL_STAGING_PATH)
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    file_id = os.path.basename(file.filename or "upload.bin")
    file_path = staging_dir / file_id
    
    content = await file.read()
    
    with open(file_path, 'wb') as f:
        f.write(content)
    
    return {
        "message": "Test file uploaded",
        "file_id": file_id,
        "file_path": str(file_path),
        "size": len(content)
    }