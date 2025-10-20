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
    - llama_router의 index_pdf_to_milvus() 재사용
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
        
        # ========== Step 2: PDF 변환 (필요시) ==========
        job_state.update(job_id, status="parsing", step="Converting to PDF if needed")
        
        is_already_pdf = file_id.lower().endswith('.pdf')
        converted_pdf_path = full_path
        
        if not is_already_pdf:
            try:
                # PDF 변환
                converted_pdf_path = convert_to_pdf(full_path)
                converted_name = os.path.splitext(file_id)[0] + '.pdf'
                
                # 서버 파일시스템에 저장
                final_pdf_path = os.path.join(SERVER_BASE_PATH, path, converted_name)
                
                # 변환된 파일 복사 (convert_to_pdf가 임시 경로에 생성했을 경우)
                if converted_pdf_path != final_pdf_path:
                    import shutil
                    shutil.copy2(converted_pdf_path, final_pdf_path)
                    converted_pdf_path = final_pdf_path
                
                # DB 업데이트: 변환된 파일 경로
                folder_only = path
                prefix = "COMMON/oskData/"
                if folder_only.startswith(prefix):
                    folder_only = folder_only[len(prefix):]
                db.update_converted_file_path(data_id, folder_only, converted_name)
                
                print(f"[PROD] ✅ PDF converted and saved: {converted_pdf_path}")
                    
            except ConvertError as ce:
                raise RuntimeError(f"PDF 변환 실패: {ce}")
            except Exception as e:
                print(f"[PROD] ❌ Conversion error: {e}")
                raise RuntimeError(f"PDF 변환/저장 실패: {e}")
        
        # PDF 변환 완료 웹훅
        if webhook_url:
            await send_webhook(
                webhook_url,
                WebhookPayload(
                    job_id=job_id,
                    data_id=data_id,
                    status="pdf_converted",
                    converted=not is_already_pdf,
                    metrics={"converted": not is_already_pdf},
                    timestamps={"start": start_time.isoformat()}
                ),
                SHARED_SECRET
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
                minio_object=None,  # 운영에서는 로컬 파일 우선
                uploaded=True,
                remove_local=False,  # 서버 파일은 유지
                doc_id=data_id
            )
            
        finally:
            # 환경변수 복원
            if original_ocr_mode is not None:
                os.environ["OCR_MODE"] = original_ocr_mode
            elif "OCR_MODE" in os.environ:
                del os.environ["OCR_MODE"]
        
        # ========== Step 4: 결과 조회 및 DB 업데이트 ==========
        state = job_state.get(job_id) or {}
        chunks = state.get('chunks', 0)
        pages = state.get('pages', 0)
        
        print(f"[PROD] ✅ Indexing completed: {pages} pages, {chunks} chunks")
        
        # DB 업데이트: RAG 인덱싱 완료
        db.update_rag_completed(
            data_id,
            chunks=chunks,
            doc_id=data_id
        )
        
        # ========== Step 5: 완료 & Webhook ==========
        end_time = datetime.utcnow()
        
        if webhook_url:
            payload = WebhookPayload(
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
                message="converted and indexed" if not is_already_pdf else "indexed"
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