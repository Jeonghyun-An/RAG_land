# app/api/java_router.py
"""
Java 시스템 연동 라우터 (운영용)
- 서버 파일시스템 사용
- DB 완전 연동
- manual-ocr-and-index 엔드포인트: rag_yn 파라미터 추가
- 단순 청커(simple_proofreading_chunker) 사용
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
    """자바 → AI 트리거 요청 (convert-and-index)"""
    data_id: str
    path: str  # 서버 파일시스템 상대 경로
    file_id: str
    callback_url: Optional[str] = None  # ✅ callback_url로 변경


class ManualOCRAndIndexRequest(BaseModel):
    """자바 → AI 트리거 요청 (manual-ocr-and-index)"""
    data_id: str
    path: str  # 서버 파일시스템 상대 경로
    file_id: str
    callback_url: Optional[str] = None
    rag_yn: str = "N"  # ✅ 신규 파라미터: "N" (신규 작업), "Y" (기존 작업 수정)


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


# ==================== Background Task (convert-and-index) ====================
async def process_convert_and_index_prod(
    job_id: str,
    data_id: str,
    path: str,
    file_id: str,
    callback_url: Optional[str]
):
    """
    운영용 백그라운드 처리 - convert-and-index
    - parse_yn: L → S
    - 단순 청킹 적용
    """
    db = DBConnector()
    start_time = datetime.utcnow()
    
    job_state.start(job_id, data_id=data_id, file_id=file_id)
    db.mark_ocr_start(data_id)  # parse_yn = 'L', parse_start_dt 설정
    
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
                converted_pdf_path = convert_to_pdf(full_path)
                print(f"[PROD] ✅ PDF converted: {converted_pdf_path}")
            except ConvertError as ce:
                raise RuntimeError(f"PDF 변환 실패: {ce}")
        
        # ========== Step 3: 단순 청킹 & 인덱싱 ==========
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
        print(f"[PROD-CHUNK] Deleting existing doc (if any): {data_id}")
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
        
        # DB 업데이트: RAG 인덱싱 완료 (parse_yn = 'S')
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
        
        if callback_url:
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
            await send_webhook(callback_url, payload, SHARED_SECRET)
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        db.update_parse_status(data_id, rag_status="error")
        print(f"[PROD] ❌ Error: {e}")
        
        if callback_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="error",
                message=str(e)
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
        
        raise


# ==================== Background Task (manual-ocr-and-index) ====================
async def process_manual_ocr_and_index(
    job_id: str,
    data_id: str,
    path: str,
    file_id: str,
    callback_url: Optional[str],
    rag_yn: str
):
    """
    운영용 백그라운드 처리 - manual-ocr-and-index
    - rag_yn = "N": 신규 작업 (parse_yn: L → S)
    - rag_yn = "Y": 기존 작업 수정 (기존 청크 삭제 후 재인덱싱)
    - 단순 청킹 적용
    """
    db = DBConnector()
    start_time = datetime.utcnow()
    
    job_state.start(job_id, data_id=data_id, file_id=file_id)
    
    # rag_yn에 따른 DB 처리
    if rag_yn == "N":
        # 신규 작업: parse_yn = 'L', parse_start_dt 설정
        db.mark_ocr_start(data_id)
        print(f"[MANUAL-OCR] 신규 작업: data_id={data_id}, parse_yn=L")
    else:
        # 기존 작업 수정: parse_yn은 그대로 유지, 재인덱싱만 진행
        print(f"[MANUAL-OCR] 기존 작업 수정: data_id={data_id}, rag_yn=Y")
    
    try:
        # ========== Step 1: 서버 파일 로드 ==========
        job_state.update(job_id, status="uploaded", step="Loading file from server")
        
        full_path = os.path.join(SERVER_BASE_PATH, path, file_id)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"서버 파일 없음: {full_path}")
        
        print(f"[MANUAL-OCR] Using server file: {full_path}")
        
        # ========== Step 2: PDF 확인 (이미 PDF여야 함) ==========
        if not file_id.lower().endswith('.pdf'):
            raise ValueError("manual-ocr-and-index는 PDF 파일만 지원합니다")
        
        converted_pdf_path = full_path
        
        # ========== Step 3: 단순 청킹 & 인덱싱 ==========
        job_state.update(job_id, status="processing", step="Simple chunking for manual OCR")
        
        # 3-1) PDF 텍스트 추출
        from app.services.file_parser import parse_pdf
        
        print(f"[MANUAL-OCR-CHUNK] Extracting text from: {converted_pdf_path}")
        pages_std = parse_pdf(converted_pdf_path, by_page=True)
        
        if not pages_std:
            raise RuntimeError("텍스트 추출 실패")
        
        print(f"[MANUAL-OCR-CHUNK] Extracted {len(pages_std)} pages")
        
        # 3-2) 단순 청킹
        from app.services.simple_proofreading_chunker import simple_chunk_by_paragraph
        from app.services.embedding_model import get_embedding_model
        
        embed_model = get_embedding_model()
        encoder_fn = embed_model.tokenizer.encode
        
        print(f"[MANUAL-OCR-CHUNK] Chunking with simple proofreading chunker (paragraph-based)")
        
        chunks = simple_chunk_by_paragraph(
            pages_std,
            encoder_fn,
            target_tokens=400  # 조정 가능
        )
        
        if not chunks:
            raise RuntimeError("청킹 실패: 청크가 생성되지 않음")
        
        print(f"[MANUAL-OCR-CHUNK] Created {len(chunks)} chunks")
        
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
        
        print(f"[MANUAL-OCR-CHUNK] Embedding {len(chunk_texts)} chunks...")
        embeddings = embed(chunk_texts)
        
        # 3-4) Milvus 저장
        job_state.update(job_id, status="indexing", step="Indexing to Milvus")
        
        from app.services.milvus_store_v2 import MilvusStoreV2
        
        mvs = MilvusStoreV2()
        collection_name = os.getenv("MILVUS_COLLECTION_NAME", "nuclear_rag")
        
        # 기존 문서 삭제 (재인덱싱 or 수정 작업)
        print(f"[MANUAL-OCR-CHUNK] Deleting existing doc: {data_id}")
        mvs.delete_by_doc_id(collection_name, data_id)
        
        # 청크 삽입
        print(f"[MANUAL-OCR-CHUNK] Inserting {len(chunks)} chunks to Milvus")
        
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
                    "type": meta.get('type', 'manual_ocr_chunk'),
                    "token_count": meta.get('token_count', 0),
                    "char_count": meta.get('char_count', 0),
                    "file_id": file_id,
                    "data_id": data_id,
                    "rag_yn": rag_yn
                }
            )
        
        print(f"[MANUAL-OCR-CHUNK] ✅ Successfully indexed {len(chunks)} chunks")
        
        # ========== Step 4: 결과 조회 및 DB 업데이트 ==========
        pages = len(pages_std)
        chunk_count = len(chunks)
        
        print(f"[MANUAL-OCR] ✅ Indexing completed: {pages} pages, {chunk_count} chunks")
        
        # DB 업데이트
        if rag_yn == "N":
            # 신규 작업: parse_yn = 'S'로 업데이트
            db.update_rag_completed(
                data_id,
                chunks=chunk_count,
                doc_id=data_id
            )
        else:
            # 기존 작업 수정: chunk_count만 업데이트
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
        
        if callback_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="done",
                converted=False,  # manual-ocr는 이미 PDF
                metrics={"pages": pages, "chunks": chunk_count},
                chunk_count=chunk_count,
                timestamps={
                    "start": start_time.isoformat(), 
                    "end": end_time.isoformat()
                },
                message=f"manual OCR indexed (rag_yn={rag_yn})"
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        db.update_parse_status(data_id, rag_status="error")
        print(f"[MANUAL-OCR] ❌ Error: {e}")
        
        if callback_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="error",
                message=str(e)
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
        
        raise


# ==================== Routes ====================
@router.post("/convert-and-index", response_model=ConvertAndIndexResponse)
async def convert_and_index(
    request: ConvertAndIndexRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: Optional[str] = Header(None)
):
    """
    자바 → AI 트리거 API (운영용) - convert-and-index
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    # 중복 체크
    db = DBConnector()
    existing = db.get_file_by_id(request.data_id)
    
    if existing and existing.get('rag_index_status') == 'done':
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
        callback_url=request.callback_url
    )
    
    return ConvertAndIndexResponse(
        status="accepted",
        job_id=job_id,
        data_id=request.data_id,
        message="processing (simple chunker)"
    )


@router.post("/manual-ocr-and-index", response_model=ConvertAndIndexResponse)
async def manual_ocr_and_index(
    request: ManualOCRAndIndexRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: Optional[str] = Header(None)
):
    """
    자바 → AI 트리거 API (운영용) - manual-ocr-and-index
    rag_yn: "N" (신규), "Y" (수정)
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    job_id = str(uuid.uuid4())[:8]
    
    background_tasks.add_task(
        process_manual_ocr_and_index,
        job_id=job_id,
        data_id=request.data_id,
        path=request.path,
        file_id=request.file_id,
        callback_url=request.callback_url,
        rag_yn=request.rag_yn
    )
    
    return ConvertAndIndexResponse(
        status="accepted",
        job_id=job_id,
        data_id=request.data_id,
        message=f"processing manual OCR (rag_yn={request.rag_yn}, simple chunker)"
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
    return {
        "status": "ok", 
        "service": "java-router-production",
        "chunker": "simple_proofreading_chunker"
    }