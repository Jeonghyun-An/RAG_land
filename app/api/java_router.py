# app/api/java_router.py
"""
Java 시스템 연동 라우터 (운영용)
- 서버 파일시스템 사용
- DB 완전 연동
- manual-ocr-and-index: DB에서 OCR 텍스트 가져와서 청킹/임베딩
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
    callback_url: Optional[str] = None


class ManualOCRAndIndexRequest(BaseModel):
    """자바 → AI 트리거 요청 (manual-ocr-and-index)"""
    data_id: str
    path: str  # 서버 파일시스템 상대 경로
    file_id: str
    callback_url: Optional[str] = None
    rag_yn: str = "N"  # "N" (신규 작업), "Y" (기존 작업 수정)


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

    
class DeleteDocumentRequest(BaseModel):
    """문서 삭제 요청"""
    data_id: str
    delete_from_minio: bool = True
    callback_url: Optional[str] = None


class DeleteDocumentResponse(BaseModel):
    """삭제 응답"""
    status: str
    data_id: str
    deleted_chunks: int
    deleted_files: List[str]
    message: str


# ==================== Helper Functions ====================
def verify_internal_token(token: Optional[str]) -> bool:
    """내부 토큰 검증"""
    if not SHARED_SECRET:
        return True  # 토큰 미설정 시 허용 (개발 환경)
    return hmac.compare_digest(token or "", SHARED_SECRET)


async def send_webhook(url: str, payload: WebhookPayload, secret: str):
    """자바 서버로 완료 웹훅 전송"""
    try:
        sig = hmac.new(secret.encode(), payload.model_dump_json().encode(), hashlib.sha256).hexdigest()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json=payload.model_dump(),
                headers={"X-Webhook-Signature": sig}
            )
            print(f"[WEBHOOK] Sent to {url}: {resp.status_code}")
    except Exception as e:
        print(f"[WEBHOOK] Failed to send: {e}")


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
    - 단순 청킹 적용
    """
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
        is_already_pdf = file_id.lower().endswith('.pdf')
        
        if is_already_pdf:
            converted_pdf_path = full_path
            print(f"[PROD] Already PDF: {converted_pdf_path}")
        else:
            job_state.update(job_id, status="converting", step="Converting to PDF")
            print(f"[PROD] Converting to PDF: {file_id}")
            
            converted_pdf_path = convert_to_pdf(full_path)
            
            if not converted_pdf_path or not os.path.exists(converted_pdf_path):
                raise ConvertError("PDF 변환 실패")
            
            print(f"[PROD] Conversion completed: {converted_pdf_path}")
            # 참고: DB 파일 경로는 자바 시스템에서 이미 관리하므로 여기서 업데이트 불필요
        
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
        from app.services.embedding_model import get_sentence_embedding_dimension
        
        mvs = MilvusStoreV2(dim=get_sentence_embedding_dimension())
        
        # 기존 문서 삭제 (재인덱싱 시)
        print(f"[PROD-CHUNK] Deleting existing doc (if any): {data_id}")
        try:
            deleted = mvs._delete_by_doc_id(data_id)
            print(f"[PROD-CHUNK] Deleted {deleted} existing chunks")
        except Exception as e:
            print(f"[PROD-CHUNK] Warning during delete: {e}")
        
        # 청크 삽입 (insert 메서드 사용)
        print(f"[PROD-CHUNK] Inserting {len(chunks)} chunks to Milvus")
        
        # insert 메서드가 기대하는 형식: [(text, metadata), ...]
        chunks_for_insert = [(text, meta) for text, meta in zip(chunk_texts, chunk_metas)]
        
        result = mvs.insert(
            doc_id=data_id,
            chunks=chunks_for_insert,
            embed_fn=embed
        )
        
        print(f"[PROD-CHUNK] ✅ Successfully indexed: {result.get('inserted', 0)} chunks")
        
        # ========== Step 4: 결과 조회 및 DB 업데이트 ==========
        pages = len(pages_std)
        chunk_count = len(chunks)
        
        print(f"[PROD] ✅ Indexing completed: {pages} pages, {chunk_count} chunks")
        
        # DB 업데이트
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
                converted=not file_id.lower().endswith('.pdf'),
                metrics={"pages": pages, "chunks": chunk_count},
                chunk_count=chunk_count,
                timestamps={
                    "start": start_time.isoformat(), 
                    "end": end_time.isoformat()
                },
                message="indexed successfully"
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
    - DB의 osk_ocr_data 테이블에서 OCR 텍스트를 가져와서 청킹/임베딩
    - rag_yn = "N": 신규 작업 (parse_yn: L → S)
    - rag_yn = "Y": 기존 작업 수정 (기존 청크 삭제 후 재인덱싱)
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
        # ========== Step 1: DB에서 OCR 텍스트 가져오기 ==========
        job_state.update(job_id, status="loading", step="Loading OCR text from DB")
        
        print(f"[MANUAL-OCR] Loading OCR text from DB for data_id={data_id}")
        
        # DB에서 페이지별 OCR 텍스트 조회
        pages_std = db.get_ocr_text_by_data_id(data_id)
        
        if not pages_std:
            raise RuntimeError(f"DB에 OCR 텍스트가 없습니다: data_id={data_id}")
        
        print(f"[MANUAL-OCR] Loaded {len(pages_std)} pages from DB")
        
        # ========== Step 2: 청킹 ==========
        job_state.update(job_id, status="chunking", step="Chunking OCR text")
        
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
        
        # ========== Step 3: 임베딩 ==========
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
        
        # ========== Step 4: Milvus 저장 ==========
        job_state.update(job_id, status="indexing", step="Indexing to Milvus")
        
        from app.services.milvus_store_v2 import MilvusStoreV2
        from app.services.embedding_model import get_sentence_embedding_dimension
        
        mvs = MilvusStoreV2(dim=get_sentence_embedding_dimension())
        
        # 기존 문서 삭제 (재인덱싱 시)
        print(f"[MANUAL-OCR-CHUNK] Deleting existing doc (if any): {data_id}")
        try:
            deleted = mvs._delete_by_doc_id(data_id)
            print(f"[MANUAL-OCR-CHUNK] Deleted {deleted} existing chunks")
        except Exception as e:
            print(f"[MANUAL-OCR-CHUNK] Warning during delete: {e}")
        
        # 청크 삽입 (insert 메서드 사용)
        print(f"[MANUAL-OCR-CHUNK] Inserting {len(chunks)} chunks to Milvus")
        
        # insert 메서드가 기대하는 형식: [(text, metadata), ...]
        chunks_for_insert = [(text, meta) for text, meta in zip(chunk_texts, chunk_metas)]
        
        result = mvs.insert(
            doc_id=data_id,
            chunks=chunks_for_insert,
            embed_fn=embed
        )
        
        print(f"[MANUAL-OCR-CHUNK] ✅ Successfully indexed: {result.get('inserted', 0)} chunks")
        
        # ========== Step 5: 결과 조회 및 DB 업데이트 ==========
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
        
        # ========== Step 6: 완료 & Webhook ==========
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


# ==================== Background Task (delete) ====================
async def process_delete_document(
    data_id: str,
    delete_from_minio: bool,
    callback_url: Optional[str]
):
    """
    문서 완전 삭제:
    1. Milvus에서 청크 삭제
    2. MinIO에서 파일 삭제 (옵션)
    3. DB 업데이트 (del_yn='Y')
    """
    db = DBConnector()
    
    try:
        print(f"[DELETE] Starting deletion for data_id={data_id}")
        
        # ========== Step 1: Milvus 청크 삭제 ==========
        from app.services.milvus_store_v2 import MilvusStoreV2
        from app.services.embedding_model import get_sentence_embedding_dimension
        
        mvs = MilvusStoreV2(dim=get_sentence_embedding_dimension())
        
        deleted_chunks = mvs._delete_by_doc_id(data_id)
        print(f"[DELETE] Deleted {deleted_chunks} chunks from Milvus")
        
        # ========== Step 2: MinIO 파일 삭제 (옵션) ==========
        deleted_files = []
        
        if delete_from_minio:
            from app.services.minio_store import MinIOStore
            
            minio = MinIOStore()
            
            # 메타데이터 조회
            meta = db.get_file_by_id(data_id)
            
            if meta:
                pdf_key = meta.get('minio_pdf_key')
                original_key = meta.get('minio_original_key')
                
                # PDF 삭제
                if pdf_key:
                    try:
                        minio.delete_file(pdf_key)
                        deleted_files.append(pdf_key)
                        print(f"[DELETE] Deleted PDF: {pdf_key}")
                    except Exception as e:
                        print(f"[DELETE] Failed to delete PDF: {e}")
                
                # 원본 파일 삭제
                if original_key and original_key != pdf_key:
                    try:
                        minio.delete_file(original_key)
                        deleted_files.append(original_key)
                        print(f"[DELETE] Deleted original: {original_key}")
                    except Exception as e:
                        print(f"[DELETE] Failed to delete original: {e}")
        
        # ========== Step 3: DB 업데이트 ==========
        sql = """
        UPDATE data_master
        SET del_yn = 'Y',
            del_dt = SYS_DATETIME,
            rag_index_status = 'deleted'
        WHERE data_id = ?
        """
        
        try:
            with db.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                cur.close()
            print(f"[DELETE] ✅ Updated DB: del_yn='Y'")
        except Exception as e:
            print(f"[DELETE] ⚠️  DB update failed: {e}")
        
        # ========== Step 4: 완료 & Webhook ==========
        if callback_url:
            payload = WebhookPayload(
                job_id="delete",
                data_id=data_id,
                status="deleted",
                message=f"Document deleted: {deleted_chunks} chunks, {len(deleted_files)} files",
                chunk_count=0
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
        
        print(f"[DELETE] ✅ Deletion completed for data_id={data_id}")
        
    except Exception as e:
        print(f"[DELETE] ❌ Error: {e}")
        
        if callback_url:
            payload = WebhookPayload(
                job_id="delete",
                data_id=data_id,
                status="error",
                message=f"Deletion failed: {e}"
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
    DB의 osk_ocr_data 테이블에서 OCR 텍스트를 가져와서 청킹/임베딩
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
        message=f"processing manual OCR from DB (rag_yn={request.rag_yn}, simple chunker)"
    )


@router.delete("/delete/{data_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    data_id: str,
    background_tasks: BackgroundTasks,
    delete_from_minio: bool = True,
    callback_url: Optional[str] = None,
    x_internal_token: Optional[str] = Header(None)
):
    """
    문서 완전 삭제 API (운영용)
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    # 문서 존재 확인
    db = DBConnector()
    existing = db.get_file_by_id(data_id)
    
    if not existing:
        raise HTTPException(404, f"Document not found: {data_id}")
    
    # 이미 삭제된 문서인지 확인
    if existing.get('del_yn') == 'Y':
        return DeleteDocumentResponse(
            status="already_deleted",
            data_id=data_id,
            deleted_chunks=0,
            deleted_files=[],
            message="Document already marked as deleted"
        )
    
    # 백그라운드 삭제 작업
    background_tasks.add_task(
        process_delete_document,
        data_id=data_id,
        delete_from_minio=delete_from_minio,
        callback_url=callback_url
    )
    
    return DeleteDocumentResponse(
        status="deleting",
        data_id=data_id,
        deleted_chunks=0,  # 백그라운드에서 처리
        deleted_files=[],
        message="Deletion in progress"
    )


@router.post("/delete-batch", response_model=Dict[str, Any])
async def delete_documents_batch(
    data_ids: List[str],
    background_tasks: BackgroundTasks,
    delete_from_minio: bool = True,
    callback_url: Optional[str] = None,
    x_internal_token: Optional[str] = Header(None)
):
    """
    여러 문서 일괄 삭제 API
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    if not data_ids:
        raise HTTPException(400, "data_ids cannot be empty")
    
    if len(data_ids) > 100:
        raise HTTPException(400, "Maximum 100 documents per batch")
    
    # 각 문서별로 백그라운드 작업 생성
    for data_id in data_ids:
        background_tasks.add_task(
            process_delete_document,
            data_id=data_id,
            delete_from_minio=delete_from_minio,
            callback_url=callback_url
        )
    
    return {
        "status": "deleting",
        "count": len(data_ids),
        "data_ids": data_ids,
        "message": f"Batch deletion started for {len(data_ids)} documents"
    }


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
        "chunker": "simple_proofreading_chunker",
        "manual_ocr": "DB-based (osk_ocr_data)"
    }