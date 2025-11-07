# app/api/java_router.py
"""
Java 시스템 연동 라우터 (운영용)
- 서버 파일시스템 사용
- DB 완전 연동 (osk_data, osk_ocr_data, osk_ocr_hist)
- manual-ocr-and-index: rag_yn에 따른 신규/수정 처리
"""
from __future__ import annotations

import os
import re
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
from app.services.minio_store import MinIOStore
from datetime import timezone
import mimetypes

router = APIRouter(prefix="/java", tags=["java-production"])

# 환경변수
SHARED_SECRET = os.getenv("JAVA_SHARED_SECRET", "")
SERVER_BASE_PATH = os.getenv("SERVER_BASE_PATH", "/mnt/shared")

def META_KEY(doc_id: str) -> str:
    return f"uploaded/__meta__/{doc_id}/meta.json"

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
    path: str  # 서버 파일시스템 상대 경로 (사용하지 않을 수도 있음)
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
    parse_yn: Optional[str] = None
    parse_start_dt: Optional[str] = None
    parse_end_dt: Optional[str] = None

    
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
        return True
    return token == SHARED_SECRET


async def send_webhook(url: str, payload: WebhookPayload, secret: str):
    """AI → 자바 웹훅 전송"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {}
            if secret:
                sig = hmac.new(secret.encode(), payload.model_dump_json().encode(), hashlib.sha256).hexdigest()
                headers["X-Webhook-Signature"] = sig
            
            resp = await client.post(url, json=payload.model_dump(), headers=headers)
            resp.raise_for_status()
            print(f"[WEBHOOK] ✅ Sent to {url}: {payload.status}")
    except Exception as e:
        print(f"[WEBHOOK] ❌ Failed: {e}")


def _normalize_pages_for_chunkers(pages) -> List[Tuple[int, str]]:
    """
    페이지 정규화 - llama_router와 동일한 로직
    """
    out = []
    if not pages:
        return out

    for i, item in enumerate(pages, start=1):
        # (int,str) 튜플/리스트
        if isinstance(item, (tuple, list)):
            if len(item) >= 2:
                pno, txt = item[0], item[1]
            else:
                pno, txt = i, (item[0] if item else "")
            try:
                pno = int(pno)
            except Exception:
                pno = i
            out.append((pno, "" if txt is None else str(txt)))
            continue

        # dict
        if isinstance(item, dict):
            pno = item.get("page") or item.get("page_no") or item.get("index") or i
            txt = (
                item.get("text")
                or item.get("body")
                or ("\n".join(item.get("lines") or []) if item.get("lines") else "")
                or ""
            )
            try:
                pno = int(pno)
            except Exception:
                pno = i
            out.append((pno, str(txt)))
            continue

        # 문자열
        if isinstance(item, str):
            out.append((i, item))
            continue

        # 기타: 문자열화
        out.append((i, str(item)))

    return out


def _coerce_chunks_for_milvus(chs):
    """
    (텍스트, 메타) 리스트를 Milvus insert 형태로 정규화
    llama_router와 동일한 로직
    """
    safe = []
    for t in chs or []:
        if not isinstance(t, (list, tuple)) or len(t) < 2:
            continue
        text, meta = t[0], t[1]
        text = "" if text is None else str(text)
        if not isinstance(meta, dict):
            meta = {}

        # section 우선 결정
        section = str(meta.get("section", ""))[:512]
        # page 정규화: pages가 있으면 첫 페이지
        pages = meta.get("pages")
        if isinstance(pages, (list, tuple)) and len(pages) > 0:
            try:
                page = int(pages[0])
            except Exception:
                page = int(meta.get("page", 0))
        else:
            try:
                page = int(meta.get("page", 0))
            except Exception:
                page = 0

        safe.append((text, {"page": page, "section": section, "pages": pages or [], "bboxes": meta.get("bboxes", {})}))

    out = []
    last = None
    for it in safe:
        if it[0] and it != last:
            out.append(it)
            last = it
    return out


def perform_advanced_chunking(
    pages_std: List[Tuple[int, str]],
    layout_map: Dict[int, List[Dict]],
    job_id: str
) -> List[Tuple[str, Dict]]:
    """
    공용 청킹 파이프라인 호출 (llama_router와 동일)
    """
    from app.services.chunking_unified import build_chunks
    job_state.update(job_id, step="chunking:unified")
    return build_chunks(pages_std, layout_map, job_id=job_id)


# ==================== Endpoints ====================

@router.post("/convert-and-index", response_model=ConvertAndIndexResponse)
async def convert_and_index(
    request: ConvertAndIndexRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: Optional[str] = Header(None)
):
    """자바 → AI 트리거 API (운영용) - convert-and-index"""
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    db = DBConnector()
    existing = db.get_file_by_id(request.data_id)
    
    # 이미 완료된 경우 스킵 로직 (필요 시)
    # if existing and existing.get('parse_yn') == 'S':
    #     return ConvertAndIndexResponse(...)
    
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
        message="processing (advanced chunking)"
    )


@router.post("/manual-ocr-and-index", response_model=ConvertAndIndexResponse)
async def manual_ocr_and_index(
    request: ManualOCRAndIndexRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: Optional[str] = Header(None)
):
    """
    자바 → AI 트리거 API (운영용) - manual-ocr-and-index
    
    프로세스:
    1. rag_yn='N': 신규 OCR 작업
       - osk_data.parse_yn = 'L' 로 시작
       - osk_ocr_data에서 텍스트 가져와서 청킹/임베딩
       - 완료 시 parse_yn = 'S', osk_ocr_hist 로깅
    
    2. rag_yn='Y': 기존 작업 수정 (사용자가 페이지 수정)
       - osk_ocr_data에서 수정된 텍스트 가져와서 재청킹/임베딩
       - Milvus에서 기존 청크 삭제 후 새로 삽입
       - 완료 시 parse_yn = 'S', osk_ocr_hist 로깅
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
        message=f"processing manual OCR from DB (rag_yn={request.rag_yn})"
    )


@router.post("/delete-document", response_model=DeleteDocumentResponse)
async def delete_document(
    request: DeleteDocumentRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: Optional[str] = Header(None)
):
    """문서 삭제 API"""
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    background_tasks.add_task(
        process_delete_document,
        data_id=request.data_id,
        delete_from_minio=request.delete_from_minio,
        callback_url=request.callback_url
    )
    
    return DeleteDocumentResponse(
        status="deleting",
        data_id=request.data_id,
        deleted_chunks=0,
        deleted_files=[],
        message="Deletion started"
    )


@router.post("/batch-delete")
async def batch_delete(
    data_ids: List[str],
    delete_from_minio: bool = True,
    callback_url: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    x_internal_token: Optional[str] = Header(None)
):
    """배치 삭제 API"""
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    if not data_ids:
        raise HTTPException(400, "data_ids cannot be empty")
    
    if len(data_ids) > 100:
        raise HTTPException(400, "Maximum 100 documents per batch")
    
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
        parse_yn=meta.get('parse_yn'),
        parse_start_dt=str(meta.get('parse_start_dt')) if meta.get('parse_start_dt') else None,
        parse_end_dt=str(meta.get('parse_end_dt')) if meta.get('parse_end_dt') else None
    )


@router.get("/health")
def health_check():
    """헬스 체크"""
    return {
        "status": "ok", 
        "service": "java-router-production",
        "chunking": "advanced (en_tech → law → layout → basic)",
        "manual_ocr": "DB-based (osk_ocr_data)"
    }


# ==================== Background Tasks ====================

async def process_convert_and_index_prod(
    job_id: str,
    data_id: str,
    path: str,
    file_id: str,
    callback_url: Optional[str]
):
    """
    운영용 백그라운드 처리 - convert-and-index
    (기존 로직 유지 - OCR 포함)
    """
    from app.services.file_parser import parse_pdf, parse_pdf_blocks
    
    db = DBConnector()
    start_time = datetime.utcnow()
    
    job_state.start(job_id, data_id=data_id, file_id=file_id)
    
    try:
        # ========== Step 1: 파일 경로 확인 ==========
        job_state.update(job_id, status="initializing", step="Resolving file path")
        
        raw_path = Path(path)
        base = raw_path if raw_path.is_absolute() else Path(SERVER_BASE_PATH) / raw_path

        # base가 폴더이거나 확장자가 없으면 file_id를 붙여 실제 파일 경로 구성
        full_path = base if base.suffix else (base / file_id)

        if not full_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {full_path}")
         
        print(f"[PROD] Processing file: {full_path}")
        
        # ========== Step 2: PDF 변환 (필요 시) ==========
        is_already_pdf = str(full_path).lower().endswith(".pdf")
        
        if is_already_pdf:
            converted_pdf_path = str(full_path)
            print(f"[PROD] Already PDF: {converted_pdf_path}")
        else:
            job_state.update(job_id, status="converting", step="Converting to PDF")
            print(f"[PROD] Converting to PDF: {full_path}")
            
            converted_pdf_path = convert_to_pdf(str(full_path))
            if not converted_pdf_path or not Path(converted_pdf_path).exists():
                raise ConvertError("PDF 변환 실패")
            
            print(f"[PROD] Converted: {converted_pdf_path}")
            
            # DB 업데이트: 변환된 파일 경로
            rel_folder = str(Path(converted_pdf_path).parent.relative_to(SERVER_BASE_PATH))
            rel_filename = Path(converted_pdf_path).name
            db.update_converted_file_path(data_id, rel_folder, rel_filename)
        
        # ========== Step 3: OCR 시작 마킹 ==========
        db.mark_ocr_start(data_id)
        
        # ========== Step 4: 텍스트 추출 (OCR 포함) ==========
        job_state.update(job_id, status="parsing", step="Extracting text with OCR")
        
        print(f"[PROD-PARSE] Extracting text from: {converted_pdf_path}")
        pages = parse_pdf(converted_pdf_path, by_page=True)
        
        if not pages:
            raise RuntimeError("텍스트 추출 실패")
        
        print(f"[PROD-PARSE] Extracted {len(pages)} pages")
        
        # ========== Step 5: OCR 결과 DB 저장 ==========
        # 자바 요구사항: OCR 추출 종료 시 osk_ocr_data에 INSERT
        job_state.update(job_id, status="saving_ocr", step="Saving OCR results to DB")
        
        for page_no, text in pages:
            db.insert_ocr_result(data_id, page_no, text)
            print(f"[PROD-OCR-DB] Saved page {page_no} to osk_ocr_data")
        
        # OCR 성공 마킹 (parse_yn='S')
        db.mark_ocr_success(data_id)
        print(f"[PROD-OCR-DB] ✅ OCR completed and saved to DB: {len(pages)} pages")
        
        # ========== Step 6: 레이아웃 정보 추출 ==========
        blocks_by_page_list = parse_pdf_blocks(converted_pdf_path)
        layout_map = {int(p): blks for p, blks in (blocks_by_page_list or [])}
        print(f"[PROD-PARSE] Layout blocks extracted for {len(layout_map)} pages")
        
        # ========== Step 7: 페이지 정규화 ==========
        pages_std = _normalize_pages_for_chunkers(pages)
        if not any((t or "").strip() for _, t in pages_std):
            print("[PROD-PARSE] Warning: No textual content after parsing")
        
        # ========== Step 8: 고도화된 청킹 ==========
        job_state.update(job_id, status="chunking", step="Advanced chunking")
        
        chunks = perform_advanced_chunking(pages_std, layout_map, job_id)
        
        if not chunks:
            raise RuntimeError("청킹 실패: 청크가 생성되지 않음")
        
        # ========== Step 9: 청크 정규화 ==========
        chunks = _coerce_chunks_for_milvus(chunks)
        print(f"[PROD-CHUNK] Normalized {len(chunks)} chunks for Milvus")
        
        # ========== Step 10: 임베딩 및 Milvus 저장 ==========
        job_state.update(job_id, status="embedding", step=f"Embedding {len(chunks)} chunks")
        
        from app.services.embedding_model import embed, get_sentence_embedding_dimension
        from app.services.milvus_store_v2 import MilvusStoreV2
        
        mvs = MilvusStoreV2(dim=get_sentence_embedding_dimension())
        
        # 기존 문서 삭제
        print(f"[PROD] Deleting existing doc (if any): {data_id}")
        try:
            deleted = mvs._delete_by_doc_id(data_id)
            print(f"[PROD] Deleted {deleted} existing chunks")
        except Exception as e:
            print(f"[PROD] Warning during delete: {e}")
        
        # Milvus insert
        print(f"[PROD] Inserting {len(chunks)} chunks to Milvus")
        
        result = mvs.insert(
            doc_id=data_id,
            chunks=chunks,
            embed_fn=embed
        )
        
        print(f"[PROD] ✅ Successfully indexed: {result.get('inserted', 0)} chunks")
        
        # ========== Step 11: RAG 완료 처리 ==========
        pages_count = len(pages_std)
        chunk_count = result.get('inserted', len(chunks))
        # ========== Step 11.5: MinIO 동기화 (프론트 목록 노출용) ==========
        try:
            SYNC_TO_MINIO = os.getenv("JAVA_SYNC_TO_MINIO", "1") == "1"
            if SYNC_TO_MINIO:
                doc_id = str(data_id)  # 프론트 doc_id와 맞춤
                pdf_path_for_upload = converted_pdf_path  # 이미 확정된 경로
                if pdf_path_for_upload and Path(pdf_path_for_upload).exists():
                    m = MinIOStore()
                    object_pdf = f"uploaded/{doc_id}.pdf"

                    # 파일 -> bytes 업로드 (upload_file 대신 upload_bytes 사용)
                    with open(pdf_path_for_upload, "rb") as f:
                        data = f.read()
                    m.upload_bytes(
                        data,
                        object_name=object_pdf,
                        content_type="application/pdf",
                        length=len(data),
                    )

                    # meta.json 갱신(존재하면 merge)
                    meta = {}
                    try:
                        if m.exists(META_KEY(doc_id)):
                            meta = m.get_json(META_KEY(doc_id)) or {}
                    except Exception:
                        meta = {}

                    meta.update({
                        "doc_id": doc_id,
                        "title": Path(pdf_path_for_upload).name,
                        "pdf_key": object_pdf,                      # llama_router가 기대하는 키
                        "original_key": f"serverfs://{full_path}",  # 원본 참조용(옵션)
                        "original_name": Path(full_path).name,
                        "is_pdf_original": True,
                        "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "indexed": True,
                        "chunk_count": int(chunk_count),
                        "last_indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    })
                    m.put_json(META_KEY(doc_id), meta)

                    print(f"[PROD-MINIO] ✅ synced: {object_pdf} (chunks={chunk_count})")
                else:
                    print("[PROD-MINIO] ⚠️ skip: no local pdf to upload")
            else:
                print("[PROD-MINIO] ⏭️ skip: JAVA_SYNC_TO_MINIO=0")
        except Exception as e:
            # 동기화 실패해도 인덱싱 플로우는 유지
            print(f"[PROD-MINIO] ❌ sync failed: {e}")

        print(f"[PROD] ✅ Indexing completed: {pages_count} pages, {chunk_count} chunks")
        # RAG 완료 마킹 (parse_yn='S' 유지, 히스토리 로깅)
        db.update_rag_completed(data_id)
        
        job_state.complete(
            job_id,
            pages=pages_count,
            chunks=chunk_count
        )
        
        # ========== Step 12: 완료 & Webhook ==========
        end_time = datetime.utcnow()
        
        if callback_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="done",
                converted=not is_already_pdf,
                metrics={"pages": pages_count, "chunks": chunk_count},
                chunk_count=chunk_count,
                timestamps={
                    "start": start_time.isoformat(), 
                    "end": end_time.isoformat()
                },
                message="indexed successfully (advanced chunking)"
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        db.update_rag_error(data_id, str(e))
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
    
    프로세스:
    1. rag_yn='N': 신규 OCR 작업
       - osk_data.parse_yn = 'L' 로 시작
       - osk_ocr_data에서 텍스트 가져와서 청킹/임베딩
       - 완료 시 parse_yn = 'S'
    
    2. rag_yn='Y': 기존 작업 수정
       - osk_ocr_data에서 수정된 텍스트 가져와서 재청킹/임베딩
       - Milvus에서 기존 청크 삭제 후 새로 삽입
       - 완료 시 parse_yn = 'S'
    """
    db = DBConnector()
    start_time = datetime.utcnow()
    
    job_state.start(job_id, data_id=data_id, file_id=file_id)
    
    # rag_yn에 따른 DB 처리
    if rag_yn == "N":
        db.mark_ocr_start(data_id)
        print(f"[MANUAL-OCR] 신규 작업: data_id={data_id}, parse_yn='L'")
    else:
        print(f"[MANUAL-OCR] 기존 작업 수정: data_id={data_id}, rag_yn='Y'")
    
    try:
        # ========== Step 1: DB에서 OCR 텍스트 가져오기 ==========
        job_state.update(job_id, status="loading", step="Loading OCR text from DB")
        
        print(f"[MANUAL-OCR] Loading OCR text from osk_ocr_data for data_id={data_id}")
        
        pages_std = db.get_ocr_text_by_data_id(data_id)
        
        if not pages_std:
            raise RuntimeError(f"DB에 OCR 텍스트가 없습니다 (osk_ocr_data): data_id={data_id}")
        
        print(f"[MANUAL-OCR] Loaded {len(pages_std)} pages from DB")
        
        # ========== Step 2: 고도화된 청킹 ==========
        job_state.update(job_id, status="chunking", step="Advanced chunking from DB text")
        
        # Manual OCR은 레이아웃 정보가 없으므로 빈 dict 전달
        layout_map = {}
        
        chunks = perform_advanced_chunking(pages_std, layout_map, job_id)
        
        if not chunks:
            raise RuntimeError("청킹 실패: 청크가 생성되지 않음")
        
        # ========== Step 3: 청크 정규화 ==========
        chunks = _coerce_chunks_for_milvus(chunks)
        print(f"[MANUAL-OCR-CHUNK] Normalized {len(chunks)} chunks for Milvus")
        
        # ========== Step 4: 임베딩 및 Milvus 저장 ==========
        job_state.update(job_id, status="embedding", step=f"Embedding {len(chunks)} chunks")
        
        from app.services.embedding_model import embed, get_sentence_embedding_dimension
        from app.services.milvus_store_v2 import MilvusStoreV2
        
        mvs = MilvusStoreV2(dim=get_sentence_embedding_dimension())
        
        # 기존 문서 삭제 (rag_yn='Y'인 경우 반드시 필요)
        print(f"[MANUAL-OCR] Deleting existing doc (if any): {data_id}")
        try:
            deleted = mvs._delete_by_doc_id(data_id)
            print(f"[MANUAL-OCR] Deleted {deleted} existing chunks")
        except Exception as e:
            print(f"[MANUAL-OCR] Warning during delete: {e}")
        
        # Milvus insert
        print(f"[MANUAL-OCR] Inserting {len(chunks)} chunks to Milvus")
        
        result = mvs.insert(
            doc_id=data_id,
            chunks=chunks,
            embed_fn=embed
        )
        
        print(f"[MANUAL-OCR] ✅ Successfully indexed: {result.get('inserted', 0)} chunks")
        
        # ========== Step 5: DB 업데이트 ==========
        pages_count = len(pages_std)
        chunk_count = result.get('inserted', len(chunks))
        
        print(f"[MANUAL-OCR] ✅ Indexing completed: {pages_count} pages, {chunk_count} chunks")
        
        # RAG 완료 마킹 (parse_yn='S', 히스토리 로깅)
        db.update_rag_completed(data_id)
        
        job_state.complete(
            job_id,
            pages=pages_count,
            chunks=chunk_count
        )
        
        # ========== Step 6: 완료 & Webhook ==========
        end_time = datetime.utcnow()
        
        if callback_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="done",
                converted=False,
                metrics={"pages": pages_count, "chunks": chunk_count},
                chunk_count=chunk_count,
                timestamps={
                    "start": start_time.isoformat(), 
                    "end": end_time.isoformat()
                },
                message=f"manual OCR indexed (rag_yn={rag_yn}, advanced chunking)"
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        db.update_rag_error(data_id, str(e))
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


async def process_delete_document(
    data_id: str,
    delete_from_minio: bool,
    callback_url: Optional[str]
):
    """
    문서 완전 삭제:
    1. Milvus에서 청크 삭제
    2. MinIO에서 파일 삭제 (옵션)
    3. DB 상태 업데이트 (옵션)
    """
    from app.services.milvus_store_v2 import MilvusStoreV2
    from app.services.minio_store import MinIOStore
    from app.services.embedding_model import get_sentence_embedding_dimension
    
    try:
        # Milvus 삭제
        mvs = MilvusStoreV2(dim=get_sentence_embedding_dimension())
        deleted_count = mvs._delete_by_doc_id(data_id)
        
        deleted_files = []
        
        # MinIO 삭제 (옵션)
        if delete_from_minio:
            mstore = MinIOStore()
            # 파일 키 조회 및 삭제 로직 (필요 시)
            pass
        
        print(f"[DELETE] ✅ Deleted {deleted_count} chunks for data_id={data_id}")
        
        if callback_url:
            payload = WebhookPayload(
                job_id=f"delete-{data_id}",
                data_id=data_id,
                status="deleted",
                message=f"Deleted {deleted_count} chunks"
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
    
    except Exception as e:
        print(f"[DELETE] ❌ Error: {e}")
        
        if callback_url:
            payload = WebhookPayload(
                job_id=f"delete-{data_id}",
                data_id=data_id,
                status="error",
                message=str(e)
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)