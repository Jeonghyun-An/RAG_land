# app/api/java_router.py
"""
Java 시스템 연동 라우터 (운영용)
- 서버 파일시스템 사용
- DB 완전 연동
- manual-ocr-and-index: llama_router의 고도화된 청킹 시스템 적용
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


def _make_encoder():
    """임베딩 모델의 토크나이저 인코더 생성"""
    from app.services.embedding_model import get_embedding_model
    
    m = get_embedding_model()
    tok = getattr(m, "tokenizer", None)
    max_len = int(getattr(m, "max_seq_length", 128))
    
    def enc(s: str):
        if tok is None:
            return []
        return tok.encode(s, add_special_tokens=False) or []
    
    return enc, max_len


def _normalize_pages_for_chunkers(pages):
    """
    pages를 [(page_no:int, text:str), ...] 로 강제 변환.
    llama_router와 동일한 로직
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
    llama_router의 고도화된 청킹 시스템 적용
    
    우선순위:
    1. English Technical Chunker (영어 기술 문서)
    2. Law Chunker (법령 문서)
    3. Layout-aware Chunker (레이아웃 정보 활용)
    4. Basic Smart Chunker
    5. Fallback Protection
    """
    print("[CHUNK] Starting advanced chunking system...")
    
    # 인코더/길이
    enc, max_len = _make_encoder()
    default_target = max(64, max_len - 16)
    default_overlap = min(50, default_target // 4)

    target_tokens = int(os.getenv("RAG_TARGET_TOKENS", str(default_target)))
    overlap_tokens = int(os.getenv("RAG_OVERLAP_TOKENS", str(default_overlap)))
    min_chunk_tokens = int(os.getenv("RAG_MIN_CHUNK_TOKENS", "100"))

    chunks: list[tuple[str, dict]] | None = None

    # 1) English Technical Chunker (최우선)
    ENABLE_EN_TECH_CHUNKER = os.getenv("RAG_ENABLE_EN_TECH_CHUNKER", "1") == "1"
    
    if ENABLE_EN_TECH_CHUNKER:
        try:
            from app.services.english_technical_chunker import english_technical_chunk_pages
            print("[CHUNK] Trying English technical chunker (IAEA/standards optimized)...")
            
            en_target_tokens = int(os.getenv("RAG_EN_TARGET_TOKENS", "800"))
            
            chunks = english_technical_chunk_pages(
                pages_std, enc, en_target_tokens, overlap_tokens, layout_map
            )
            
            if chunks and len(chunks) > 0:
                print(f"[CHUNK] ✅ English technical chunker: {len(chunks)} chunks")
                job_state.update(job_id, status="chunking", step=f"en_tech:{len(chunks)}")
            else:
                print("[CHUNK] English technical chunker returned empty, falling back")
                chunks = None
        except Exception as e:
            print(f"[CHUNK] English technical chunker error: {e}")
            chunks = None

    # 2) Law Chunker (법령 문서)
    if chunks is None:
        ENABLE_LAW_CHUNKER = os.getenv("RAG_ENABLE_LAW_CHUNKER", "1") == "1"
        
        if ENABLE_LAW_CHUNKER:
            try:
                from app.services.law_chunker import law_chunk_pages
                print("[CHUNK] Trying law chunker (nuclear/legal optimized)...")
                
                chunks = law_chunk_pages(
                    pages_std, enc, target_tokens, overlap_tokens,
                    layout_blocks=layout_map, min_chunk_tokens=min_chunk_tokens
                )
                
                if chunks and len(chunks) > 0:
                    print(f"[CHUNK] ✅ Law chunker: {len(chunks)} chunks")
                    job_state.update(job_id, status="chunking", step=f"law:{len(chunks)}")
                else:
                    print("[CHUNK] Law chunker returned empty, falling back")
                    chunks = None
            except Exception as e:
                print(f"[CHUNK] Law chunker error: {e}")
                chunks = None

    # 3) Layout-aware Chunker (레이아웃 정보 활용)
    if chunks is None:
        ENABLE_LAYOUT_CHUNKER = os.getenv("RAG_ENABLE_LAYOUT_CHUNKER", "1") == "1"
        
        if ENABLE_LAYOUT_CHUNKER and layout_map:
            try:
                from app.services.chunker import smart_chunk_pages_plus
                print("[CHUNK] Using layout-aware chunker (SmartChunkerPlus)...")
                
                chunks = smart_chunk_pages_plus(
                    pages_std, enc, target_tokens, overlap_tokens, layout_map
                )
                
                if chunks and len(chunks) > 0:
                    print(f"[CHUNK] ✅ Layout chunker: {len(chunks)} chunks")
                    job_state.update(job_id, status="chunking", step=f"layout:{len(chunks)}")
                else:
                    print("[CHUNK] Layout chunker returned empty, falling back")
                    chunks = None
            except Exception as e:
                print(f"[CHUNK] Layout chunker error: {e}")
                chunks = None

    # 4) Basic Smart Chunker
    if chunks is None:
        try:
            from app.services.chunker import smart_chunk_pages
            print("[CHUNK] Using basic smart chunker...")
            
            chunks = smart_chunk_pages(
                pages_std, enc, target_tokens, overlap_tokens, layout_map
            )
            
            if chunks and len(chunks) > 0:
                print(f"[CHUNK] ✅ Basic chunker: {len(chunks)} chunks")
                job_state.update(job_id, status="chunking", step=f"basic:{len(chunks)}")
            else:
                raise RuntimeError("Basic chunker returned empty")
        except Exception as e:
            print(f"[CHUNK] Basic chunker error: {e}")
            raise RuntimeError(f"모든 청킹 방법 실패: {e}")

    # 5) Fallback Protection
    if not chunks or len(chunks) == 0:
        print("[CHUNK] All chunkers failed - using fallback protection")
        
        flat_texts = []
        for _, t in pages_std or []:
            tt = (t or "").strip()
            if tt:
                # 이상한 라벨 제거
                tt = re.sub(r'\b인접행\s*묶음\b', '', tt)
                tt = re.sub(r'\b[가-힣]*\s*묶음\b', '', tt)
                tt = re.sub(r'[\r\n\s]+', ' ', tt)
                if tt.strip():
                    flat_texts.append(tt.strip())

        fallback_text = "\n\n".join(flat_texts).strip()

        if not fallback_text:
            if os.getenv("RAG_ALLOW_EMPTY_FALLBACK", "1") == "1":
                fallback_text = "[Document processed but no readable text content found]"
            else:
                raise RuntimeError("모든 청킹 방법이 실패했습니다.")

        # 폴백 청크 생성
        tokens = len(enc(fallback_text))
        chunks = [(fallback_text, {"page": 1, "pages": [1], "section": "", "token_count": tokens, "bboxes": {}})]
        print(f"[CHUNK] Fallback chunk created: {tokens} tokens")
        job_state.update(job_id, status="chunking", step=f"fallback:1")

    print(f"[CHUNK] ✅ Final result: {len(chunks)} chunks ready for embedding")
    return chunks


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
    - 고도화된 청킹 시스템 적용
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
        
        # ========== Step 3: PDF 파싱 (텍스트 + 레이아웃) ==========
        job_state.update(job_id, status="parsing", step="Parsing PDF")
        
        from app.services.file_parser import parse_pdf, parse_pdf_blocks
        
        print(f"[PROD-PARSE] Extracting text from: {converted_pdf_path}")
        pages = parse_pdf(converted_pdf_path, by_page=True)
        
        if not pages:
            raise RuntimeError("텍스트 추출 실패")
        
        print(f"[PROD-PARSE] Extracted {len(pages)} pages")
        
        # 레이아웃 정보 추출
        blocks_by_page_list = parse_pdf_blocks(converted_pdf_path)
        layout_map = {int(p): blks for p, blks in (blocks_by_page_list or [])}
        print(f"[PROD-PARSE] Layout blocks extracted for {len(layout_map)} pages")
        
        # ========== Step 4: 페이지 정규화 ==========
        pages_std = _normalize_pages_for_chunkers(pages)
        if not any((t or "").strip() for _, t in pages_std):
            print("[PROD-PARSE] Warning: No textual content after parsing")
        
        # ========== Step 5: 고도화된 청킹 ==========
        job_state.update(job_id, status="chunking", step="Advanced chunking")
        
        chunks = perform_advanced_chunking(pages_std, layout_map, job_id)
        
        if not chunks:
            raise RuntimeError("청킹 실패: 청크가 생성되지 않음")
        
        # ========== Step 6: 청크 정규화 ==========
        chunks = _coerce_chunks_for_milvus(chunks)
        print(f"[PROD-CHUNK] Normalized {len(chunks)} chunks for Milvus")
        
        # ========== Step 7: 임베딩 및 Milvus 저장 ==========
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
        
        # Milvus insert 메서드 사용
        print(f"[PROD] Inserting {len(chunks)} chunks to Milvus")
        
        result = mvs.insert(
            doc_id=data_id,
            chunks=chunks,
            embed_fn=embed
        )
        
        print(f"[PROD] ✅ Successfully indexed: {result.get('inserted', 0)} chunks")
        
        # ========== Step 8: DB 업데이트 ==========
        pages_count = len(pages_std)
        chunk_count = result.get('inserted', len(chunks))
        
        print(f"[PROD] ✅ Indexing completed: {pages_count} pages, {chunk_count} chunks")
        
        db.update_rag_completed(
            data_id,
            chunks=chunk_count,
            doc_id=data_id
        )
        
        job_state.complete(
            job_id,
            pages=pages_count,
            chunks=chunk_count
        )
        
        # ========== Step 9: 완료 & Webhook ==========
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
    - 고도화된 청킹 시스템 적용
    """
    db = DBConnector()
    start_time = datetime.utcnow()
    
    job_state.start(job_id, data_id=data_id, file_id=file_id)
    
    # rag_yn에 따른 DB 처리
    if rag_yn == "N":
        db.mark_ocr_start(data_id)
        print(f"[MANUAL-OCR] 신규 작업: data_id={data_id}, parse_yn=L")
    else:
        print(f"[MANUAL-OCR] 기존 작업 수정: data_id={data_id}, rag_yn=Y")
    
    try:
        # ========== Step 1: DB에서 OCR 텍스트 가져오기 ==========
        job_state.update(job_id, status="loading", step="Loading OCR text from DB")
        
        print(f"[MANUAL-OCR] Loading OCR text from DB for data_id={data_id}")
        
        pages_std = db.get_ocr_text_by_data_id(data_id)
        
        if not pages_std:
            raise RuntimeError(f"DB에 OCR 텍스트가 없습니다: data_id={data_id}")
        
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
        
        # 기존 문서 삭제
        print(f"[MANUAL-OCR] Deleting existing doc (if any): {data_id}")
        try:
            deleted = mvs._delete_by_doc_id(data_id)
            print(f"[MANUAL-OCR] Deleted {deleted} existing chunks")
        except Exception as e:
            print(f"[MANUAL-OCR] Warning during delete: {e}")
        
        # Milvus insert 메서드 사용
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
        
        db.update_rag_completed(
            data_id,
            chunks=chunk_count,
            doc_id=data_id
        )
        
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
    고도화된 청킹 시스템 적용
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
        message="processing (advanced chunking: en_tech → law → layout → basic)"
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
    고도화된 청킹 시스템 적용
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
        message=f"processing manual OCR from DB (rag_yn={request.rag_yn}, advanced chunking)"
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
        deleted_chunks=0,
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
        "chunking": "advanced (en_tech → law → layout → basic)",
        "manual_ocr": "DB-based (osk_ocr_data)"
    }