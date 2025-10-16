# ==================== Helper Functions ====================
def verify_internal_token(token: Optional[str]) -> bool:
    """내부 API 토큰 검증"""
    if not token:
        return False
    return token == SHARED_SECRET


def generate_hmac_signature(payload: str, secret: str) -> str:
    """HMAC-SHA256 서명 생성"""
    return hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def _simple_chunk_fallback(pages: List[Tuple[int, str]], encoder_fn, target_tokens: int = 400) -> List[Tuple[str, Dict]]:
    """
    최종 폴백 청커 - 단순 토큰 기반 분할
    """
    chunks = []
    chunk_id = 0
    
    for page_no, text in pages:
        if not text.strip():
            continue
        
        # 텍스트를 문장 단위로 분리
        sentences = text.replace('\n', ' ').split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            try:
                sentence_tokens = len(encoder_fn(sentence))
            except:
                sentence_tokens = len(sentence.split())  # 단어 수로 근사
            
            if current_tokens + sentence_tokens > target_tokens and current_chunk:
                # 청크 완성
                chunk_text = ' '.join(current_chunk)
                chunks.append((
                    chunk_text,
                    {
                        'page': page_no,
                        'section': f'page_{page_no}_chunk_{chunk_id}',
                        'token_count': current_tokens
                    }
                ))
                chunk_id += 1
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # 마지막 청크 처리
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((
                chunk_text,
                {
                    'page': page_no,
                    'section': f'page_{page_no}_chunk_{chunk_id}',
                    'token_count': current_tokens
                }
            ))
            chunk_id += 1
    
    return chunks


async def send_webhook(url: str, payload: WebhookPayload, secret: str, max_retries: int = 3):
    """
    Webhook 전송 (지수 백오프 재시도)
    """
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
                if response.status_code < 500:  # 2xx, 3xx, 4xx는 재시도 안함
                    print(f"[WEBHOOK] Sent to {url}, status={response.status_code}")
                    return
                else:
                    print(f"[WEBHOOK] Attempt {attempt+1} failed with {response.status_code}")
            except Exception as e:
                print(f"[WEBHOOK] Attempt {attempt+1} error: {e}")
            
            # 지수 백오프
            if attempt < max_retries - 1:
                import asyncio
                await asyncio.sleep(2 ** attempt)


def get_converted_path(base_path: str, file_id: str) -> str:
    """
    변환된 파일 경로 계산
    - PDF는 동일명 유지
    - 기타는 .pdf 확장자로 변경
    """
    if file_id.lower().endswith('.pdf'):
        return os.path.join(base_path, file_id)
    else:
        name_without_ext = os.path.splitext(file_id)[0]
        return os.path.join(base_path, f"{name_without_ext}.pdf")


def generate_minio_pdf_key(data_id: str) -> str:
    """MinIO PDF 키 생성 (simulate_remote 모드용)"""
    now = datetime.utcnow()
    return f"converted/{now.year}/{now.month:02d}/{data_id}.pdf"# app/api/java_router.py
"""
Java 시스템과 통신하는 전용 라우터
- 트리거 API: convert-and-index
- 폴링 API: status, files
- Webhook 콜백 전송
- simulate_remote 플래그 처리 (운영/개발 모드 전환)

[수정 필요 파일]
1. app/main.py - java_router 등록 필요
2. app/services/db_connector.py - simulated_yn 컬럼 처리 추가 (선택)
3. .env - JAVA_SHARED_SECRET, SERVER_BASE_PATH 추가
"""
from __future__ import annotations

import os
import hashlib
import hmac
import json
import tempfile
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Header, BackgroundTasks, Request
from pydantic import BaseModel
import httpx

from app.services.db_connector import DBConnector
from app.services.minio_store import MinIOStore
from app.services.milvus_store_v2 import MilvusStoreV2
from app.services.embedding_model import get_embedding_model
from app.services.pdf_converter import convert_to_pdf, ConvertError
from app.services.file_parser import parse_pdf, parse_pdf_blocks
from app.services import job_state

router = APIRouter(prefix="/java", tags=["java"])

# 공유 시크릿 (환경변수)
SHARED_SECRET = os.getenv("JAVA_SHARED_SECRET", "default-secret-change-me")

# 서버 파일 시스템 베이스 경로 (실제 운영 시 마운트 경로)
SERVER_BASE_PATH = os.getenv("SERVER_BASE_PATH", "/mnt/shared")

# OCR 관련 설정
OCR_BYTES_ENABLED = os.getenv("OCR_BYTES_ENABLED", "1") == "1"


# ==================== Schemas ====================
class ConvertAndIndexRequest(BaseModel):
    """자바 → AI 트리거 요청"""
    data_id: str  # 필수: 문서 PK
    path: str  # 필수: 서버 기준 상대 경로 (예: COMMON/oskData/2023/12/21)
    file_id: str  # 필수: 원본 파일명
    simulate_remote: bool = False  # 임시 모드 여부
    webhook_url: Optional[str] = None  # 콜백 URL
    ocr_manual_required: bool = False  # OCR 수동 처리 필요 여부
    reindex_required_yn: bool = False  # 재임베딩 필요 여부


class ConvertAndIndexResponse(BaseModel):
    """즉시 응답 (접수 확인)"""
    status: str  # "accepted"
    job_id: str
    data_id: str
    message: str


class WebhookPayload(BaseModel):
    """AI → 자바 콜백 페이로드"""
    job_id: str
    data_id: str
    status: str  # "done" | "error" | "self_ocr_required"
    converted: bool = False
    simulated: bool = False
    metrics: Optional[Dict[str, Any]] = None
    timestamps: Optional[Dict[str, str]] = None
    message: str = ""
    pdf_key_minio: Optional[str] = None  # MinIO 키 (simulate_remote=True일 때)


class StatusResponse(BaseModel):
    """폴링용 상태 응답"""
    data_id: str
    rag_index_status: str  # "queued" | "running" | "done" | "error" | "self_ocr_required"
    parse_yn: Optional[str] = None
    chunk_count: Optional[int] = None
    parse_start_dt: Optional[str] = None
    parse_end_dt: Optional[str] = None
    ocr_failed_yn: Optional[str] = None


class FilesResponse(BaseModel):
    """변환본 메타 정보"""
    data_id: str
    converted_path: Optional[str] = None  # 서버 파일 경로
    minio_pdf_key: Optional[str] = None  # MinIO 키
    minio_original_key: Optional[str] = None
    simulated_yn: Optional[str] = None


# ==================== Helper Functions ====================
def verify_internal_token(token: Optional[str]) -> bool:
    """내부 API 토큰 검증"""
    if not token:
        return False
    return token == SHARED_SECRET


def generate_hmac_signature(payload: str, secret: str) -> str:
    """HMAC-SHA256 서명 생성"""
    return hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


async def send_webhook(url: str, payload: WebhookPayload, secret: str, max_retries: int = 3):
    """
    Webhook 전송 (지수 백오프 재시도)
    """
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
                if response.status_code < 500:  # 2xx, 3xx, 4xx는 재시도 안함
                    print(f"[WEBHOOK] Sent to {url}, status={response.status_code}")
                    return
                else:
                    print(f"[WEBHOOK] Attempt {attempt+1} failed with {response.status_code}")
            except Exception as e:
                print(f"[WEBHOOK] Attempt {attempt+1} error: {e}")
            
            # 지수 백오프
            if attempt < max_retries - 1:
                import asyncio
                await asyncio.sleep(2 ** attempt)


def get_converted_path(base_path: str, file_id: str) -> str:
    """
    변환된 파일 경로 계산
    - PDF는 동일명 유지
    - 기타는 .pdf 확장자로 변경
    """
    if file_id.lower().endswith('.pdf'):
        return os.path.join(base_path, file_id)
    else:
        name_without_ext = os.path.splitext(file_id)[0]
        return os.path.join(base_path, f"{name_without_ext}.pdf")


def generate_minio_pdf_key(data_id: str) -> str:
    """MinIO PDF 키 생성 (simulate_remote 모드용)"""
    now = datetime.utcnow()
    return f"converted/{now.year}/{now.month:02d}/{data_id}.pdf"


# ==================== Background Task ====================
async def process_convert_and_index(
    job_id: str,
    data_id: str,
    path: str,
    file_id: str,
    simulate_remote: bool,
    webhook_url: Optional[str],
    ocr_manual_required: bool,
    reindex_required_yn: bool
):
    """
    백그라운드 처리 작업
    1. 파일 로드 (로컬 또는 임시)
    2. PDF 변환
    3. OCR 추출
    4. 청킹/임베딩
    5. Milvus 적재
    6. DB 업데이트
    7. Webhook 전송
    """
    db = DBConnector()
    store = MinIOStore()
    start_time = datetime.utcnow()
    
    # 상태 초기화
    job_state.start(job_id, data_id, "")
    
    # ✅ 자바 규격: OCR 시작 (parse_yn='L')
    db.mark_ocr_start(data_id)
    
    try:
        # ========== Step 1: 파일 로드 ==========
        job_state.update(job_id, status="uploaded", step="Loading file")
        
        if simulate_remote:
            # 임시 모드: 로컬 스테이징 파일 사용
            # 실제로는 Java가 미리 업로드한 파일을 MinIO에서 가져오거나
            # 로컬 임시 경로에서 읽는다고 가정
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, file_id)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"임시 파일을 찾을 수 없습니다: {file_path}")
        else:
            # 운영 모드: 서버 파일시스템 접근
            full_path = os.path.join(SERVER_BASE_PATH, path, file_id)
            
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"서버 파일을 찾을 수 없습니다: {full_path}")
            
            file_path = full_path
        
        # ========== Step 2: PDF 변환 ==========
        job_state.update(job_id, status="parsing", step="Converting to PDF")
        
        converted_pdf_path = None
        is_already_pdf = file_id.lower().endswith('.pdf')
        
        if not is_already_pdf:
            # 변환 필요
            try:
                with open(file_path, 'rb') as f:
                    original_bytes = f.read()
                
                pdf_bytes = convert_to_pdf(file_path)
                
                # 변환된 PDF 저장 위치
                converted_name = os.path.splitext(file_id)[0] + '.pdf'
                
                if simulate_remote:
                    # MinIO에 저장
                    pdf_key = generate_minio_pdf_key(data_id)
                    store.put_bytes(pdf_key, pdf_bytes)
                    converted_pdf_path = None  # MinIO 키로 대체
                else:
                    # 서버 파일시스템에 저장 (기존 파일 replace)
                    converted_pdf_path = os.path.join(SERVER_BASE_PATH, path, converted_name)
                    with open(converted_pdf_path, 'wb') as f:
                        f.write(pdf_bytes)
                    file_path = converted_pdf_path  # 이후 처리는 변환본 사용
                    
            except ConvertError as e:
                raise RuntimeError(f"PDF 변환 실패: {e}")
        else:
            # 이미 PDF - 변환 스킵
            converted_pdf_path = file_path
        
        # ========== Step 3: OCR 추출 ==========
        job_state.update(job_id, status="parsing", step="Extracting text (OCR)")
        
        pages_text = parse_pdf(file_path, by_page=True)
        pages_blocks = parse_pdf_blocks(file_path)
        
        if not pages_text or not any(text.strip() for _, text in pages_text):
            # OCR 실패 처리
            if ocr_manual_required:
                # 자바측 OCR 필요
                db.update_parse_status(data_id, ocr_failed=True, rag_status="self_ocr_required")
                
                if webhook_url:
                    payload = WebhookPayload(
                        job_id=job_id,
                        data_id=data_id,
                        status="self_ocr_required",
                        message="Low quality scan - manual OCR required"
                    )
                    await send_webhook(webhook_url, payload, SHARED_SECRET)
                
                return
            else:
                raise RuntimeError("텍스트 추출 실패")
        
        # OCR 텍스트 DB 저장
        for page_no, text in pages_text:
            if text.strip():
                db.insert_ocr_result(data_id, page_no, text)
        
        # ========== Step 4: 청킹 ==========
        job_state.update(job_id, status="chunking", step="Chunking text")
        
        # 레이아웃 맵 생성
        layout_map = {}
        if pages_blocks:
            for page_no, blocks in pages_blocks:
                layout_map[page_no] = blocks
        
        # 임베딩 모델 및 인코더 준비
        model = get_embedding_model()
        from sentence_transformers import SentenceTransformer
        encoder_model = SentenceTransformer(
            model.model_name_or_path if hasattr(model, 'model_name_or_path') 
            else 'sentence-transformers/all-MiniLM-L6-v2'
        )
        encoder_fn = encoder_model.tokenizer.encode
        
        # 청킹 전략: 다단계 폴백
        chunks = None
        
        # 1순위: 원자력 법령/매뉴얼 전용 청커
        try:
            from app.services.law_chunker import NuclearLegalChunker
            law_chunker = NuclearLegalChunker(
                encoder_fn=encoder_fn,
                target_tokens=400,
                overlap_tokens=100
            )
            chunks = law_chunker.chunk_pages(pages_text, layout_map)
            if chunks:
                job_state.update(job_id, step="Using nuclear legal chunker")
                print(f"[CHUNK] Nuclear legal chunker succeeded: {len(chunks)} chunks")
        except Exception as e:
            print(f"[CHUNK] Nuclear legal chunker failed: {e}")
        
        # 2순위: 레이아웃 인지 청커
        if not chunks:
            try:
                from app.services.layout_chunker import LayoutAwareChunker
                layout_chunker = LayoutAwareChunker(
                    encoder_fn=encoder_fn,
                    target_tokens=400,
                    overlap_tokens=100
                )
                chunks = layout_chunker.chunk_pages(pages_text, layout_map)
                if chunks:
                    job_state.update(job_id, step="Using layout-aware chunker")
                    print(f"[CHUNK] Layout-aware chunker succeeded: {len(chunks)} chunks")
            except Exception as e:
                print(f"[CHUNK] Layout-aware chunker failed: {e}")
        
        # 3순위: 스마트 청커 플러스
        if not chunks:
            try:
                from app.services.chunker import smart_chunk_pages_plus
                chunks = smart_chunk_pages_plus(
                    pages_text,
                    encoder_fn,
                    target_tokens=400,
                    overlap_tokens=100,
                    layout_blocks=layout_map
                )
                if chunks:
                    job_state.update(job_id, step="Using smart chunker plus")
                    print(f"[CHUNK] Smart chunker plus succeeded: {len(chunks)} chunks")
            except Exception as e:
                print(f"[CHUNK] Smart chunker plus failed: {e}")
        
        # 4순위: 기본 스마트 청커
        if not chunks:
            try:
                from app.services.chunker import smart_chunk_pages
                chunks = smart_chunk_pages(
                    pages_text,
                    encoder_fn,
                    target_tokens=400,
                    overlap_tokens=100,
                    layout_blocks=layout_map
                )
                if chunks:
                    job_state.update(job_id, step="Using basic smart chunker")
                    print(f"[CHUNK] Basic smart chunker succeeded: {len(chunks)} chunks")
            except Exception as e:
                print(f"[CHUNK] Basic smart chunker failed: {e}")
        
        # 최종 폴백: 단순 분할
        if not chunks:
            print("[CHUNK] All advanced chunkers failed, using simple fallback")
            job_state.update(job_id, step="Using fallback chunker")
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
        
        # Milvus 데이터 준비
        chunk_ids = [f"{data_id}_{i}" for i in range(len(chunks))]
        doc_ids = [data_id] * len(chunks)
        sections = []
        
        for chunk_text, chunk_meta in chunks:
            section_info = chunk_meta.get('section', '')
            if len(section_info) > 512:
                section_info = section_info[:512]
            sections.append(section_info)
        
        # Upsert
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
        
        # simulate_remote 플래그 기록 (필요시 커스텀 컬럼 추가)
        if simulate_remote:
            # simulated_yn 컬럼이 있다면
            try:
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("UPDATE data_master SET simulated_yn='Y' WHERE data_id=?", (data_id,))
                    cur.close()
            except Exception:
                pass
        
        # MinIO 키 저장
        if simulate_remote and converted_pdf_path is None:
            pdf_key = generate_minio_pdf_key(data_id)
            try:
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("UPDATE data_master SET minio_pdf_key=? WHERE data_id=?", (pdf_key, data_id))
                    cur.close()
            except Exception:
                pass
        
        # ========== Step 7: 완료 & Webhook ==========
        job_state.complete(
            job_id,
            pages=len(pages_text),
            chunks=len(chunks)
        )
        
        if webhook_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="done",
                converted=not is_already_pdf,
                simulated=simulate_remote,
                metrics={
                    "pages": len(pages_text),
                    "chunks": len(chunks)
                },
                timestamps={
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                message="converted and indexed",
                pdf_key_minio=generate_minio_pdf_key(data_id) if simulate_remote else None
            )
            await send_webhook(webhook_url, payload, SHARED_SECRET)
    
    except Exception as e:
        # 에러 처리
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
    자바 → AI 트리거 API
    파일 변환 및 인덱싱 작업 시작
    """
    # 토큰 검증
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    # 중복 처리 방지 (같은 data_id 재처리 시)
    db = DBConnector()
    existing = db.get_file_by_id(request.data_id)
    
    if existing and existing.get('rag_index_status') == 'done':
        # 이미 완료된 경우 - 재처리 안함 (reindex_required_yn=True면 예외)
        if not request.reindex_required_yn:
            return ConvertAndIndexResponse(
                status="already_done",
                job_id="",
                data_id=request.data_id,
                message="Already indexed"
            )
    
    # Job ID 생성
    job_id = str(uuid.uuid4())[:8]
    
    # 백그라운드 작업 등록
    background_tasks.add_task(
        process_convert_and_index,
        job_id=job_id,
        data_id=request.data_id,
        path=request.path,
        file_id=request.file_id,
        simulate_remote=request.simulate_remote,
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
    """
    폴링용 상태 조회 API
    """
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
        ocr_failed_yn=meta.get('ocr_failed_yn')
    )


@router.get("/files/{data_id}", response_model=FilesResponse)
def get_files(data_id: str, x_internal_token: Optional[str] = Header(None)):
    """
    변환본 메타 정보 조회 API
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    db = DBConnector()
    meta = db.get_file_by_id(data_id)
    
    if not meta:
        raise HTTPException(404, f"data_id {data_id} not found")
    
    # 변환 경로 계산
    converted_path = None
    if meta.get('file_folder') and meta.get('file_id'):
        converted_path = get_converted_path(meta['file_folder'], meta['file_id'])
    
    # simulated_yn 확인 (커스텀 컬럼)
    simulated_yn = None
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT simulated_yn FROM data_master WHERE data_id=?", (data_id,))
            row = cur.fetchone()
            if row:
                simulated_yn = row[0]
            cur.close()
    except Exception:
        pass
    
    return FilesResponse(
        data_id=data_id,
        converted_path=converted_path,
        minio_pdf_key=meta.get('minio_pdf_key'),
        minio_original_key=meta.get('minio_original_key'),
        simulated_yn=simulated_yn
    )


@router.get("/health")
def health_java():
    """헬스체크"""
    return {"status": "ok", "service": "java_router"}