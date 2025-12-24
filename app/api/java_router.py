# app/api/java_router.py
"""
Java 시스템 연동 라우터 (운영용)
- 서버 파일시스템 사용
- DB 완전 연동 (osk_data, osk_ocr_data, osk_ocr_hist, osk_data_sc)
- convert-and-index: PDF 변환 + OCR + 청킹 + 임베딩
- manual-ocr-and-index: DB 기반 수동 OCR 청킹 + 임베딩
- sc-index: SC 문서 (preface + contents + conclusion) 단일 청크 처리
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
from app.services.pdf_converter import convert_to_pdf, convert_bytes_to_pdf_bytes, ConvertError
from app.services import job_state
from app.services.minio_store import MinIOStore
from datetime import timezone
import mimetypes

router = APIRouter(prefix="/java", tags=["java-production"])

# 환경변수
SHARED_SECRET = os.getenv("JAVA_SHARED_SECRET", "")
SERVER_BASE_PATH = os.getenv("SERVER_BASE_PATH", "/mnt/shared")
MILVUS_VARCHAR_MAX = int(os.getenv("MILVUS_VARCHAR_MAX", "8192"))


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


class SCIndexRequest(BaseModel):
    """자바 → AI 트리거 요청 (sc-index) - SC 문서 전용"""
    data_id: str
    callback_url: Optional[str] = None


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

class UpdateMetadataRequest(BaseModel):
    """자바 → AI 메타데이터 업데이트 요청"""
    data_id: str
    callback_url: str  # 완료 콜백 URL


class UpdateMetadataResponse(BaseModel):
    """메타데이터 업데이트 즉시 응답"""
    status: str
    job_id: str
    data_id: str
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
            print(f"[WEBHOOK] Sent to {url}: {payload.status}")
    except Exception as e:
        print(f"[WEBHOOK] Failed: {e}")


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

def _split_for_milvus(text: str, meta: dict) -> list[tuple[str, dict]]:
    max_bytes = MILVUS_VARCHAR_MAX

    b = (text or "").encode("utf-8")
    if len(b) <= max_bytes:
        return [(text, meta)]

    pieces = []
    step = max_bytes - 32  # 메타/후처리 여유

    start = 0
    idx = 0
    while start < len(b):
        chunk_b = b[start:start + step]

        # UTF-8 경계 깨짐 방지: 디코딩이 안되면 뒤에서 조금씩 줄임
        while True:
            try:
                sub = chunk_b.decode("utf-8")
                break
            except UnicodeDecodeError:
                chunk_b = chunk_b[:-1]
                if not chunk_b:
                    sub = ""
                    break

        m = dict(meta)
        m["split_index_for_milvus"] = idx
        pieces.append((sub, m))

        start += len(chunk_b) if chunk_b else step
        idx += 1

    return pieces


def _coerce_chunks_for_milvus(chs):
    """
    (텍스트, 메타) 리스트를 Milvus insert 형태로 정규화
    - page/section/bboxes만 유지
    - 최종적으로는 모든 text가 MILVUS_VARCHAR_MAX 이하가 되도록 강제 분할
    """
    safe: list[tuple[str, dict]] = []

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
                try:
                    page = int(meta.get("page", 0))
                except Exception:
                    page = 0
        else:
            try:
                page = int(meta.get("page", 0))
            except Exception:
                page = 0

        base_meta = {
            "page": page,
            "section": section,
            "pages": pages or [],
            "bboxes": meta.get("bboxes", {}),
        }

        # 여기서 한 번 더 길이 체크 + 분할
        #    (각 청커 구현이 길이 제한을 안 지켜도 여기서 최종 방어)
        split_items = _split_for_milvus(text, base_meta)
        safe.extend(split_items)

    # 중복 제거
    out: list[tuple[str, dict]] = []
    last: tuple[str, dict] | None = None
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


def _render_text_pdf(text: str, out_path: str) -> str:
    """
    주어진 text를 간단한 PDF로 렌더링해 out_path에 저장하고 경로를 반환.
    reportlab이 없으면 ImportError 발생 -> 상위에서 처리.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import os

    # ========== 한글 폰트 등록 ==========
    font_registered = False
    try:
        # 나눔고딕 폰트 경로
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("NanumGothic", font_path))
            font_registered = True
            print(f"[SC-PDF] Font registered: {font_path}")
        else:
            print(f"[SC-PDF] WARNING: Font not found at {font_path}")
            
    except Exception as e:
        print(f"[SC-PDF] Font registration error: {e}")

    # PDF 캔버스 생성
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4

    margin_x = 20 * mm
    margin_y = 20 * mm
    max_width = width - 2 * margin_x
    y = height - margin_y

    # ========== 폰트 설정 ==========
    if font_registered:
        c.setFont("NanumGothic", 10)
    else:
        c.setFont("Helvetica", 10)
        print("[SC-PDF] WARNING: Using Helvetica (Korean may not display)")

    # ========== 텍스트 줄바꿈 처리 ==========
    import textwrap
    lines = []
    
    for para in (text or "").splitlines():
        if not para.strip():
            lines.append("")  # 빈 줄 유지
            continue
        
        # 한글 문자 비율 계산
        korean_chars = sum(1 for c in para if ord(c) > 127)
        total_chars = len(para)
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
        
        # 한글이 많으면 줄바꿈 폭 줄이기
        if korean_ratio > 0.3:  # 30% 이상 한글
            wrap_width = 50  # 한글 기준
        else:
            wrap_width = 95  # 영문 기준
        
        wrapped = textwrap.wrap(para, width=wrap_width) or [""]
        lines.extend(wrapped)

    line_height = 14  # pt (한글 가독성 고려)
    
    for line in lines:
        # 페이지 넘김 체크
        if y <= margin_y + line_height:
            c.showPage()
            if font_registered:
                c.setFont("NanumGothic", 10)
            else:
                c.setFont("Helvetica", 10)
            y = height - margin_y
        
        try:
            c.drawString(margin_x, y, line)
        except Exception as e:
            # 특수 문자 에러 방지
            print(f"[SC-PDF] WARNING: Error drawing line, using fallback: {e}")
            safe_line = line.encode('utf-8', errors='ignore').decode('utf-8')
            try:
                c.drawString(margin_x, y, safe_line)
            except:
                # 그래도 안 되면 스킵
                print(f"[SC-PDF] ERROR: Cannot draw line: {safe_line[:50]}...")
        
        y -= line_height

    c.showPage()
    c.save()
    
    print(f"[SC-PDF] PDF created successfully: {out_path}")
    return out_path

def _clean_html_and_split_paragraphs(html_text: str) -> List[str]:
    """
    HTML 텍스트를 정리하고 <p><br></p> 단위로 단락 분리
    
    사용자가 엔터로 구분한 단락을 유지하기 위함.
    """
    from html import unescape
    import re
    
    # HTML 엔티티 디코딩
    text = unescape(html_text)
    
    # 빈 <p> 태그를 단락 구분자로 변경
    # <p><br></p>, <p><br/></p>, <p> </p>, <p></p> 등
    # 연속된 빈 태그도 하나의 구분자로 처리
    empty_p_pattern = r'(<p[^>]*>\s*(?:<br\s*/?>)?\s*</p>\s*)+'
    text = re.sub(empty_p_pattern, '\n\n__PARAGRAPH_BREAK__\n\n', text, flags=re.IGNORECASE)
    
    # 나머지 <p> 태그 제거 (내용은 유지)
    text = re.sub(r'</?p[^>]*>', '', text, flags=re.IGNORECASE)
    
    # 모든 HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 단락 분리
    paragraphs = text.split('__PARAGRAPH_BREAK__')
    
    cleaned_paragraphs = []
    for para in paragraphs:
        # 연속된 공백을 하나로
        para = re.sub(r'\s+', ' ', para)
        para = para.strip()
        
        if para:
            cleaned_paragraphs.append(para)
    
    return cleaned_paragraphs

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


@router.post("/sc-index", response_model=ConvertAndIndexResponse)
async def sc_index(
    request: SCIndexRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: Optional[str] = Header(None)
):
    """
    자바 → AI 트리거 API (운영용) - sc-index (SC 문서 전용)
    
    프로세스:
    1. osk_data_sc 테이블에서 data_id 조회
    2. preface_text + contents_text + conclusion_text 합치기
    3. 단일 청크로 처리 (1~2 페이지 분량)
    4. 임베딩 후 Milvus 저장
    5. osk_data.parse_yn = 'S', osk_ocr_hist 로깅
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    # SC 문서 존재 여부 확인
    db = DBConnector()
    sc_doc = db.get_sc_document(request.data_id)
    
    if not sc_doc:
        raise HTTPException(404, f"SC document not found: data_id={request.data_id}")
    
    job_id = str(uuid.uuid4())[:8]
    
    background_tasks.add_task(
        process_sc_index,
        job_id=job_id,
        data_id=request.data_id,
        callback_url=request.callback_url
    )
    
    return ConvertAndIndexResponse(
        status="accepted",
        job_id=job_id,
        data_id=request.data_id,
        message="processing SC document (single chunk)"
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

@router.post("/update-metadata", response_model=UpdateMetadataResponse)
async def update_metadata(
    request: UpdateMetadataRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: Optional[str] = Header(None)
):
    """
    자바 → AI 메타데이터 업데이트 API
    
    자바측에서 osk_data 테이블의 메타데이터(data_title, data_code 등)를 수정한 후 호출
    
    프로세스:
    1. data_id로 osk_data 테이블 조회
    2. MinIO meta.json 업데이트
    3. Milvus 청크 메타데이터 업데이트 (청크별 metadata 필드)
    4. 완료 후 callback_url로 웹훅 전송
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(401, "Unauthorized - Invalid token")
    
    # data_id 존재 확인
    db = DBConnector()
    row = db.get_file_by_id(request.data_id)
    
    if not row:
        raise HTTPException(404, f"data_id not found: {request.data_id}")
    
    job_id = str(uuid.uuid4())[:8]
    
    background_tasks.add_task(
        process_update_metadata,
        job_id=job_id,
        data_id=request.data_id,
        callback_url=request.callback_url
    )
    
    return UpdateMetadataResponse(
        status="accepted",
        job_id=job_id,
        data_id=request.data_id,
        message="metadata update processing"
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
        "manual_ocr": "DB-based (osk_ocr_data)",
        "sc_index": "SC document single chunk (preface + contents + conclusion)",
        "metadata_update": "MinIO + Milvus metadata sync" 
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
    
    1. PDF 외 확장자 → PDF 변환 (bytes 기반)
    2. 변환된 PDF를 MinIO에 업로드
    3. DB에는 경로를 쓰지 않고 상태만 업데이트
    """
    from app.services.file_parser import parse_pdf, parse_pdf_blocks
    
    db = DBConnector()
    m = MinIOStore()
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
        
        # ========== Step 2: PDF 변환 (필요 시) + MinIO 업로드 + (변환 시) 볼륨 저장 + DB file_id만 변경 ==========
        src_ext = full_path.suffix.lower()
        is_already_pdf = (src_ext == ".pdf")
        
        doc_id = str(data_id)
        object_pdf = f"uploaded/{doc_id}.pdf"  # MinIO 업로드 키
        
        converted_pdf_path: Optional[str] = None
        pdf_bytes: Optional[bytes] = None
        
        if is_already_pdf:
            # (A) 이미 PDF면: 파일 그대로 사용 + MinIO 업로드 (DB file_id 변경 안 함)
            converted_pdf_path = str(full_path)
            print(f"[PROD] Already PDF: {converted_pdf_path}")
        
            with open(full_path, "rb") as f:
                pdf_bytes = f.read()
        
            m.upload_bytes(
                pdf_bytes,
                object_name=object_pdf,
                content_type="application/pdf",
                length=len(pdf_bytes),
            )
            print(f"[PROD] PDF uploaded to MinIO: {object_pdf}")
        
        else:
            # (B) PDF가 아니면: 변환 → MinIO 업로드 → 동일 폴더에 *.pdf 저장 → DB에는 file_id만 *.pdf로 변경
            job_state.update(job_id, status="converting", step=f"Converting {src_ext} to PDF")
            print(f"[PROD] Converting {src_ext} to PDF: {full_path}")

            try:
                # 1) bytes 변환 우선 시도
                with open(full_path, "rb") as f:
                    content = f.read()

                pdf_bytes = convert_bytes_to_pdf_bytes(content, src_ext)

                # 2) bytes 변환 실패 시 로컬 경로 기반 변환기로 폴백
                if pdf_bytes is None:
                    print(f"[PROD] ⚠️ Bytes conversion failed, trying local file converter...")
                    temp_pdf_path = convert_to_pdf(str(full_path))

                    if not temp_pdf_path or not Path(temp_pdf_path).exists():
                        raise ConvertError(f"PDF 변환 실패(출력 없음): {src_ext}")

                    # temp_pdf_path를 converted_pdf_path로 사용
                    converted_pdf_path = temp_pdf_path
                    print(f"[PROD] Local converter success: {converted_pdf_path}")

                    with open(converted_pdf_path, "rb") as f:
                        pdf_bytes = f.read()

                assert pdf_bytes is not None and len(pdf_bytes) > 0, "pdf_bytes is None or empty"
                print(f"[PROD] PDF converted: {len(pdf_bytes)} bytes")

                # 3) MinIO 업로드
                m.upload_bytes(
                    pdf_bytes,
                    object_name=object_pdf,
                    content_type="application/pdf",
                    length=len(pdf_bytes),
                )
                print(f"[PROD] PDF uploaded to MinIO: {object_pdf}")

                # 4) bytes 변환 성공 시에만 파일 저장 (로컬 변환은 이미 저장됨)
                if converted_pdf_path is None:
                    save_path = full_path.with_suffix(".pdf")
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, "wb") as fw:
                        fw.write(pdf_bytes)
                    converted_pdf_path = str(save_path)
                    print(f"[PROD] PDF saved to volume: {converted_pdf_path}")

                # 5) DB에는 file_id만 *.pdf로 업데이트 (폴더는 건드리지 않음)
                # 핵심 수정: converted_pdf_path에서 직접 추출
                new_file_id_pdf = Path(converted_pdf_path).name  # ex) f20231212M3Uv.pdf
                db.update_file_id_only(data_id, new_file_id_pdf)

            except Exception as e:
                error_msg = f"PDF 변환 실패 ({src_ext}): {e}"
                print(f"[PROD] {error_msg}")
                raise ConvertError(error_msg)

        # converted_pdf_path가 설정되지 않았다면 문제 발생
        if not converted_pdf_path:
            raise RuntimeError("converted_pdf_path가 설정되지 않음")

        if not Path(converted_pdf_path).exists():
            raise RuntimeError(f"converted_pdf_path가 존재하지 않음: {converted_pdf_path}")

        print(f"[PROD] Final PDF path for parsing: {converted_pdf_path}")          
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
        print(f"[PROD-OCR-DB] OCR completed and saved to DB: {len(pages)} pages")
        
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
        
        print(f"[PROD] Successfully indexed: {result.get('inserted', 0)} chunks")
        
        # ========== Step 11: RAG 완료 처리 ==========
        pages_count = len(pages_std)
        chunk_count = result.get('inserted', len(chunks))
        
        # ========== Step 11.5: MinIO 동기화 (프론트 목록 노출용) ==========
        try:
            SYNC_TO_MINIO = os.getenv("JAVA_SYNC_TO_MINIO", "1") == "1"
            if SYNC_TO_MINIO:
                doc_id = str(data_id)
                pdf_path_for_upload = converted_pdf_path

                # DB에서 제목/코드 등 메타 읽기
                row = None
                try:
                    row = db.get_file_by_id(data_id)  # { data_title, data_code, ... }
                except Exception as _e:
                    row = None

                # 표시용 타이틀 결정: DB data_title 우선, 없으면 파일명
                display_title = None
                if isinstance(row, dict):
                    display_title = (row.get("data_title") or "").strip() or None
                if not display_title:
                    display_title = Path(pdf_path_for_upload).name  # fallback

                if pdf_path_for_upload and Path(pdf_path_for_upload).exists():
                    # 이미 MinIO에 업로드했으므로 중복 업로드 불필요
                    # meta.json만 업데이트
                    object_pdf = f"uploaded/{doc_id}.pdf"

                    # meta.json 갱신(존재하면 merge)
                    meta = {}
                    try:
                        if m.exists(META_KEY(doc_id)):
                            meta = m.get_json(META_KEY(doc_id)) or {}
                    except Exception:
                        meta = {}

                    # DB 메타를 함께 저장(필터에 쓰고 싶으면 프론트에서 활용 가능)
                    extra_meta = {}
                    if isinstance(row, dict):
                        for k in [
                            "data_id","data_title","data_code","data_code_detail","data_code_detail_sub",
                            "file_folder","file_id","reg_nm","reg_id","reg_dt","reg_type","parse_yn"
                        ]:
                            if k in row:
                                extra_meta[k] = row[k]

                    meta.update({
                        "doc_id": doc_id,
                        "title": display_title,                 # DB data_title
                        "pdf_key": object_pdf,                  # MinIO 변환 PDF
                        "original_key": None,                   # MinIO 오브젝트가 아니면 None
                        "original_fs_path": str(full_path),     # 로컬 경로는 별도 필드에
                        "original_name": Path(full_path).name,
                        "is_pdf_original": is_already_pdf,
                        "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "indexed": True,
                        "chunk_count": int(chunk_count),
                        "last_indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        **extra_meta,
                    })
                    m.put_json(META_KEY(doc_id), meta)

                    print(f"[PROD-MINIO] synced: {object_pdf} (title='{display_title}', chunks={chunk_count})")
                else:
                    print("[PROD-MINIO] skip: no local pdf to upload")
            else:
                print("[PROD-MINIO] ⏭skip: JAVA_SYNC_TO_MINIO=0")
        except Exception as e:
            print(f"[PROD-MINIO] sync failed: {e}")

        print(f"[PROD] Indexing completed: {pages_count} pages, {chunk_count} chunks")
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
                status="api_indexed",
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
        print(f"[PROD] Error: {e}")
        
        if callback_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="api_error",
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
       - 완료 시 parse_yn = 'S', osk_ocr_hist 로깅
    
    2. rag_yn='Y': 기존 작업 수정 (사용자가 페이지 수정)
       - osk_ocr_data에서 수정된 텍스트 가져와서 재청킹/임베딩
       - Milvus에서 기존 청크 삭제 후 새로 삽입
       - 완료 시 parse_yn = 'S', osk_ocr_hist 로깅
    """
    db = DBConnector()
    start_time = datetime.utcnow()
    
    job_state.start(job_id, data_id=data_id, file_id=file_id)
    rag_yn = (rag_yn or "N").upper()
    
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
        
        # 기존 문서 삭제
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
        
        print(f"[MANUAL-OCR] Successfully indexed: {result.get('inserted', 0)} chunks")
        
        # ========== Step 5: RAG 완료 처리 ==========
        pages_count = len(pages_std)
        chunk_count = result.get('inserted', len(chunks))
        
        print(f"[MANUAL-OCR] Indexing completed: {pages_count} pages, {chunk_count} chunks")
        # RAG 완료 마킹 (parse_yn='S' 유지, 히스토리 로깅)
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
                status="corrected_indexed",
                converted=False,
                metrics={"pages": pages_count, "chunks": chunk_count},
                chunk_count=chunk_count,
                timestamps={
                    "start": start_time.isoformat(), 
                    "end": end_time.isoformat()
                },
                message="indexed successfully from manual OCR (advanced chunking)"
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        db.update_rag_error(data_id, str(e))
        print(f"[MANUAL-OCR] Error: {e}")
        
        if callback_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="corrected_error",
                message=str(e)
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
        
        raise


async def process_sc_index(
    job_id: str,
    data_id: str,
    callback_url: Optional[str]
):
    """
    운영용 백그라운드 처리 - sc-index (SC 문서 전용)
    
    1. osk_data_sc 테이블에서 구조화된 데이터 조회
    2. 헤더(문서번호 + 수신처 + 제목 + 발신일) 생성
    3. 본문, 맺음말 포함 전체 텍스트 조합
    4. 각 청크에 SC 메타데이터 첨부
    5. 토큰 수에 따라 청크 분할 (필요시)
    6. 임베딩 및 Milvus 저장
    7. osk_data.parse_yn = 'S', osk_ocr_hist 로깅
    개선: HTML 태그 제거 + <br> 단위 단락 청킹
    """
    db = DBConnector()
    m = MinIOStore()
    start_time = datetime.utcnow()
    
    job_state.start(job_id, data_id=data_id, file_id="sc_document")
    
    # ========== 환경변수 백업 및 오버라이드 ==========
    old_dedup = os.environ.get("RAG_DEDUP_MANIFEST")
    old_skip = os.environ.get("RAG_SKIP_IF_EXISTS")
    old_replace = os.environ.get("RAG_REPLACE_DOC")
    
    try:
        # SC 인덱싱은 항상 강제 재삽입 (환경변수 임시 오버라이드)
        os.environ["RAG_DEDUP_MANIFEST"] = "0"
        os.environ["RAG_SKIP_IF_EXISTS"] = "0"
        os.environ["RAG_REPLACE_DOC"] = "0"
        print(f"[SC-INDEX] Environment override: DEDUP=0, SKIP=0, REPLACE=0")
        
        # ========== Step 1: OCR 시작 마킹 ==========
        db.mark_ocr_start(data_id)
        print(f"[SC-INDEX] 신규 SC 문서 작업: data_id={data_id}, parse_yn='L'")
        
        # ========== Step 2: DB에서 SC 문서 구조화 데이터 가져오기 ==========
        job_state.update(job_id, status="loading", step="Loading structured SC document from DB")
        
        print(f"[SC-INDEX] Loading structured SC document from osk_data_sc for data_id={data_id}")
        
        sc_data = db.get_sc_document_with_structure(data_id)
        
        if not sc_data or not sc_data.get('full_text'):
            raise RuntimeError(f"SC 문서 텍스트가 비어있습니다: data_id={data_id}")
        
        metadata = sc_data['metadata']
        header = sc_data['header']
        contents_raw = sc_data['contents']
        conclusion = sc_data['conclusion']
        
        print(f"[SC-INDEX] Loaded SC document:")
        print(f"     문서번호: {metadata['sc_code']}")
        print(f"     수신처: {metadata['receiver_agency']}")
        print(f"     제목: {metadata['sc_title']}")
        print(f"     발신일: {metadata['send_date']}")
        print(f"     Header: {len(header)} characters")
        print(f"     Contents (raw): {len(contents_raw)} characters")
        print(f"     Conclusion: {len(conclusion)} characters")
        
        print(f"[SC-INDEX] Cleaning HTML and splitting paragraphs...")
        paragraphs = _clean_html_and_split_paragraphs(contents_raw)
        print(f"[SC-INDEX] Extracted {len(paragraphs)} paragraphs from contents")
        if len(paragraphs) > 0:
            print(f"[SC-INDEX] First paragraph preview: {paragraphs[0][:100]}...")
        full_text_clean = header  # 헤더는 그대로 사용
        if paragraphs:
            full_text_clean += "\n\n" + "\n\n".join(paragraphs)
        if conclusion:
            full_text_clean += "\n\n" + conclusion
        
        print(f"[SC-INDEX] Full text (cleaned): {len(full_text_clean)} characters")

        # ========== Step 3: 단락 기반 청크 생성 ==========
        job_state.update(job_id, status="chunking", step="Creating chunks for SC document")
        
        from app.services.embedding_model import get_embedding_model
        
        embedding_model = get_embedding_model()
        tokenizer = getattr(embedding_model, "tokenizer", None)
        
        max_seq_length = int(getattr(embedding_model, "max_seq_length", 1024))
        embed_max_tokens = int(os.getenv("EMBED_MAX_TOKENS", "1024"))
        # 둘 중 작은 값 사용 (안전 마진 20% 확보)
        safe_max_tokens = int(min(max_seq_length, embed_max_tokens) * 0.8)
        
        # Milvus VARCHAR 제한 (바이트 기준)
        MILVUS_VARCHAR_MAX = int(os.getenv("MILVUS_VARCHAR_MAX", "8192"))
        # UTF-8 한글은 3바이트, 안전 마진 포함
        SAFE_CHAR_LIMIT = (MILVUS_VARCHAR_MAX // 3) - 200  # 약 2500자
        
        print(f"[SC-INDEX] Max seq length: {max_seq_length}, EMBED_MAX_TOKENS: {embed_max_tokens}")
        print(f"[SC-INDEX] Safe max tokens per chunk: {safe_max_tokens}")
        print(f"[SC-INDEX] Safe char limit: {SAFE_CHAR_LIMIT} (for Milvus VARCHAR)")

        # ========== Step 3.1: SC 메타데이터 준비 ==========
        # 모든 청크에 공통으로 들어갈 SC 메타데이터
        sc_metadata = {
            "sc_code": metadata['sc_code'] or "",              # 문서번호
            "receiver_agency": metadata['receiver_agency'] or "",  # 수신처
            "sc_title": metadata['sc_title'] or "",            # 제목
            "send_date": metadata['send_date'] or "",          # 발신일
            "sender_file_id": metadata['sender_file_id'] or "",  # 발신 파일 ID
        }
        
        chunks = []
        
        # ========== Step 3.2: 단락 기반 청크 생성 로직 ==========
        current_chunk_text = header  # 헤더로 시작
        current_token_count = 0
        
        if tokenizer:
            current_token_count = len(tokenizer.encode(current_chunk_text, add_special_tokens=False))
        else:
            current_token_count = len(current_chunk_text) // 4
        
        chunk_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            # 단락의 토큰 수 계산
            if tokenizer:
                para_tokens = len(tokenizer.encode(paragraph, add_special_tokens=False))
            else:
                para_tokens = len(paragraph) // 4
            
            # 현재 청크에 단락을 추가했을 때 제한을 넘는지 확인
            would_exceed_tokens = (current_token_count + para_tokens) > safe_max_tokens
            would_exceed_chars = (len(current_chunk_text) + len(paragraph) + 2) > SAFE_CHAR_LIMIT
            
            if would_exceed_tokens or would_exceed_chars:
                # 현재 청크 저장
                if current_chunk_text.strip():
                    chunk_metadata = {
                        "page": chunk_index + 1,
                        "pages": [chunk_index + 1],
                        "section": f"SC Document (Part {chunk_index + 1})" if chunk_index > 0 else "SC Document",
                        "token_count": current_token_count,
                        "bboxes": {},
                        "type": "sc_document_part" if chunk_index > 0 else "sc_document",
                        **sc_metadata
                    }
                    
                    chunks.append((current_chunk_text, chunk_metadata))
                    print(f"[SC-INDEX] Created chunk {chunk_index + 1}: {current_token_count} tokens, {len(current_chunk_text)} chars")
                    chunk_index += 1
                
                # 새 청크 시작 (헤더 포함)
                current_chunk_text = f"[계속]\n{header}\n\n{paragraph}"
                if tokenizer:
                    header_tokens = len(tokenizer.encode(header, add_special_tokens=False))
                    current_token_count = para_tokens + header_tokens + 10  # [계속]\n 부분
                else:
                    current_token_count = (len(header) + len(paragraph)) // 4
            else:
                # 현재 청크에 단락 추가
                current_chunk_text += f"\n\n{paragraph}"
                current_token_count += para_tokens
        
        # 마지막 청크 저장
        if current_chunk_text.strip():
            # 결론 추가
            if conclusion:
                current_chunk_text += f"\n\n{conclusion}"
            
            if tokenizer:
                current_token_count = len(tokenizer.encode(current_chunk_text, add_special_tokens=False))
            else:
                current_token_count = len(current_chunk_text) // 4
            
            chunk_metadata = {
                "page": chunk_index + 1,
                "pages": [chunk_index + 1],
                "section": f"SC Document (Part {chunk_index + 1})" if chunk_index > 0 else "SC Document",
                "token_count": current_token_count,
                "bboxes": {},
                "type": "sc_document_part" if chunk_index > 0 else "sc_document",
                **sc_metadata
            }
            
            chunks.append((current_chunk_text, chunk_metadata))
            print(f"[SC-INDEX] Created final chunk {chunk_index + 1}: {current_token_count} tokens, {len(current_chunk_text)} chars")
        
        print(f"[SC-INDEX] Total chunks created: {len(chunks)} (paragraph-based)")
            
        # ========== Step 4: 청크 정규화 ==========
        chunks = _coerce_chunks_for_milvus(chunks)
        print(f"[SC-INDEX] Normalized {len(chunks)} chunk(s) for Milvus")
        
        for i, (text, meta) in enumerate(chunks):
            byte_len = len(text.encode('utf-8', errors='ignore'))
            if byte_len > MILVUS_VARCHAR_MAX:
                print(f"[SC-INDEX] ERROR: Chunk {i} exceeds byte limit: {byte_len} > {MILVUS_VARCHAR_MAX}")
                # 강제 자르기
                while byte_len > MILVUS_VARCHAR_MAX and text:
                    text = text[:int(len(text) * 0.9)]
                    byte_len = len(text.encode('utf-8', errors='ignore'))
                chunks[i] = (text, meta)
                print(f"[SC-INDEX] Truncated chunk {i} to {byte_len} bytes")

        # ========== Step 5: 임베딩 및 Milvus 저장 ==========
        job_state.update(job_id, status="embedding", step=f"Embedding {len(chunks)} chunk(s)")
        
        from app.services.embedding_model import embed, get_sentence_embedding_dimension
        from app.services.milvus_store_v2 import MilvusStoreV2
        
        mvs = MilvusStoreV2(dim=get_sentence_embedding_dimension())
        
        # 기존 문서 삭제
        print(f"[SC-INDEX] Deleting existing doc (if any): {data_id}")
        try:
            deleted = mvs._delete_by_doc_id(data_id)
            print(f"[SC-INDEX] Deleted {deleted} existing chunks")
        except Exception as e:
            print(f"[SC-INDEX] Warning during delete: {e}")
        
        # Milvus insert
        print(f"[SC-INDEX] Inserting {len(chunks)} chunk(s) to Milvus")
        
        result = mvs.insert(
            doc_id=data_id,
            chunks=chunks,
            embed_fn=embed
        )
        
        inserted_count = result.get('inserted', 0)
        print(f"[SC-INDEX] Milvus insert result: {result}")
        print(f"[SC-INDEX] Successfully indexed: {inserted_count} chunk(s)")
        
        if inserted_count == 0 and len(chunks) > 0:
            error_msg = f"Milvus 삽입 실패: {result.get('reason', 'unknown')}"
            print(f"[SC-INDEX] ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        # ========== Step 6: MinIO sync (SC) ==========
        try:
            SYNC_TO_MINIO = os.getenv("JAVA_SYNC_TO_MINIO", "1") == "1"
            if SYNC_TO_MINIO:
                doc_id = str(data_id)

                # 표시용 타이틀: SC 제목 우선, 없으면 osk_data.data_title
                display_title = metadata['sc_title'] or ""
                
                if not display_title:
                    try:
                        row = db.get_file_by_id(data_id)
                        if isinstance(row, dict):
                            display_title = (row.get("data_title") or "").strip()
                    except Exception:
                        pass
                
                if not display_title:
                    display_title = f"SC Document {doc_id}"

                # SC 텍스트를 PDF로 생성 (임시 경로)
                import tempfile
                from pathlib import Path as _Path
                tmpdir = tempfile.gettempdir()
                local_pdf = _Path(tmpdir) / f"sc_{doc_id}.pdf"

                try:
                    _render_text_pdf(full_text_clean, str(local_pdf))
                except ImportError as e:
                    # reportlab 미설치 시 안내 로그
                    print(f"[SC-MINIO] reportlab not installed, skipping PDF generation")
                    local_pdf = None
                except Exception as e:
                    print(f"[SC-MINIO] PDF generation failed: {e}")
                    local_pdf = None

                # MinIO 업로드 (PDF가 생성된 경우만)
                object_pdf = None 
                if local_pdf and local_pdf.exists():
                    object_pdf = f"uploaded/sc/{doc_id}.pdf"
                    
                    # upload_bytes 메서드 사용
                    with open(local_pdf, "rb") as f:
                        pdf_bytes = f.read()
                        m.upload_bytes(
                            data=pdf_bytes,
                            object_name=object_pdf,
                            content_type="application/pdf",
                            length=len(pdf_bytes)
                        )

                    print(f"[SC-MINIO] Uploaded SC PDF: {object_pdf}")
                    
                    # 임시 파일 삭제
                    try:
                        local_pdf.unlink()
                    except:
                        pass

                # meta.json 생성/업데이트
                extra_meta = {
                    "sc_code": metadata['sc_code'],
                    "receiver_agency": metadata['receiver_agency'],
                    "sc_title": metadata['sc_title'],
                    "send_date": metadata['send_date'],
                    "type": "sc",
                    "source_type": "sc_document",
                    "paragraph_count": len(paragraphs),
                }
                
                # get_json도 예외 처리
                meta = {}
                try:
                    if m.exists(META_KEY(doc_id)):
                        meta = m.get_json(META_KEY(doc_id)) or {}
                except Exception as e:
                    print(f"[SC-MINIO] Failed to load existing meta.json: {e}")
                    meta = {}
                
                meta.update({
                    "doc_id": doc_id,
                    "title": display_title,
                    "pdf_key": object_pdf,
                    "object_key": object_pdf,  # backward compat
                    "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "indexed": True,
                    "chunk_count": int(inserted_count),
                    "last_indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "type": "sc_document",
                    **extra_meta,
                })
                m.put_json(META_KEY(doc_id), meta)

                print(f"[SC-MINIO] synced meta.json: title='{display_title}', pdf_key='{object_pdf}', chunks={inserted_count}")
                
            else:
                print("[SC-MINIO] skip: JAVA_SYNC_TO_MINIO=0")
        except Exception as e:
            import traceback
            print(f"[SC-MINIO] sync failed: {e}")
            traceback.print_exc()

        # ========== Step 7: OCR 성공 마킹 ==========
        # SC 문서는 페이지 개념이 없으므로 osk_ocr_data에 저장하지 않음
        # 바로 OCR 성공 처리
        db.mark_ocr_success(data_id)
        print(f"[SC-INDEX] Marked OCR success for SC document: data_id={data_id}")
        
        # ========== Step 8: RAG 완료 처리 ==========
        chunk_count = inserted_count
        
        print(f"[SC-INDEX] Indexing completed: 1 SC document, {chunk_count} chunk(s)")
        # RAG 완료 마킹 (parse_yn='S' 유지, 히스토리 로깅)
        db.update_rag_completed(data_id)
        
        job_state.complete(
            job_id,
            pages=1,
            chunks=chunk_count
        )
        
        # ========== Step 9: 완료 & Webhook ==========
        end_time = datetime.utcnow()
        
        if callback_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="sc_indexed",
                converted=False,
                metrics={
                    "pages": 1,
                    "chunks": chunk_count,
                    "sc_metadata": sc_metadata
                },
                chunk_count=chunk_count,
                timestamps={
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                message="SC document indexed successfully with structured metadata"
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        db.update_rag_error(data_id, str(e))
        print(f"[SC-INDEX] Error: {e}")
        import traceback
        traceback.print_exc()
        
        if callback_url:
            payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="sc_error",
                message=str(e)
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
        
        raise
    
    finally:
        if old_dedup is not None:
            os.environ["RAG_DEDUP_MANIFEST"] = old_dedup
        else:
            os.environ.pop("RAG_DEDUP_MANIFEST", None)
        
        if old_skip is not None:
            os.environ["RAG_SKIP_IF_EXISTS"] = old_skip
        else:
            os.environ.pop("RAG_SKIP_IF_EXISTS", None)
            
        if old_replace is not None:
            os.environ["RAG_REPLACE_DOC"] = old_replace
        else:
            os.environ.pop("RAG_REPLACE_DOC", None)
        
        print(f"[SC-INDEX] Environment variables restored")


async def process_delete_document(
    data_id: str,
    delete_from_minio: bool,
    callback_url: Optional[str]
):
    """문서 삭제 백그라운드 처리"""
    from app.services.milvus_store_v2 import MilvusStoreV2
    from app.services.embedding_model import get_sentence_embedding_dimension
    
    try:
        # Milvus 삭제
        mvs = MilvusStoreV2(dim=get_sentence_embedding_dimension())
        deleted_chunks = mvs._delete_by_doc_id(data_id)
        print(f"[DELETE] Deleted {deleted_chunks} chunks from Milvus for data_id={data_id}")
        
        # MinIO 삭제 (선택)
        deleted_files = []
        if delete_from_minio:
            m = MinIOStore()
            doc_id = str(data_id)
            
            # PDF 삭제
            pdf_key = f"uploaded/{doc_id}.pdf"
            if m.exists(pdf_key):
                m.delete(pdf_key)
                deleted_files.append(pdf_key)
            # SC PDF 삭제
            sc_pdf_key = f"uploaded/sc/{doc_id}.pdf"
            if m.exists(sc_pdf_key):
                m.delete(sc_pdf_key)
                deleted_files.append(sc_pdf_key)

            # meta.json 삭제
            meta_key = META_KEY(doc_id)
            if m.exists(meta_key):
                m.delete(meta_key)
                deleted_files.append(meta_key)
            
            print(f"[DELETE] Deleted {len(deleted_files)} files from MinIO")
        
        # Webhook
        if callback_url:
            payload = WebhookPayload(
                job_id=str(uuid.uuid4())[:8],
                data_id=data_id,
                status="deleted",
                converted=False,
                metrics={"deleted_chunks": deleted_chunks, "deleted_files": len(deleted_files)},
                message=f"Document deleted: {deleted_chunks} chunks, {len(deleted_files)} files"
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
    
    except Exception as e:
        print(f"[DELETE] Error: {e}")
        
        if callback_url:
            payload = WebhookPayload(
                job_id=str(uuid.uuid4())[:8],
                data_id=data_id,
                status="delete_error",
                message=str(e)
            )
            await send_webhook(callback_url, payload, SHARED_SECRET)
            
async def process_update_metadata(
    job_id: str,
    data_id: str,
    callback_url: Optional[str]
):
    """
    메타데이터 업데이트 백그라운드 처리
    
    빠른 응답 전략:
    1. DB 조회 성공 → 즉시 자바에게 성공 콜백 전송 (빠른 응답)
    2. MinIO meta.json 업데이트 (백그라운드)
    3. Milvus 청크 메타데이터 업데이트 (백그라운드)
    
    """
    db = DBConnector()
    start_time = datetime.utcnow()
    row = None
    job_state.start(job_id, data_id=data_id)
    
    try:
        # ========== Step 1: DB에서 최신 메타데이터 조회 (빠름) ==========
        job_state.update(job_id, status="loading", step="Loading metadata from DB")
        
        print(f"[META-UPDATE] Loading metadata for data_id={data_id}")
        
        row = db.get_file_by_id(data_id)
        
        if not row:
            # DB 조회 실패 시 즉시 에러 콜백
            if callback_url:
                error_payload = WebhookPayload(
                    job_id=job_id,
                    data_id=data_id,
                    status="metadata_update_error",
                    message=f"데이터를 찾을 수 없습니다: data_id={data_id}"
                )
                await send_webhook(callback_url, error_payload, SHARED_SECRET)
            
            raise RuntimeError(f"데이터를 찾을 수 없습니다: data_id={data_id}")
        
        print(f"[META-UPDATE] Loaded: data_title={row.get('data_title')}, data_code={row.get('data_code')}")
        
        # ========== Step 1.5: DB 조회 성공 → 즉시 자바에게 콜백 전송  ==========
        display_title = (row.get("data_title") or "").strip() or f"Document {data_id}"
        
        if callback_url:
            quick_response = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="metadata_loaded",
                converted=False,
                metrics={
                    "data_title": display_title,
                    "data_code": row.get("data_code"),
                    "data_code_detail": row.get("data_code_detail"),
                    "data_code_detail_sub": row.get("data_code_detail_sub")
                },
                timestamps={
                    "start": start_time.isoformat(),
                    "loaded_at": datetime.utcnow().isoformat()
                },
                message="metadata loaded from DB, updating storage in background"
            )
            await send_webhook(callback_url, quick_response, SHARED_SECRET)
            print(f"[META-UPDATE] Quick callback sent to Java")
        
        # ========== Step 2: MinIO meta.json 업데이트 (백그라운드) ==========
        job_state.update(job_id, status="updating_minio", step="Updating MinIO metadata")
        
        try:
            m = MinIOStore()
            doc_id = str(data_id)
            meta_key = META_KEY(doc_id)
            
            # 기존 meta.json 로드
            meta = {}
            if m.exists(meta_key):
                try:
                    meta = m.get_json(meta_key) or {}
                except Exception:
                    meta = {}
            
            # 업데이트할 필드들
            db_meta_fields = {}
            for k in [
                "data_id","data_title","data_code","data_code_detail","data_code_detail_sub",
                "file_folder","file_id","reg_nm","reg_id","reg_dt","reg_type","parse_yn"
            ]:
                if k in row:
                    db_meta_fields[k] = row[k]
            
            # 기존 메타 유지하면서 업데이트
            meta.update({
                "title": display_title,
                "last_metadata_updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                **db_meta_fields
            })
            
            # MinIO에 저장
            m.put_json(meta_key, meta)
            
            print(f"[META-UPDATE] MinIO meta.json updated: title='{display_title}'")
        
        except Exception as e:
            print(f"[META-UPDATE] MinIO update failed: {e}")

 
        
        # ========== Step 4: 완료 처리 ==========
        job_state.complete(
            job_id,
            pages=0,
            chunks=0
        )
        
        print(f"[META-UPDATE] Metadata update completed for data_id={data_id}")
    
    except Exception as e:
        job_state.fail(job_id, str(e))
        print(f"[META-UPDATE] Error: {e}")
        import traceback
        print(traceback.format_exc())
        
        # 에러 발생 시에만 에러 콜백 (DB 조회 실패는 위에서 이미 전송)
        if callback_url and not row:  # DB 조회 실패가 아닌 경우
            error_payload = WebhookPayload(
                job_id=job_id,
                data_id=data_id,
                status="metadata_update_error",
                message=str(e)
            )
            await send_webhook(callback_url, error_payload, SHARED_SECRET)
        
        raise