# app/api/llama_router.py
from __future__ import annotations

import mimetypes
import hashlib, tempfile
import os, re
import uuid
from urllib.parse import unquote, quote
from typing import List, Optional
from starlette.responses import StreamingResponse

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
from pydantic import BaseModel
import asyncio, json
from sse_starlette.sse import EventSourceResponse  # ✅ 요구사항: sse-starlette
from app.services import job_state, milvus_store
from datetime import datetime, timedelta, timezone

from app.services.file_parser import (
    parse_pdf,                    # (local path) -> [(page_no, text)]
    parse_pdf_blocks,             # (local path) -> [(page_no, [ {text,bbox}, ... ])]
    parse_any_bytes,              # (filename, bytes) -> {"kind":"pdf", "pages":[...], "blocks":[...]}
    parse_pdf_blocks_from_bytes,  # (bytes) -> [(page_no, [ {text,bbox}, ... ])]
)
from app.services.llama_model import generate_answer_unified
from app.services.minio_store import MinIOStore
from app.services.pdf_converter import convert_stream_to_pdf_bytes, convert_to_pdf,convert_bytes_to_pdf_bytes, ConvertError
from app.services.milvus_store_v2 import MilvusStoreV2
from app.services.embedding_model import get_embedding_model, embed
from app.services.reranker import rerank

router = APIRouter(tags=["llama"])

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# ---------- Schemas ----------
class GenerateReq(BaseModel):
    prompt: str
    model_name: str = "llama-3.2-3b"

class AskReq(BaseModel):
    question: str
    model_name: str = "llama-3.2-3b"
    top_k: int = 3

class UploadResp(BaseModel):
    filename: str
    minio_object: str
    indexed: str  # "background"
    job_id: Optional[str] = None

class AskResp(BaseModel):
    answer: str
    used_chunks: int
    sources: Optional[List[dict]] = None  # (선택) 출처 제공

# --- 폴백 전용: pages 정규화 도우미 ---------------------------------
def _normalize_pages_for_chunkers(pages):
    """
    pages를 [(page_no:int, text:str), ...] 로 강제 변환.
    허용 입력:
      - [(int, str)], [[int, str]]
      - ["page text", ...]  -> enumerate 1-based
      - [{"page":..,"text":..}], [{"page_no":..,"body":..}], [{"index":..,"lines":[..]}]
    그 외는 문자열화해서 안전하게 수용.
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

# ---------- Helpers ----------
def _coerce_chunks_for_milvus(chs):
    """
    (텍스트, 메타) 리스트를 Milvus insert 형태로 정규화:
    - 메타 타입 보정(dict 강제), page=int, section<=512자
    - 다중 페이지 지원: meta.pages가 있으면 page는 첫 페이지로
    - 빈 텍스트/연속 중복 제거
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

def _make_encoder():
    m = get_embedding_model()
    tok = getattr(m, "tokenizer", None)
    max_len = int(getattr(m, "max_seq_length", 128))
    def enc(s: str):
        if tok is None:
            return []
        return tok.encode(s, add_special_tokens=False) or []
    return enc, max_len

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def meta_key(doc_id: str) -> str:
    # 폴더형 메타(권장)
    return f"uploaded/__meta__/{doc_id}/meta.json"

def legacy_meta_key(doc_id: str) -> str:
    # 네가 기존에 쓰던 단일 파일형도 읽기 폴백
    return f"uploaded/__meta__/{doc_id}.json"

def index_pdf_to_milvus(
    job_id: str,
    file_path: str | None = None,
    minio_object: str | None = None,
    uploaded: bool = True,
    remove_local: bool = True,
    doc_id: str | None = None,
) -> None:
    """
    업로드된(또는 MinIO 상의) PDF를 파싱 → 청킹 → 임베딩 → Milvus upsert.
    - bytes 경로(minio) 우선 사용 (RAG_NO_LOCAL=1 이거나 file_path가 None)
    - 페이지 텍스트가 빈약하면 bytes-OCR 폴백 수행(ENV: OCR_MODE != off)
    - 레이아웃 청킹 실패 시 smart/기본 청킹 순차 폴백
    - 그래도 0청크면 최후 보호막: 통짜 텍스트로 1개 이상 청크 생성
    """
    try:
        job_state.update(job_id, status="parsing", step="parse_pdf:start")
        print(f"[INDEX] start: {file_path or minio_object}")

        NO_LOCAL = os.getenv("RAG_NO_LOCAL", "0") == "1"
        SKIP_IF_ALREADY_UPLOADED = os.getenv("RAG_SKIP_IF_UPLOADED", "1") == "1"

        if not uploaded and SKIP_IF_ALREADY_UPLOADED:
            job_state.update(job_id, status="done", step="skipped:already_uploaded", progress=100)
            print(f"[INDEX] skip: uploaded=False (already uploaded), job_id={job_id}")
            return

        # ---------- 1) PDF → 텍스트/레이아웃 ----------
        pages: list | None = None          # 페이지 텍스트 (문자열/튜플 혼재 가능 → 아래서 표준화)
        layout_map: dict[int, list[dict]] = {}  # {page_no: [ {text, bbox}, ... ]}
        pdf_bytes: bytes | None = None

        use_bytes_path = (NO_LOCAL or file_path is None) and bool(minio_object)
        if use_bytes_path:
            # MinIO → bytes
            from app.services.minio_store import MinIOStore
            mstore = MinIOStore()
            pdf_bytes = mstore.get_bytes(minio_object)

            # bytes 파서
            try:
                from app.services.file_parser import parse_any_bytes, parse_pdf_blocks_from_bytes
                parsed = parse_any_bytes(os.path.basename(minio_object), pdf_bytes)
                if parsed.get("kind") != "pdf":
                    raise RuntimeError("PDF 파이프라인만 인덱싱합니다. (변환 단계 확인)")

                pages = parsed.get("pages") or []  # 리스트(문자열 리스트 또는 혼재)
                blocks_by_page_list = parsed.get("blocks")

                # blocks 표준화 → layout_map
                if isinstance(blocks_by_page_list, dict):
                    layout_map = {int(k): v for k, v in blocks_by_page_list.items()}
                else:
                    layout_map = {int(p): blks for p, blks in (blocks_by_page_list or [])}
            except Exception as ee:
                raise RuntimeError(f"bytes parsing unavailable or failed: {ee}") from ee
        else:
            # 로컬 경로 파서
            from app.services.file_parser import parse_pdf, parse_pdf_blocks
            pages = parse_pdf(file_path, by_page=True)
            if not pages:
                raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다.")
            blocks_by_page_list = parse_pdf_blocks(file_path)
            layout_map = {int(p): blks for p, blks in (blocks_by_page_list or [])}

        # ---------- 1-1) bytes-OCR 폴백 (페이지 텍스트 밀도 낮을 때) ----------
        def _count_chars(pages_like) -> int:
            total = 0
            for it in (pages_like or []):
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    total += len(str(it[1] or ""))
                elif isinstance(it, dict):
                    total += len(str(it.get("text", "")))
                elif isinstance(it, str):
                    total += len(it)
            return total

        # OCR 활성 조건
        OCR_BYTES_ENABLED = (os.getenv("OCR_MODE", "auto").lower() != "off")
        MIN_TOTAL = int(os.getenv("OCR_MIN_CHARS_TOTAL", "120"))

        if OCR_BYTES_ENABLED and use_bytes_path:
            total_chars = _count_chars(pages)
            if total_chars < MIN_TOTAL:
                try:
                    # 지연 임포트(이 파일 상단에 import 안 되어 있으면 NameError 방지)
                    from app.services.ocr_service import try_ocr_pdf_bytes
                    ocr_text = try_ocr_pdf_bytes(pdf_bytes, enabled=True)  # returns str|None
                except Exception as _e:
                    print(f"[OCR] bytes fallback failed: {getattr(_e, 'message', _e)}")
                    ocr_text = None
                if ocr_text:
                    print(f"[OCR] bytes fallback used, chars={len(ocr_text)}")
                    pages = [(1, ocr_text)]        # 단일 페이지 취급
                    layout_map = {}                # OCR 텍스트엔 bbox 신뢰 불가 → 비움

        # ---------- 1-2) 페이지 표준화 ----------
        pages_std = _normalize_pages_for_chunkers(pages)
        if not any((t or "").strip() for _, t in pages_std):
            print("[PARSE] no textual content after parsing/OCR; will use fallback if chunkers return empty")
        job_state.update(job_id, status="parsing", step="parse_pdf:done", progress=25)

        # ---------- 2) 청킹 ----------
        job_state.update(job_id, status="chunking", step="chunk:start", progress=35)

        # 인코더/길이
        enc, max_len = _make_encoder()
        default_target = max(64, max_len - 16)
        default_overlap = min(96, default_target // 3)
        target_tokens = int(os.getenv("RAG_CHUNK_TOKENS", str(default_target)))
        overlap_tokens = int(os.getenv("RAG_CHUNK_OVERLAP", str(default_overlap)))

        chunks: list[tuple[str, dict]] | None = None

        # 2-1) 레이아웃 인지 청킹
        try:
            from app.services.layout_chunker import layout_aware_chunks  # type: ignore
            chunks = layout_aware_chunks(
                pages_std, enc, target_tokens, overlap_tokens,
                slide_rows=4, layout_blocks=layout_map
            )
            if not chunks:
                raise RuntimeError("layout-aware 결과 비어있음")
        except Exception as e1:
            # 2-2) 플러스 청커
            try:
                from app.services.chunker import smart_chunk_pages_plus, smart_chunk_pages
                chunks = smart_chunk_pages_plus(
                    pages_std, enc,
                    target_tokens=target_tokens, overlap_tokens=overlap_tokens,
                    layout_blocks=layout_map
                )
                if not chunks:
                    raise RuntimeError("plus 결과 비어있음")
            except Exception as e2:
                print(f"[CHUNK] layout-aware/plus failed ({e1}); fallback to smart ({e2})")
                # 2-3) 기본 청커
                chunks = smart_chunk_pages(
                    pages_std, enc,
                    target_tokens=target_tokens, overlap_tokens=overlap_tokens
                )

        # ---------- 2-4) bytes-OCR 세컨드 찬스 + 기본 청커 ----------
        if (not chunks or len(chunks) == 0) and use_bytes_path and OCR_BYTES_ENABLED:
            if pdf_bytes:
                try:
                    from app.services.ocr_service import try_ocr_pdf_bytes
                    ocr_text2 = try_ocr_pdf_bytes(pdf_bytes, enabled=True)
                except Exception as _e:
                    print(f"[CHUNK] second-chance OCR failed: {getattr(_e, 'message', _e)}")
                    ocr_text2 = None

                if ocr_text2:
                    pages_std = _normalize_pages_for_chunkers([(1, ocr_text2)])
                    from app.services.chunker import smart_chunk_pages
                    chunks = smart_chunk_pages(
                        pages_std, enc,
                        target_tokens=target_tokens, overlap_tokens=overlap_tokens
                    )
                    print(f"[CHUNK] second-chance via OCR succeeded, chunks={len(chunks or [])})")

        # ---------- 2-5) 최후 보호막: 통짜 텍스트라도 1개 청크 생성 ----------
        if not chunks or len(chunks) == 0:
            flat_texts = []
            for _, t in pages_std or []:
                tt = (t or "").strip()
                if tt:
                    flat_texts.append(tt)

            # 텍스트가 전혀 없으면, 페이지 수만큼 placeholder라도 만들어 크래시 방지
            fallback_text = "\n\n".join(flat_texts).strip()
            if not fallback_text:
                # bytes가 있고 페이지 수를 구할 수 있으면 이미지-플레이스홀더
                try:
                    if pdf_bytes:
                        import fitz
                        doc_dbg = fitz.open(stream=pdf_bytes, filetype="pdf")
                        placeholders = [f"[page {i+1}: image-only]" for i in range(doc_dbg.page_count)]
                        fallback_text = "\n".join(placeholders).strip()
                except Exception:
                    pass

            # 그래도 비면, ENV로 동작 결정: 기본은 크래시 방지용 더미 텍스트 생성
            if not fallback_text:
                if os.getenv("RAG_ALLOW_EMPTY_FALLBACK", "1") == "1":
                    fallback_text = "[empty document after parsing/OCR]"
                else:
                    raise RuntimeError("Chunking 결과가 비었습니다.")

            # META 구성
            import json
            meta = {
                "type": "text",
                "section": "",
                "pages": [int(p) for p, _ in (pages_std or [])] or [1],
                "bboxes": {},
            }
            meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
            chunk_text = meta_line + "\n" + fallback_text
            first_page = int(meta["pages"][0]) if meta["pages"] else 0
            chunks = [(chunk_text, {"page": first_page, "section": "", "pages": meta["pages"], "bboxes": {}})]

        # ---------- 2-6) 최종 검사 ----------
        chunks = _coerce_chunks_for_milvus(chunks)
        if not chunks:
            raise RuntimeError("Chunking 결과가 비었습니다.")

        job_state.update(job_id, status="chunking", step="chunk:done", chunks=len(chunks), progress=50)

        # ---------- 3) doc_id 확정 ----------
        if not doc_id:
            base_from_obj = os.path.splitext(os.path.basename(minio_object or ""))[0] if minio_object else None
            doc_id = base_from_obj or (os.path.splitext(os.path.basename(file_path))[0] if file_path else None)
            if not doc_id:
                import uuid
                doc_id = uuid.uuid4().hex

        REPLACE_BEFORE_INSERT = os.getenv("RAG_REPLACE_BEFORE_INSERT", "0") == "1"
        RETRY_AFTER_DELETE_ON_DUP = os.getenv("RAG_RETRY_AFTER_DELETE", "1") == "1"

        st = job_state.get(job_id) or {}
        mode = st.get("mode")  # 'replace' | 'version' | 'skip' 등

        from app.services.milvus_store_v2 import MilvusStoreV2
        from app.services.embedding_model import embed, get_sentence_embedding_dimension
        store = MilvusStoreV2(dim=get_sentence_embedding_dimension())

        if mode == "replace" or REPLACE_BEFORE_INSERT:
            try:
                if hasattr(store, "delete_by_doc_id"):
                    deleted = store.delete_by_doc_id(doc_id)  # type: ignore
                else:
                    deleted = store._delete_by_doc_id(doc_id)  # type: ignore
                print(f"[INDEX] pre-delete for replace: doc_id={doc_id}, deleted={deleted}")
            except Exception as e:
                print(f"[INDEX] pre-delete warn: {e}")

        # ---------- 4) Milvus upsert ----------
        job_state.update(job_id, status="embedding", step="embed:start", progress=60)
        res = store.insert(doc_id, chunks, embed_fn=embed)  # {inserted, skipped, reason, doc_id}
        real_doc_id = res.get("doc_id", doc_id)

        if res.get("skipped") and (mode == "replace" or RETRY_AFTER_DELETE_ON_DUP):
            reason = (res.get("reason") or "").lower()
            if any(k in reason for k in ["duplicate", "exists", "doc_id"]):
                try:
                    if hasattr(store, "delete_by_doc_id"):
                        deleted = store.delete_by_doc_id(real_doc_id)  # type: ignore
                    else:
                        deleted = store._delete_by_doc_id(real_doc_id)  # type: ignore
                    print(f"[INDEX] retry-after-delete: deleted={deleted}, doc_id={real_doc_id}")
                    res = store.insert(doc_id, chunks, embed_fn=embed)
                    real_doc_id = res.get("doc_id", doc_id)
                except Exception as e:
                    print(f"[INDEX] retry-after-delete failed: {e}")

        if res.get("skipped"):
            job_state.update(job_id, status="indexing", step=f"milvus:skipped:{res.get('reason')}",
                             progress=90, doc_id=real_doc_id)
            print(f"[INDEX] skipped: doc_id={real_doc_id}, reason={res.get('reason')}")
        else:
            job_state.update(job_id, status="indexing", step=f"milvus:inserted:{res.get('inserted',0)}",
                             progress=90, doc_id=real_doc_id)
            print(f"[INDEX] done: {minio_object or file_path} (doc_id={real_doc_id}, chunks={len(chunks)}, "
                  f"inserted={res.get('inserted',0)})")

        # ---------- 5) MinIO 원본 삭제(옵션) ----------
        if os.getenv("RAG_DELETE_AFTER_INDEX", "0") == "1" and minio_object and uploaded:
            try:
                from app.services.minio_store import MinIOStore
                MinIOStore().delete(minio_object)
                print(f"[CLEANUP] deleted from MinIO: {minio_object}")
                job_state.update(job_id, status="cleanup", step="minio:deleted",
                                 minio_object=minio_object, progress=95)
            except Exception as e:
                print(f"[CLEANUP] delete failed: {e}")
                job_state.update(job_id, status="cleanup", step=f"minio:delete_failed:{e!s}")

        # ---------- 6) 로컬 파일 정리 ----------
        if remove_local and file_path and not use_bytes_path:
            try:
                os.remove(file_path)
            except Exception:
                pass
        # TOTAL 청크 수 집계    
        total = len(chunks) if isinstance(chunks, list) else None

        if total is None:
            try:
                total = milvus_store.count_by_doc(doc_id)  # 네가 가진 카운트 함수
            except Exception:
                total = None
    
        # 메타 갱신
        try:
            store = MinIOStore()
            meta = {}
            try:
                meta = store.get_json(meta_key(doc_id))
            except Exception:
                meta = {}
    
            meta = dict(meta or {})
            if total is not None:
                meta["chunk_count"] = int(total)
    
            store.put_json(meta_key(doc_id), meta)
        except Exception as e:
            print(f"[INDEXER] warn: failed to update meta chunk_count: {e}")
    
        # ---------- 7) 완료 ----------
        job_state.complete(
            job_id,
            pages=len(pages_std or []),
            chunks=len(chunks or []),
            doc_id=real_doc_id,
            inserted=int(res.get("inserted", 0)),
            skipped=bool(res.get("skipped", False)),
            reason=res.get("reason"),
        )

    except Exception as e:
        job_state.fail(job_id, str(e))
        raise

def _content_disposition(disposition: str, filename: str) -> str:
    """
    latin-1 제한을 피하기 위해:
    - ASCII fallback: 파일명에서 비ASCII를 _ 로 대체
    - filename*: UTF-8''<percent-encoded> 함께 제공
    """
    # fallback: ASCII만 남기기
    ascii_fallback = re.sub(r'[^A-Za-z0-9._-]+', '_', filename) or 'file'
    utf8_quoted = quote(filename)  # UTF-8 percent-encode
    return f"{disposition}; filename=\"{ascii_fallback}\"; filename*=UTF-8''{utf8_quoted}"

def _strip_meta_line(chunk_text: str) -> str:
    """청크 맨 위 META: 라인을 제거하고 본문만 반환"""
    t = chunk_text or ""
    if t.startswith("META:"):
        nl = t.find("\n")
        t = t[nl+1:] if nl != -1 else ""
    return t.strip()

_DEF_PATTS = ("뭐야", "무엇", "뭔가", "의미", "정의", "설명", "어떤", "무엇인가", "무엇인지")

def normalize_query(q: str) -> str:
    """
    정의/설명형 질문을 검색 친화적으로 보강:
    - '... 뭐야/무엇/의미' 등을 '... 내용'으로 보강
    - 너무 과하게 바꾸지 않고 원문을 유지하되 '내용', '정의' 토큰을 추가
    """
    base = q.strip()
    lowered = base.lower()
    if any(p in base for p in _DEF_PATTS):
        # 핵심 키워드 보존 + 내용/정의를 덧붙여 벡터 검색 친화화
        return f"{base} 내용 정의"
    return base

_KW_TOKEN_RE = re.compile(r"[A-Za-z가-힣0-9\.#\-]+")  # '57b항', '§57(b)', 'AEA-57b' 류 보존

def extract_keywords(q: str) -> list[str]:
    """
    질문에서 검색 키워드 후보 추출(짧은 조사류/한 글자 토큰 제거)
    """
    toks = [t for t in _KW_TOKEN_RE.findall(q) if len(t) >= 2]
    # 중복 제거(순서 보존)
    seen, out = set(), []
    for t in toks:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl); out.append(t)
    return out

# ---------- Routes ----------
@router.get("/test")
def test():
    return {"status": "LLaMA router is working"}

@router.post("/generate")
def generate(body: GenerateReq):
    try:
        result = generate_answer_unified(body.prompt, body.model_name)
        return {"response": result}
    except Exception as e:
        raise HTTPException(500, f"모델 응답 생성 실패: {e}")

@router.post("/upload", response_model=UploadResp)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Query("version", regex="^(skip|version|replace)$"),
):
    # 0) 업로드 원본을 메모리로 읽음
    safe_name = os.path.basename(file.filename or "upload.bin")
    orig_ct = file.content_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    content = await file.read()
    if not content:
        raise HTTPException(400, "빈 파일입니다.")

    m = MinIOStore()

    # 1) 비-PDF → PDF bytes 변환 (DOC_CONVERTER_URL 있으면 스트림 변환, 없으면 임시파일-폴백)
    src_ext = os.path.splitext(safe_name)[1].lower()
    # 1순위: Gotenberg 바이트 변환 (완전 무-디스크)
    pdf_bytes: bytes | None = None
    if src_ext == ".pdf":
        pdf_bytes = content
    else:
        pdf_bytes = convert_bytes_to_pdf_bytes(content, src_ext)
    
    # 2순위: 사내/외부 컨버터 (DOC_CONVERTER_URL), 있으면 사용
    if pdf_bytes is None and src_ext != ".pdf":
        try:
            pdf_bytes = convert_stream_to_pdf_bytes(content, src_ext)
        except Exception:
            pdf_bytes = None
    
    # 3순위: 임시폴더 폴백 (convert_to_pdf) → 변환 후 즉시 삭제
    if pdf_bytes is None:
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, safe_name)
            with open(src_path, "wb") as f:
                f.write(content)
            out_path = convert_to_pdf(src_path)
            with open(out_path, "rb") as f:
                pdf_bytes = f.read()
    # 2) 해시/중복판정
    pdf_filename = safe_name if src_ext == ".pdf" else (os.path.splitext(safe_name)[0] + ".pdf")
    pdf_sha = _sha256_bytes(pdf_bytes)
    hash_flag_key = f"uploaded/__hash__/sha256/{pdf_sha}.flag"
    object_pdf = f"uploaded/{pdf_filename}"
    uploaded = True
    duplicate_reason = None

    if m.exists(hash_flag_key):
        uploaded = False
        duplicate_reason = "same_content_hash"

    if uploaded and m.exists(object_pdf):
        try:
            remote_size = m.size(object_pdf)
        except Exception:
            remote_size = -1
        local_size = len(pdf_bytes)
        if remote_size == local_size and remote_size > -1:
            uploaded = False
            duplicate_reason = (duplicate_reason or "same_name_and_size")
        else:
            if mode == "replace":
                m.upload_bytes(pdf_bytes, object_name=object_pdf, content_type="application/pdf", length=len(pdf_bytes))
            else:
                object_pdf = f"uploaded/{uuid.uuid4().hex}_{pdf_filename}"
                m.upload_bytes(pdf_bytes, object_name=object_pdf, content_type="application/pdf", length=len(pdf_bytes))
    elif uploaded:
        m.upload_bytes(pdf_bytes, object_name=object_pdf, content_type="application/pdf", length=len(pdf_bytes))

    # 2-1) 해시 플래그(바이트 직업로드: 로컬 파일 사용하지 않음)
    try:
        if uploaded and not m.exists(hash_flag_key):
            m.upload_bytes(b"1", object_name=hash_flag_key, content_type="text/plain", length=1)
    except Exception as e:
        # 치명적 아님
        print(f"[UPLOAD] warn: failed to write hash flag: {e}")

    # 3) doc_id 결정 및 ‘원본’ 바이트 업로드 (문서별 폴더로)
    doc_id = os.path.splitext(os.path.basename(object_pdf))[0]
    object_orig = f"uploaded/originals/{doc_id}/{safe_name}"
    if m.exists(object_orig):
        try:
            rsize = m.size(object_orig)
        except Exception:
            rsize = -1
        if rsize != len(content):
            object_orig = f"uploaded/originals/{doc_id}/{uuid.uuid4().hex}_{safe_name}"

    m.upload_bytes(content, object_name=object_orig, content_type=orig_ct, length=len(content))
    is_pdf_original = (src_ext == ".pdf")
    # 4) 매핑 메타 JSON
    try:
        meta = {
            "doc_id": doc_id,
            "title": safe_name,                # 보기용
            "pdf_key": object_pdf,             # ← 키 이름을 pdf_key로 통일
            "original_key": object_orig,       # ← original_key 통일
            "original_name": safe_name,
            "is_pdf_original": is_pdf_original,
            "sha256": pdf_sha,
            # 업로드 시간은 UTC로 저장(프런트에서 KST로 렌더 추천)
            "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": mode,
            # 아직 모름 → 나중에 인덱서가 덮어씀
            # "chunk_count": null
        }
        m.put_json(meta_key(doc_id), meta)
    except Exception as e:
        print(f"[UPLOAD] warn: failed to write meta json: {e}")

    # 5) 백그라운드 인덱싱 (MinIO bytes 경로만 넘김)
    job_id = uuid.uuid4().hex
    job_state.start(job_id, doc_id=doc_id, minio_object=object_pdf)
    job_state.update(
        job_id,
        status="uploaded",
        step="minio:ok",
        filename=safe_name,
        progress=10,
        mode=mode,
        content_sha256=pdf_sha,
        duplicate_reason=duplicate_reason,
        uploaded=uploaded,
    )
    background_tasks.add_task(index_pdf_to_milvus, job_id, None, object_pdf, uploaded, False, doc_id)

    return UploadResp(filename=safe_name, minio_object=object_pdf, indexed="background", job_id=job_id)

@router.post("/ask", response_model=AskResp)
def ask_question(req: AskReq):
    try:
        # 0) 모델/스토어 준비
        model = get_embedding_model()
        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())

        # 1) 질문 전처리(쿼리 보강) + 초기 넉넉히 검색
        query_for_search = normalize_query(req.question)
        raw_topk = max(20, req.top_k * 5)
        cands = store.search(query_for_search, embed_fn=embed, topk=raw_topk)

        if not cands:
            return AskResp(
                answer="업로드된 문서에서 관련 내용을 찾을 수 없습니다. 문서가 올바르게 인덱싱되었는지 확인해주세요.",
                used_chunks=0,
                sources=[]
            )

        # 2) 키워드 부스트
        kws = extract_keywords(req.question)
        def _kw_boost_score(c: dict) -> int:
            txt = _strip_meta_line(c.get("chunk", "")).lower()
            return sum(1 for k in kws if k.lower() in txt)
        for c in cands:
            c["kw_boost"] = _kw_boost_score(c)

        ARTICLE_BOOST = float(os.getenv("RAG_ARTICLE_BOOST", "2.5"))
        m = re.search(r"제\s*(\d+)\s*조", req.question)
        if m:
            art = m.group(1)
            patt = re.compile(rf"제\s*{art}\s*조")
            for c in cands:
                sec = c.get("section") or ""
                txt = c.get("chunk") or ""
                if patt.search(sec) or patt.search(txt):
                    c["kw_boost"] = c.get("kw_boost", 0.0) + ARTICLE_BOOST

        cands.sort(key=lambda x: (x.get("kw_boost", 0), x.get("score", 0.0)), reverse=True)

        # 3) 리랭크
        topk = rerank(req.question, cands, top_k=req.top_k)
        if not topk:
            return AskResp(
                answer="문서에서 신뢰할 수 있는 관련 내용을 찾지 못했습니다.",
                used_chunks=0,
                sources=[]
            )

        # 4) 스코어 컷오프
        THRESH = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))
        if "re_score" in topk[0] and topk[0]["re_score"] < THRESH:
            return AskResp(
                answer="문서에서 해당 질문에 대한 확실한 답변을 찾기 어렵습니다.",
                used_chunks=0,
                sources=[]
            )

        # 5) 컨텍스트/출처 구성
        context_lines = []
        sources = []
        for i, c in enumerate(topk, 1):
            sec = (c.get("section") or "").strip()
            chunk_body = _strip_meta_line(c.get("chunk", ""))
            body_only = f"{sec}\n{chunk_body}" if sec and not chunk_body.startswith(sec) else chunk_body
            context_lines.append(f"[{i}] (doc:{c['doc_id']} p.{c['page']} {c.get('section','')})\n{body_only}")
            sources.append({
                "id": i,
                "doc_id": c.get("doc_id"),
                "page": c.get("page"),
                "section": c.get("section"),
                "chunk": c.get("chunk"),
                "score": c.get("re_score", c.get("score")),
            })
        context = "\n\n".join(context_lines)

        # 6) 프롬프트
        prompt = f"""다음 문서 내용을 바탕으로 질문에 답하세요.

[중요 규칙]
- 문서에 명확한 근거가 있는 경우에만 답변하세요
- 추측이나 일반적인 지식으로 답하지 마세요
- 답변은 2-3문장으로 간결하게 작성하세요
- 같은 내용을 반복하지 마세요
- 문서에서 찾을 수 없으면 "문서에서 해당 내용을 찾을 수 없습니다"라고 답하세요

[참고 문서]
{context}

[질문]
{req.question}

[답변]"""

        answer = generate_answer_unified(prompt, req.model_name)
        answer = _clean_repetitive_answer(answer)

        return AskResp(answer=answer, used_chunks=len(topk), sources=sources)

    except HTTPException:
        raise
    except RuntimeError as milvus_error:
        raise HTTPException(503, f"Milvus 연결 대기/검색 실패: {milvus_error}")
    except Exception as e:
        raise HTTPException(500, f"질의 처리 중 오류: {e}")


def _clean_repetitive_answer(answer: str) -> str:
    """반복되는 답변 패턴을 정리"""
    if not answer:
        return answer
    
    # 매우 긴 답변 잘라내기 (1000자 초과 시)
    if len(answer) > 1000:
        sentences = answer.split('.')
        clean_sentences = []
        for sentence in sentences[:5]:  # 최대 5문장만
            if sentence.strip() and len(sentence.strip()) > 10:
                clean_sentences.append(sentence.strip())
        answer = '. '.join(clean_sentences) + '.'
    
    # 반복되는 구문 제거 (같은 구문이 3번 이상 반복되면 제거)
    lines = answer.split('\n')
    seen_lines = {}
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line in seen_lines:
            seen_lines[line] += 1
            if seen_lines[line] <= 2:  # 최대 2번까지만 허용
                filtered_lines.append(line)
        else:
            seen_lines[line] = 1
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines).strip()

# ---------- Job State Management ----------
@router.get("/job/{job_id}")
def get_job(job_id: str):
    st = job_state.get(job_id)
    if not st:
        raise HTTPException(404, "해당 job_id를 찾을 수 없습니다.")
    return st

@router.get("/jobs")
def list_jobs(status: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=500)):
    return {"jobs": job_state.list_jobs(status=status, limit=limit)}

@router.get("/doc/{doc_id}")
def doc_status(doc_id: str):
    s = MinIOStore()

    # 1) 메타 우선 (신규 경로)
    try:
        if s.exists(meta_key(doc_id)):
            meta = s.get_json(meta_key(doc_id))
        elif s.exists(legacy_meta_key(doc_id)):  # 구버전 파일형 폴백
            meta = s.get_json(legacy_meta_key(doc_id))
        else:
            meta = None
    except Exception:
        meta = None

    if isinstance(meta, dict):
        # 키 호환(pdf/pdf_key, original/original_key 등) 처리
        chunk_count = meta.get("chunk_count")
        if isinstance(chunk_count, int):
            return {"doc_id": doc_id, "chunks": chunk_count, "indexed": chunk_count > 0}

    # 2) 폴백: Milvus에서 세고 메타에 캐시
    try:
        model = get_embedding_model()
        m = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())
        total = m.count_by_doc(doc_id)

        # 메타에 캐시
        try:
            meta = (s.get_json(meta_key(doc_id)) if s.exists(meta_key(doc_id)) else {}) or {}
            meta["doc_id"] = doc_id
            meta["chunk_count"] = int(total)
            s.put_json(meta_key(doc_id), meta)
        except Exception:
            pass

        return {"doc_id": doc_id, "chunks": total, "indexed": total > 0}
    except Exception as e:
        raise HTTPException(500, f"doc status 실패: {e}")


# ========== SSE Stream for Job Status =========
@router.get("/job/{job_id}/stream")
async def stream_job(job_id: str):
    async def event_gen():
        last_serialized = None
        while True:
            st = job_state.get(job_id)
            if not st:
                yield {"event": "error", "data": json.dumps({"error": "not found"}, ensure_ascii=False)}
                break

            data = json.dumps(st, ensure_ascii=False)
            if data != last_serialized:
                yield {"event": "update", "data": data}
                last_serialized = data

                if st.get("status") in ("done", "error"):
                    break

            await asyncio.sleep(1)

    return EventSourceResponse(event_gen())

# ---------- MinIO Utilities ----------

    
@router.get("/files")
def list_files(prefix: str = "uploaded/", include_internal: bool = False, only_pdf: bool = False):
    m = MinIOStore()
    try:
        keys = m.list_files(prefix=prefix) 
    except Exception as e:
        raise HTTPException(500, f"MinIO 파일 조회 실패: {e}")

    # 내부 관리 오브젝트 숨기기 (원하면 include_internal=True로 노출)
    if not include_internal:
        keys = [k for k in keys if not (k.endswith(".flag") or "/__hash__/" in k or "/__meta__/" in k)]

    if only_pdf:
        keys = [k for k in keys if k.lower().endswith(".pdf")]

    return {"files": keys}


@router.get("/file/{object_name:path}")
def get_file_presigned(
    object_name: str,
    minutes: int = Query(60, ge=1, le=7*24*60),
    download_name: Optional[str] = None,
    inline: bool = False,  # true면 inline, false면 attachment
):
    key = unquote(object_name)
    m = MinIOStore()
    if not m.exists(key):
        raise HTTPException(404, f"object not found: {key}")

    try:
        if inline:
            url = m.presign_view(key, filename=download_name, ttl_seconds=minutes * 60)
        else:
            url = m.presign_download(key, filename=download_name, ttl_seconds=minutes * 60)
        return {"url": url}
    except Exception as e:
        raise HTTPException(500, f"presign failed: {e}")

# 기존
# @router.get("/docs")
# def list_docs():

@router.get("/rag/docs")
def list_docs():
    """
    변환 PDF와 원본 파일의 매핑을 반환.
    - doc_id: PDF 파일명(확장자 제외)
    - title: 사용자에게 보일 이름 (원본 이름이 있으면 원본명, 없으면 PDF 파일명)
    - object_key: 변환 PDF의 MinIO 키 (uploaded/...)
    - original_key: 원본의 MinIO 키 (uploaded/originals/...), 없으면 None
    - original_name: 원본 파일명 (예: .docx)
    - is_pdf_original: 원본도 pdf인지 여부
    - uploaded_at: 메타에 기록된 업로드 시간
    """
    m = MinIOStore()
    try:
        all_keys = m.list_files("uploaded/")
    except Exception as e:
        raise HTTPException(500, f"minio list failed: {e}")

    def is_internal(k: str) -> bool:
        return (
            k.endswith(".flag")
            or "/__hash__/" in k
            or "/__meta__/" in k
            or k.startswith("uploaded/originals/")
        )

    pdf_keys = [k for k in all_keys if not is_internal(k) and k.lower().endswith(".pdf")]

    items = []
    for k in pdf_keys:
        base = os.path.basename(k)
        doc_id = os.path.splitext(base)[0]
        meta_key = f"uploaded/__meta__/{doc_id}.json"

        meta = None
        if m.exists(meta_key):
            try:
                meta = m.get_json(meta_key)
            except Exception:
                meta = None

        original_key = meta.get("original") if meta else None
        original_name = meta.get("original_name") if meta else None
        uploaded_at = meta.get("uploaded_at") if meta else None

        # ✅ original_key 보정: 메타 경로가 없거나 존재하지 않으면 패턴별로 재탐색
        def _resolve_original() -> Optional[str]:
            # 1) 메타에 있는 경로가 정상이면 그걸 사용
            if original_key and m.exists(original_key):
                return original_key
            # 2) 표준 폴더 패턴: uploaded/originals/{doc_id}/{original_name}
            if original_name:
                cand1 = f"uploaded/originals/{doc_id}/{original_name}"
                if m.exists(cand1):
                    return cand1
                # 3) 레거시 평면 패턴: uploaded/originals/{original_name}
                cand2 = f"uploaded/originals/{original_name}"
                if m.exists(cand2):
                    return cand2
            # 4) 폴더 스캔(혹시 이름이 바뀐 경우)
            cands = m.list_files(f"uploaded/originals/{doc_id}/")
            if cands:
                return cands[0]
            return None

        resolved_orig = _resolve_original()
        if resolved_orig:
            original_key = resolved_orig
            if not original_name:
                original_name = os.path.basename(resolved_orig)

        title = original_name or base
        is_pdf_original = bool(original_key and original_key.lower().endswith(".pdf"))

        items.append({
            "doc_id": doc_id,
            "title": title,
            "object_key": k,                  # 변환 PDF (뷰어용)
            "original_key": original_key,     # 원본 (다운로드용)
            "original_name": original_name,
            "is_pdf_original": is_pdf_original,
            "uploaded_at": uploaded_at,
        })

    return {"docs": items}

# 메타 JSON 단건 조회 추가(디버그/점검용)
@router.get("/rag/meta/{doc_id}")
def get_meta(doc_id: str):
    m = MinIOStore()
    key = f"uploaded/__meta__/{doc_id}.json"
    if not m.exists(key):
        raise HTTPException(404, f"meta not found for {doc_id}")
    try:
        return m.get_json(key)
    except Exception as e:
        raise HTTPException(500, f"read meta failed: {e}")

@router.get("/status")
def status():
    m = MinIOStore()
    try:
        keys = m.list_files("uploaded/")
        pdfs = [k for k in keys if k.lower().endswith(".pdf") and "/__hash__/" not in k and "/__meta__/" not in k]
        return {"has_data": len(pdfs) > 0, "doc_count": len(pdfs)}
    except Exception:
        return {"has_data": False, "doc_count": 0}


@router.delete("/file/{object_name}")
def delete_file(object_name: str):
    try:
        minio = MinIOStore()
        if not minio.exists(object_name):
            raise HTTPException(404, "파일이 존재하지 않습니다.")
        minio.delete(object_name)
        return {"status": "ok", "deleted": object_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"파일 삭제 실패: {e}")
    
@router.get("/view/{object_name:path}")
def view_object(object_name: str, name: Optional[str] = None):
    key = unquote(object_name)
    m = MinIOStore()
    if not m.exists(key):
        raise HTTPException(404, f"object not found: {key}")

    disp_name = name or os.path.basename(key)
    ext = os.path.splitext(key)[1].lower()
    media = "application/pdf" if ext == ".pdf" else (mimetypes.guess_type(disp_name)[0] or "application/octet-stream")

    try:
        obj = m.client.get_object(m.bucket, key)
    except Exception as e:
        raise HTTPException(500, f"MinIO get_object failed: {e}")

    headers = {
        # 👇 latin-1 안전하게
        "Content-Disposition": _content_disposition("inline", disp_name)
    }

    def _iter():
        try:
            for chunk in obj.stream(32 * 1024):
                yield chunk
        finally:
            obj.close()
            obj.release_conn()

    return StreamingResponse(_iter(), media_type=media, headers=headers)


@router.get("/download/{object_name:path}")
def download_object(object_name: str, name: Optional[str] = None):
    key = unquote(object_name)
    m = MinIOStore()
    if not m.exists(key):
        raise HTTPException(404, f"object not found: {key}")

    disp_name = name or os.path.basename(key)
    media = mimetypes.guess_type(disp_name)[0] or "application/octet-stream"

    try:
        obj = m.client.get_object(m.bucket, key)
    except Exception as e:
        raise HTTPException(500, f"MinIO get_object failed: {e}")

    headers = {
        # 👇 latin-1 안전하게
        "Content-Disposition": _content_disposition("attachment", disp_name)
    }

    def _iter():
        try:
            for chunk in obj.stream(32 * 1024):
                yield chunk
        finally:
            obj.close()
            obj.release_conn()

    return StreamingResponse(_iter(), media_type=media, headers=headers)

# ---------- Bulk delete MinIO files under a prefix ----------
@router.delete("/files/purge", tags=["llama"])
def purge_files(
    prefix: str = Query("uploaded/", description="지울 경로 prefix (반드시 'uploaded/'로 시작)"),
    dry_run: bool = Query(False, description="true면 실제 삭제하지 않고 목록만 반환"),
    limit_preview: int = Query(50, ge=1, le=500, description="dry_run 때 미리보기 최대 개수"),
):
    """
    MinIO에서 특정 prefix 하위 객체들을 일괄 삭제.
    - 안전장치: prefix가 'uploaded/'로 시작하지 않으면 400 에러
    - dry_run=True 면 삭제 없이 목록 미리보기만
    """
    if not prefix or not prefix.startswith("uploaded/"):
        raise HTTPException(400, "prefix는 반드시 'uploaded/'로 시작해야 합니다.")

    try:
        minio = MinIOStore()
        files = minio.list_files(prefix=prefix)
    except Exception as e:
        raise HTTPException(500, f"MinIO 목록 조회 실패: {e}")

    matched = len(files)
    if dry_run:
        preview = files[:limit_preview]
        more = max(0, matched - len(preview))
        return {"status": "dry-run", "prefix": prefix, "matched": matched, "preview": preview, "more": more}

    deleted = 0
    failed = 0
    errors = []
    for obj in files:
        try:
            minio.delete(obj)
            deleted += 1
        except Exception as e:
            failed += 1
            errors.append({"object": obj, "error": str(e)})

    return {"status": "ok", "prefix": prefix, "matched": matched, "deleted": deleted, "failed": failed, "errors": errors}

# ========= Debug / Inspection =========
@router.get("/debug/milvus/info")
def debug_milvus_info():
    """ Milvus 상태 정보 조회"""
    try:
        model = get_embedding_model()
        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())
        return store.stats()
    except Exception as e:
        raise HTTPException(500, f"Milvus info 조회 실패: {e}")

@router.get("/debug/milvus/peek")
def debug_milvus_peek(limit: int = 100, full: bool = True, max_chars:int|None = None):
    """ Milvus 컬렉션의 일부 데이터 미리보기 """
    try:
        model = get_embedding_model()
        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())
        if full:
            os.environ["DEBUG_PEEK_MAX_CHARS"] = "0"
        elif max_chars is not None:
            os.environ["DEBUG_PEEK_MAX_CHARS"] = str(max_chars)
        return {"items": store.peek(limit=limit)}
    except Exception as e:
        raise HTTPException(500, f"Milvus peek 실패: {e}")

@router.get("/debug/milvus/by-doc")
def debug_milvus_by_doc(
    doc_id: str,
    limit: int = 100,
    full: bool = False,
    max_chars: int | None = None
):
    items: list = []            # 미리 초기화 (UnboundLocalError 방지)
    total: int | None = None

    try:
        model = get_embedding_model()
        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())

        # 길이 트렁케이션 제어
        if full or max_chars == 0:
            os.environ["DEBUG_PEEK_MAX_CHARS"] = "0"
        elif max_chars is not None:
            os.environ["DEBUG_PEEK_MAX_CHARS"] = str(max_chars)

        # 데이터 조회
        items = store.query_by_doc(doc_id=doc_id, limit=limit)

        # 총 개수(가능하면)
        try:
            total = store.count_by_doc(doc_id)
        except Exception:
            total = None

        # 항상 동일한 스키마로 반환
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "doc_id": doc_id,
        }

    except Exception as e:
        # 여기서는 로컬 변수 참조 금지!
        raise HTTPException(500, f"Milvus by-doc 실패: {e}")

@router.get("/debug/search")
def debug_vector_search(q: str, k: int = 5):
    try:
        model = get_embedding_model()
        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())
        raw = store.debug_search(q, embed_fn=embed, topk=k)
        return {"results": raw}
    except Exception as e:
        raise HTTPException(500, f"디버그 검색 실패: {e}")
