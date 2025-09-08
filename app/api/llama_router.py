# app/api/llama_router.py
from __future__ import annotations

import mimetypes
import hashlib, tempfile
import os, re
import uuid
from urllib.parse import unquote, quote
from typing import List, Optional
from starlette.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
from pydantic import BaseModel
import asyncio, json
from sse_starlette.sse import EventSourceResponse  # ✅ 요구사항: sse-starlette
from app.services import job_state
from datetime import datetime, timedelta

from app.services.file_parser import (
    parse_pdf,                    # (local path) -> [(page_no, text)]
    parse_pdf_blocks,             # (local path) -> [(page_no, [ {text,bbox}, ... ])]
    parse_any_bytes,              # (filename, bytes) -> {"kind":"pdf", "pages":[...], "blocks":[...]}
    parse_pdf_blocks_from_bytes,  # (bytes) -> [(page_no, [ {text,bbox}, ... ])]
)
from app.services.llama_model import generate_answer_unified
from app.services.minio_store import MinIOStore
from app.services.pdf_converter import convert_to_pdf, ConvertError
from app.services.chunker import smart_chunk_pages, smart_chunk_pages_plus
from app.services.layout_chunker import layout_aware_chunks
from app.services.milvus_store_v2 import MilvusStoreV2
from app.services.embedding_model import get_embedding_model, embed, get_sentence_embedding_dimension
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

def index_pdf_to_milvus(
    job_id: str,
    file_path: str | None = None,
    minio_object: str | None = None,
    uploaded: bool = True,
    remove_local: bool = True,
    doc_id: str | None = None,
) -> None:
    try:
        job_state.update(job_id, status="parsing", step="parse_pdf:start")
        print(f"[INDEX] start: {file_path}")

        NO_LOCAL = os.getenv("RAG_NO_LOCAL", "0") == "1"

        SKIP_IF_ALREADY_UPLOADED = os.getenv("RAG_SKIP_IF_UPLOADED", "1") == "1"
        if not uploaded and SKIP_IF_ALREADY_UPLOADED:
            job_state.update(job_id, status="done", step="skipped:already_uploaded", progress=100)
            print(f"[INDEX] skip: uploaded=False (already uploaded), job_id={job_id}")
            return

        # 1) PDF → 페이지 텍스트 (+ 레이아웃 블록)
        pages = None
        layout_map = {}

        use_bytes_path = (NO_LOCAL or file_path is None) and bool(minio_object)
        if use_bytes_path:
            # MinIO → bytes → bytes 파서
            from app.services.minio_store import MinIOStore
            mstore = MinIOStore()
            pdf_bytes = mstore.get_bytes(minio_object)

            try:
                from app.services.file_parser import parse_any_bytes, parse_pdf_blocks_from_bytes
                parsed = parse_any_bytes(os.path.basename(minio_object), pdf_bytes)
                if parsed.get("kind") != "pdf":
                    raise RuntimeError("PDF 파이프라인만 인덱싱합니다. (변환 단계 확인)")
                pages = parsed.get("pages") or []

                # 핵심: BBox를 bytes 기반으로 생성
                blocks_by_page_list = parsed.get("blocks")
                if not blocks_by_page_list:
                    # parse_any_bytes가 blocks를 안 채웠다면, 전용 함수로 보완
                    blocks_by_page_list = parse_pdf_blocks_from_bytes(pdf_bytes)

                # blocks가 dict로 올 수도 있고(list of tuples로 올 수도 있음)
                if isinstance(blocks_by_page_list, dict):
                    layout_map = {int(k): v for k, v in blocks_by_page_list.items()}
                else:
                    layout_map = {int(p): blks for p, blks in (blocks_by_page_list or [])}
            except Exception as ee:
                raise RuntimeError(f"bytes parsing unavailable or failed: {ee}") from ee
        else:
            # 기존 로컬 경로 파서 유지
            pages = parse_pdf(file_path, by_page=True)
            if not pages:
                raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다.")
            blocks_by_page_list = parse_pdf_blocks(file_path)
            layout_map = {int(p): blks for p, blks in blocks_by_page_list}  # 🔹int 캐스팅

        # 여기서 표준화: 이후 모든 청킹 함수는 이 변수만 사용
        pages_std = _normalize_pages_for_chunkers(pages)

        job_state.update(job_id, status="parsing", step="parse_pdf:done", progress=25)

        # 2) 청킹
        job_state.update(job_id, status="chunking", step="chunk:start", progress=35)
        model = get_embedding_model()
        enc, max_len = _make_encoder()
        default_target = max(64, max_len - 16)
        default_overlap = min(96, default_target // 3)
        target_tokens = int(os.getenv("RAG_CHUNK_TOKENS", str(default_target)))
        overlap_tokens = int(os.getenv("RAG_CHUNK_OVERLAP", str(default_overlap)))

        chunks = None
        try:
            from app.services.layout_chunker import layout_aware_chunks  # type: ignore
            chunks = layout_aware_chunks(
                pages_std, enc, target_tokens, overlap_tokens,
                slide_rows=4, layout_blocks=layout_map
            )
            if not chunks:
                raise RuntimeError("layout-aware 결과 비어있음")
        except Exception as e1:
            try:
                chunks = smart_chunk_pages_plus(
                    pages_std, enc,
                    target_tokens=target_tokens, overlap_tokens=overlap_tokens,
                    layout_blocks=layout_map
                )
                if not chunks:
                    raise RuntimeError("plus 결과 비어있음")
            except Exception as e2:
                print(f"[CHUNK] layout-aware/plus failed ({e1}); fallback to smart ({e2})")
                chunks = smart_chunk_pages(
                    pages_std, enc,
                    target_tokens=target_tokens, overlap_tokens=overlap_tokens
                )

        chunks = _coerce_chunks_for_milvus(chunks)
        if not chunks:
            raise RuntimeError("Chunking 결과가 비었습니다.")

        job_state.update(job_id, status="chunking", step="chunk:done", chunks=len(chunks), progress=50)

        # 3) doc_id 확정 (넘겨받은 값 > MinIO 객체명 > 파일명)
        if not doc_id:
            base_from_obj = os.path.splitext(os.path.basename(minio_object or ""))[0] if minio_object else None
            doc_id = base_from_obj or (os.path.splitext(os.path.basename(file_path))[0] if file_path else None)
            if not doc_id:
                import uuid
                doc_id = uuid.uuid4().hex

        REPLACE_BEFORE_INSERT = os.getenv("RAG_REPLACE_BEFORE_INSERT", "0") == "1"
        RETRY_AFTER_DELETE_ON_DUP = os.getenv("RAG_RETRY_AFTER_DELETE", "1") == "1"

        st = job_state.get(job_id) or {}
        mode = st.get("mode")  # 'replace' | 'version' | 'skip' 등이 있으면 활용

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

        # 4) Milvus upsert
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
            print(f"[INDEX] done: {minio_object} (doc_id={real_doc_id}, chunks={len(chunks)}, "
                  f"inserted={res.get('inserted',0)})")

        # 5) MinIO 원본 삭제(옵션)
        if os.getenv("RAG_DELETE_AFTER_INDEX", "0") == "1" and minio_object and uploaded:
            try:
                MinIOStore().delete(minio_object)
                print(f"[CLEANUP] deleted from MinIO: {minio_object}")
                job_state.update(job_id, status="cleanup", step="minio:deleted",
                                 minio_object=minio_object, progress=95)
            except Exception as e:
                print(f"[CLEANUP] delete failed: {e}")
                job_state.update(job_id, status="cleanup", step=f"minio:delete_failed:{e!s}")

        # 6) 로컬 파일 정리(옵션) — 무디스크면 아무것도 안 함
        if remove_local and file_path and not use_bytes_path:
            try:
                os.remove(file_path)
            except Exception:
                pass

        # 7) 완료
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
    # 1) 로컬 저장
    safe_name = os.path.basename(file.filename)
    local_path = os.path.join(UPLOAD_DIR, safe_name)
    content = await file.read()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(content)

    # 1-1) 원본 MinIO 업로드 (중복/충돌 처리)
    minio = MinIOStore()
    object_orig = f"uploaded/originals/{safe_name}"
    if minio.exists(object_orig):
        try:
            rsize = minio.size(object_orig)
        except Exception:
            rsize = -1
        lsize = os.path.getsize(local_path)
        if rsize != lsize:
            object_orig = f"uploaded/originals/{uuid.uuid4().hex}_{safe_name}"
    orig_ct = file.content_type or mimetypes.guess_type(local_path)[0] or "application/octet-stream"
    # 원본 업로드 (항상 보관)
    minio.upload(local_path, object_name=object_orig, content_type=orig_ct)

    # 2) 비-PDF면 PDF로 변환
    try:
        pdf_path = convert_to_pdf(local_path)
    except ConvertError as e:
        raise HTTPException(400, f"파일 변환 실패: {e}")
    except Exception as e:
        raise HTTPException(500, f"파일 변환 중 예외: {e}")

    # 2-1) PDF 해시
    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    pdf_sha = _sha256_file(pdf_path)
    hash_flag_key = f"uploaded/__hash__/sha256/{pdf_sha}.flag"

    # 3) PDF 업로드 (기존 로직 유지 + 충돌 처리)
    pdf_name = os.path.basename(pdf_path)
    object_pdf = f"uploaded/{pdf_name}"
    uploaded = True
    duplicate_reason = None

    if minio.exists(hash_flag_key):
        uploaded = False
        duplicate_reason = "same_content_hash"
        print(f"[UPLOAD] dedup by hash: {hash_flag_key}")
        # mode=skip 이면 여기서 바로 '스킵'으로 표시(색인 단계에서 참조)

    # 기존: 이름 같고, 사이즈 같으면 스킵 
    if uploaded and minio.exists(object_pdf):
        try:
            remote_size = minio.size(object_pdf)
        except Exception:
            remote_size = -1
        local_size = os.path.getsize(pdf_path)

        if remote_size == local_size and remote_size > -1:
            uploaded = False
            duplicate_reason = duplicate_reason or "same_name_and_size"
            print(f"[UPLOAD] dedup hit: {object_pdf} (same name & size)")
        else:
            # 이름은 같지만 사이즈 다르면
            if mode == "replace":  # 동일 키로 덮어쓰기 (=교체)
                minio.upload(pdf_path, object_name=object_pdf, content_type="application/pdf")
                print(f"[UPLOAD] replaced existing: {object_pdf}")
            else:
                # 기존 동작: 충돌 회피용 새 키로 저장(버전 관리)
                object_pdf = f"uploaded/{uuid.uuid4().hex}_{pdf_name}"
                minio.upload(pdf_path, object_name=object_pdf, content_type="application/pdf")
                print(f"[UPLOAD] name match but size differs -> stored as: {object_pdf}")
    elif uploaded:
        # 최초 업로드
        minio.upload(pdf_path, object_name=object_pdf, content_type="application/pdf")
        print(f"[UPLOAD] stored: {object_pdf}")

    # 3-1) 해시 플래그 기록 (기존 유지)
    try:
        if uploaded and not minio.exists(hash_flag_key):
            flag_local = os.path.join(UPLOAD_DIR, "__hashflags__", f"{pdf_sha}.flag")
            os.makedirs(os.path.dirname(flag_local), exist_ok=True)
            with open(flag_local, "wb") as ff:
                ff.write(b"1")
            minio.upload(flag_local, object_name=hash_flag_key, content_type="text/plain")
            print(f"[UPLOAD] hash flag written: {hash_flag_key}")
    except Exception as e:
        print(f"[UPLOAD] warn: failed to write hash flag: {e}")

    # === NEW: PDF↔원본 매핑 메타 저장 (sidecar JSON) ===
    try:
        doc_id = os.path.splitext(os.path.basename(object_pdf))[0]
        meta = {
            "pdf": object_pdf,
            "original": object_orig,
            "original_name": safe_name,
            "original_mime": orig_ct,
            "sha256": pdf_sha,
            "uploaded_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "mode": mode,
        }
        meta_key = f"uploaded/__meta__/{doc_id}.json"
        minio.put_json(meta_key, meta)
    except Exception as e:
        print(f"[UPLOAD] warn: failed to write meta json: {e}")
        # 메타 실패는 치명적 아님

    # 인덱싱(백그라운드) 이하 기존 유지
    doc_id = os.path.splitext(os.path.basename(object_pdf))[0]
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

    background_tasks.add_task(index_pdf_to_milvus, job_id, pdf_path, object_pdf, uploaded, False, doc_id)

    return UploadResp(filename=safe_name, minio_object=object_pdf, indexed="background", job_id=job_id)

@router.post("/ask", response_model=AskResp)
def ask_question(body: AskReq):
    try:
        # 0) 모델/스토어 준비
        model = get_embedding_model()
        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())

        # 1) 질문 전처리(쿼리 보강) + 초기 넉넉히 검색
        query_for_search = normalize_query(body.question)
        raw_topk = max(20, body.top_k * 5)
        cands = store.search(query_for_search, embed_fn=embed, topk=raw_topk)
        
        # 검색 결과 없음 처리 개선
        if not cands:
            return AskResp(
                answer="업로드된 문서에서 관련 내용을 찾을 수 없습니다. 문서가 올바르게 인덱싱되었는지 확인해주세요.",
                used_chunks=0,
                sources=[]
            )

        # 2) 키워드 부스트(간단 가산점) — rerank 전에 상위권으로 끌어올림
        kws = extract_keywords(body.question)
        def _kw_boost_score(c: dict) -> int:
            txt = _strip_meta_line(c.get("chunk", "")).lower()
            return sum(1 for k in kws if k.lower() in txt)

        for c in cands:
            c["kw_boost"] = _kw_boost_score(c)

        ARTICLE_BOOST = float(os.getenv("RAG_ARTICLE_BOOST", "2.5"))

        m = re.search(r"제\s*(\d+)\s*조", body.question)
        if m:
            art = m.group(1)
            patt = re.compile(rf"제\s*{art}\s*조")
            for c in cands:
                sec = c.get("section") or ""
                txt = c.get("chunk") or ""
                # META 줄 제거한 본문에 대해서도 체크하고 싶으면 _strip_meta_line(txt) 사용
                if patt.search(sec) or patt.search(txt):
                    c["kw_boost"] = c.get("kw_boost", 0.0) + ARTICLE_BOOST
        # kw_boost 우선 → 동점 시 원래 score 유지
        cands.sort(key=lambda x: (x.get("kw_boost", 0), x.get("score", 0.0)), reverse=True)

        # 3) 리랭크 후 상위 K
        topk = rerank(body.question, cands, top_k=body.top_k)
        if not topk:
            return AskResp(
                answer="문서에서 신뢰할 수 있는 관련 내용을 찾지 못했습니다.",
                used_chunks=0,
                sources=[]
            )

        # 4) 임계값 컷오프(리랭커 스코어 기준) - 더 엄격하게
        THRESH = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))  # 0.2 → 0.3으로 상향
        if "re_score" in topk[0] and topk[0]["re_score"] < THRESH:
            return AskResp(
                answer="문서에서 해당 질문에 대한 확실한 답변을 찾기 어렵습니다.",
                used_chunks=0,
                sources=[]
            )

        # 5) 컨텍스트 + 출처 구성 (본문만, META 제거)
        context_lines = []
        sources = []
        for i, c in enumerate(topk, 1):
            sec = (c.get("section") or "").strip()
            body = _strip_meta_line(c.get("chunk",""))
            body_only = f"{sec}\n{body}" if sec and not body.startswith(sec) else body
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

        # 6) 개선된 프롬프트 (반복 방지 + 간결함 강조)
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
{body.question}

[답변]"""

        # 7) 모델 호출 시 추가 파라미터로 반복 방지
        answer = generate_answer_unified(prompt, body.model_name)
        
        # 8) 답변 후처리 - 반복되는 패턴 제거
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
    try:
        model = get_embedding_model()
        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())
        cnt = store.count_by_doc(doc_id)
        return {"doc_id": doc_id, "chunks": cnt, "indexed": cnt > 0}
    except Exception as e:
        raise HTTPException(500, f"Milvus 조회 실패: {e}")
    


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

    headers = None
    if download_name:
        disp = "inline" if inline else "attachment"
        headers = {"response-content-disposition": f'{disp}; filename="{download_name}"'}

    try:
        url = m.presigned_url(key, method="GET", expires=timedelta(minutes=minutes), response_headers=headers)
        return {"url": url}
    except Exception as e:
        raise HTTPException(500, f"presign failed: {e}")


@router.get("/docs")
def list_docs(limit: int = 200):
    m = MinIOStore()
    try:
        keys = m.list_files("uploaded/")
    except Exception as e:
        raise HTTPException(500, f"minio list failed: {e}")

    def is_internal(k: str) -> bool:
        return k.endswith(".flag") or "/__hash__/" in k or "/__meta__/" in k

    pdf_keys = [k for k in keys if k.lower().endswith(".pdf") and not is_internal(k)]
    out = []

    for k in pdf_keys:
        base = os.path.splitext(os.path.basename(k))[0]
        # 메타 JSON에서 원본 찾기
        meta_key = f"uploaded/__meta__/{base}.json"
        orig_key = None
        title = os.path.basename(k)

        try:
            if m.exists(meta_key):
                # 임시 파일로 받아서 로드
                tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.json")
                m.download(meta_key, tmp)
                with open(tmp, "r", encoding="utf-8") as fp:
                    meta = json.load(fp)
                os.remove(tmp)
                orig_key = meta.get("original") or meta.get("orig_object")
                title = meta.get("original_name") or title
        except Exception:
            pass

        # 폴백: originals/ 에서 같은 base 이름을 가진 원본 탐색
        if not orig_key:
            for ext in (".pdf", ".docx", ".hwpx", ".hwp"):
                cand = f"uploaded/originals/{base}{ext}"
                if m.exists(cand):
                    orig_key = cand
                    break

        # presigned URL 생성
        try:
            pdf_url = m.presigned_url(k, method="GET", expires=timedelta(minutes=60))
        except Exception as e:
            raise HTTPException(500, f"presign(pdf) failed: {e}")

        if orig_key and m.exists(orig_key):
            try:
                download_url = m.presigned_url(
                    orig_key, method="GET", expires=timedelta(minutes=60),
                    response_headers={"response-content-disposition": f'attachment; filename="{title}"'}
                )
            except Exception:
                download_url = pdf_url
        else:
            download_url = pdf_url  # 원본 없으면 PDF로 폴백

        out.append({
            "doc_id": base,
            "title": title,
            "object_key": k,        # PDF object
            "url": pdf_url,         # PDF 보기용
            "download_url": download_url,  # 원본 다운로드용
            "uploaded_at": None,
        })

        if len(out) >= limit:
            break

    return out

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
def debug_milvus_by_doc(doc_id: str, limit: int = 10, full: bool = False, max_chars:int|None = None):
    try:
        model = get_embedding_model()
        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())
        if full:
            os.environ["DEBUG_PEEK_MAX_CHARS"] = "0"
        elif max_chars is not None:
            os.environ["DEBUG_PEEK_MAX_CHARS"] = str(max_chars)
        return {"items": store.query_by_doc(doc_id=doc_id, limit=limit)}
    except Exception as e:
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
