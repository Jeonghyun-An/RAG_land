# app/api/llama_router.py
from __future__ import annotations

import hashlib, tempfile
import os, re
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
from pydantic import BaseModel
import asyncio, json
from sse_starlette.sse import EventSourceResponse  # ✅ 요구사항: sse-starlette
from app.services import job_state

from app.services.file_parser import parse_pdf, parse_pdf_blocks
from app.services.llama_model import generate_answer_unified
from app.services.minio_store import MinIOStore
from app.services.pdf_converter import convert_to_pdf, ConvertError
from app.services.chunker import smart_chunk_pages, smart_chunk_pages_plus
# (지연 import로 대체) from app.services.layout_chunker import layout_aware_chunks
from app.services.milvus_store_v2 import MilvusStoreV2
from app.services.embedding_model import get_embedding_model, embed
from app.services.reranker import rerank

router = APIRouter(tags=["llama"])

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- Schemas ----------
class GenerateReq(BaseModel):
    prompt: str
    model_name: str = "llama-1b"

class AskReq(BaseModel):
    question: str
    model_name: str = "llama-1b"
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


def index_pdf_to_milvus(
    job_id: str,
    file_path: str,
    minio_object: str | None = None,
    uploaded: bool = True,
    remove_local: bool = True,
    doc_id: str | None = None,
) -> None:
    try:
        # 0) 시작 로그
        job_state.update(job_id, status="parsing", step="parse_pdf:start")
        print(f"[INDEX] start: {file_path}")

        # [ADDED] 업로더가 '중복 업로드'로 판단해 uploaded=False로 넘긴 경우,
        #         환경변수로 색인 스킵(기본 1) 여부 결정
        SKIP_IF_ALREADY_UPLOADED = os.getenv("RAG_SKIP_IF_UPLOADED", "1") == "1"
        if not uploaded and SKIP_IF_ALREADY_UPLOADED:
            job_state.update(job_id, status="done", step="skipped:already_uploaded", progress=100)
            print(f"[INDEX] skip: uploaded=False (already uploaded), job_id={job_id}")
            return

        # 1) PDF → 페이지 텍스트
        pages = parse_pdf(file_path, by_page=True)
        if not pages:
            raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다.")
        job_state.update(job_id, status="parsing", step="parse_pdf:done", progress=25)
        
        # 레이아웃 블록(BBOX)
        blocks_by_page_list = parse_pdf_blocks(file_path)  # [(page_no, [ {text,bbox} ])]
        layout_map = {p: blks for p, blks in blocks_by_page_list}

        # 2) 토큰 기준 청킹
        job_state.update(job_id, status="chunking", step="chunk:start", progress=35)

        model = get_embedding_model()
        tokenizer = getattr(model, "tokenizer", None)
        encode = (
            (lambda s: tokenizer.encode(s, add_special_tokens=False))
            if (tokenizer and hasattr(tokenizer, "encode"))
            else (lambda s: s.split())
        )

        # embedding 모델의 최대 길이에 맞춰 target/overlap 산정
        max_len = int(getattr(model, "max_seq_length", 128))
        default_target = max(64, max_len - 16)
        default_overlap = min(96, default_target // 3)
        target_tokens = int(os.getenv("RAG_CHUNK_TOKENS", str(default_target)))
        overlap_tokens = int(os.getenv("RAG_CHUNK_OVERLAP", str(default_overlap)))

        # >>> CHUNKING START
        chunks = None
        try:
            from app.services.layout_chunker import layout_aware_chunks  # type: ignore
            chunks = layout_aware_chunks(
                pages, encode, target_tokens, overlap_tokens, slide_rows=4, layout_blocks=layout_map
            )
            if not chunks:
                raise RuntimeError("layout-aware 결과 비어있음")
        except Exception as e1:
            try:
                chunks = smart_chunk_pages_plus(
                    pages, encode, target_tokens=target_tokens, overlap_tokens=overlap_tokens, layout_blocks=layout_map
                )
                if not chunks:
                    raise RuntimeError("plus 결과 비어있음")
            except Exception as e2:
                print(f"[CHUNK] layout-aware/plus failed ({e1}); fallback to smart")
                chunks = smart_chunk_pages(
                    pages, encode, target_tokens=target_tokens, overlap_tokens=overlap_tokens
                )
        # <<< CHUNKING END

        chunks = _coerce_chunks_for_milvus(chunks)
        if not chunks:
            raise RuntimeError("Chunking 결과가 비었습니다.")

        job_state.update(job_id, status="chunking", step="chunk:done", chunks=len(chunks), progress=50)

        # 3) doc_id 확정 (넘겨받은 값 > MinIO 객체명 > 파일명)
        if not doc_id:
            base_from_obj = os.path.splitext(os.path.basename(minio_object or ""))[0] if minio_object else None
            doc_id = base_from_obj or os.path.splitext(os.path.basename(file_path))[0]

        # [ADDED] 교체 정책(사전삭제/사후재시도) 플래그
        REPLACE_BEFORE_INSERT = os.getenv("RAG_REPLACE_BEFORE_INSERT", "0") == "1"
        RETRY_AFTER_DELETE_ON_DUP = os.getenv("RAG_RETRY_AFTER_DELETE", "1") == "1"

        # [ADDED] job_state에 모드/중복 사유가 있으면 반영(있으면 쓰고, 없으면 env만 사용)
        st = job_state.get(job_id) or {}
        mode = st.get("mode")  # 'replace' | 'version' | 'skip' 등이 있다면 사용
        duplicate_reason = st.get("duplicate_reason")

        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())

        # [ADDED] 모드가 replace면 삽입 전에 기존 doc_id 행들을 PK로 삭제
        if mode == "replace" or REPLACE_BEFORE_INSERT:
            try:
                # public 메서드가 있으면 사용, 없으면 _delete_by_doc_id로 폴백
                if hasattr(store, "delete_by_doc_id"):
                    deleted = store.delete_by_doc_id(doc_id)  # type: ignore
                else:
                    deleted = store._delete_by_doc_id(doc_id)  # type: ignore  # pylint: disable=protected-access
                print(f"[INDEX] pre-delete for replace: doc_id={doc_id}, deleted={deleted}")
            except Exception as e:
                print(f"[INDEX] pre-delete warn: {e}")

        # 4) Milvus upsert
        job_state.update(job_id, status="embedding", step="embed:start", progress=60)
        res = store.insert(doc_id, chunks, embed_fn=embed)  # {inserted, skipped, reason, doc_id}
        real_doc_id = res.get("doc_id", doc_id)

        # [ADDED] 만약 '중복' 때문에 skipped가 떴고, 정책상 교체라면: 삭제 후 1회 재시도
        if res.get("skipped") and (mode == "replace" or RETRY_AFTER_DELETE_ON_DUP):
            reason = (res.get("reason") or "").lower()
            if any(k in reason for k in ["duplicate", "exists", "doc_id"]):
                try:
                    if hasattr(store, "delete_by_doc_id"):
                        deleted = store.delete_by_doc_id(real_doc_id)  # type: ignore
                    else:
                        deleted = store._delete_by_doc_id(real_doc_id)  # type: ignore
                    print(f"[INDEX] retry-after-delete: deleted={deleted}, doc_id={real_doc_id}")
                    # 재시도
                    res = store.insert(doc_id, chunks, embed_fn=embed)
                    real_doc_id = res.get("doc_id", doc_id)
                except Exception as e:
                    print(f"[INDEX] retry-after-delete failed: {e}")

        if res.get("skipped"):
            job_state.update(
                job_id,
                status="indexing",
                step=f"milvus:skipped:{res.get('reason')}",
                progress=90,
                doc_id=real_doc_id,
            )
            print(f"[INDEX] skipped: doc_id={real_doc_id}, reason={res.get('reason')}")
        else:
            job_state.update(
                job_id,
                status="indexing",
                step=f"milvus:inserted:{res.get('inserted',0)}",
                progress=90,
                doc_id=real_doc_id,
            )
            print(
                f"[INDEX] done: {file_path} (doc_id={real_doc_id}, chunks={len(chunks)}, "
                f"inserted={res.get('inserted',0)})"
            )

        # 5) MinIO 원본 삭제(옵션) — 이번 요청에서 새로 올린(uploaded=True) 건만 정리
        if os.getenv("RAG_DELETE_AFTER_INDEX", "0") == "1" and minio_object and uploaded:
            try:
                MinIOStore().delete(minio_object)
                print(f"[CLEANUP] deleted from MinIO: {minio_object}")
                job_state.update(
                    job_id, status="cleanup", step="minio:deleted",
                    minio_object=minio_object, progress=95
                )
            except Exception as e:
                print(f"[CLEANUP] delete failed: {e}")
                job_state.update(job_id, status="cleanup", step=f"minio:delete_failed:{e!s}")

        # 6) 로컬 파일 정리(옵션)
        if remove_local:
            try:
                os.remove(file_path)
            except Exception:
                pass

        # 7) 완료
        job_state.complete(
            job_id,
            pages=len(pages),
            chunks=len(chunks),
            doc_id=real_doc_id,
            inserted=int(res.get("inserted", 0)),
            skipped=bool(res.get("skipped", False)),
            reason=res.get("reason"),
        )

    except Exception as e:
        job_state.fail(job_id, str(e))
        raise

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

    # 2) 비-PDF면 PDF로 변환
    try:
        pdf_path = convert_to_pdf(local_path)   # 이미 PDF면 그대로 반환
    except ConvertError as e:
        raise HTTPException(400, f"파일 변환 실패: {e}")
    except Exception as e:
        raise HTTPException(500, f"파일 변환 중 예외: {e}")

    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    pdf_sha = _sha256_file(pdf_path)
    hash_flag_key = f"uploaded/__hash__/sha256/{pdf_sha}.flag"

    # 3) MinIO 업로드 (원본 + 변환본 둘 다 올리고 싶으면 선택)
    minio = MinIOStore()

    pdf_name = os.path.basename(pdf_path)               # 예: doc.pdf
    object_pdf = f"uploaded/{pdf_name}"                 # 예: uploaded/doc.pdf
    uploaded = True
    duplicate_reason = None  

    #  3.0) 해시 플래그가 이미 있으면 동일 콘텐츠가 과거에 업로드됨
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

    #  새로 업로드(또는 replace로 덮어쓰기)된 경우 해시 플래그를 기록
    try:
        if uploaded and not minio.exists(hash_flag_key):
            # 간단한 flag 파일을 만들어 업로드
            flag_local = os.path.join(UPLOAD_DIR, "__hashflags__", f"{pdf_sha}.flag")
            os.makedirs(os.path.dirname(flag_local), exist_ok=True)
            with open(flag_local, "wb") as ff:
                ff.write(b"1")
            minio.upload(flag_local, object_name=hash_flag_key, content_type="text/plain")
            print(f"[UPLOAD] hash flag written: {hash_flag_key}")
    except Exception as e:
        # 플래그 실패는 치명적이지 않으므로 경고만
        print(f"[UPLOAD] warn: failed to write hash flag: {e}")

    # doc_id는 오브젝트 파일명(확장자 제외)로 통일하면 충돌 안 남 (기존 유지)
    doc_id = os.path.splitext(os.path.basename(object_pdf))[0]

    # 인덱싱(백그라운드)
    job_id = uuid.uuid4().hex
    job_state.start(job_id, doc_id=doc_id, minio_object=object_pdf)
    # [ADDED] 모드/해시/중복사유를 상태에 넣어 색인기에서 활용(교체/스킵 등)
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

    # index_pdf_to_milvus는 기존 시그니처 유지
    #   uploaded: False면 색인쪽에서 스킵/경량 처리 가능
    #   mode: replace인 경우 job_state에서 읽어 기존 벡터 삭제 후 재색인하도록 구현 권장
    background_tasks.add_task(index_pdf_to_milvus, job_id, pdf_path, object_pdf, uploaded, False, doc_id)

    return UploadResp(
        filename=safe_name,
        minio_object=object_pdf,
        indexed="background",
        job_id=job_id,
    )

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
        if not cands:
            raise HTTPException(404, "관련 문서를 찾지 못했습니다. 먼저 문서를 업로드/인덱싱 해주세요.")

        # 2) 키워드 부스트(간단 가산점) — rerank 전에 상위권으로 끌어올림
        kws = extract_keywords(body.question)
        def _kw_boost_score(c: dict) -> int:
            txt = _strip_meta_line(c.get("chunk", "")).lower()
            return sum(1 for k in kws if k.lower() in txt)

        for c in cands:
            c["kw_boost"] = _kw_boost_score(c)

        # kw_boost 우선 → 동점 시 원래 score 유지
        cands.sort(key=lambda x: (x.get("kw_boost", 0), x.get("score", 0.0)), reverse=True)

        # 3) 리랭크 후 상위 K
        topk = rerank(body.question, cands, top_k=body.top_k)
        if not topk:
            return AskResp(answer="문서에서 확신할 수 있는 근거를 찾지 못했습니다.", used_chunks=0, sources=[])

        # 4) 임계값 컷오프(리랭커 스코어 기준)
        THRESH = float(os.getenv("RAG_SCORE_THRESHOLD", "0.2"))
        if "re_score" in topk[0] and topk[0]["re_score"] < THRESH:
            return AskResp(answer="문서에서 답을 확실히 찾기 어렵습니다.", used_chunks=0, sources=[])

        # 5) 컨텍스트 + 출처 구성 (본문만, META 제거)
        context_lines = []
        sources = []
        for i, c in enumerate(topk, 1):
            body_only = _strip_meta_line(c.get("chunk", ""))
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

        # 6) 프롬프트(정의/설명형 대응 지시 추가)
        prompt = f"""아래 문서를 참고해서 질문에 정확히 답하세요. 근거가 없으면 모른다고 답합니다.
- 정의/설명형 질문이면 문서 원문의 해당 내용을 간결히 요약하세요.
- 문서에 근거가 있으면 출처 번호로 인용하세요: [1]

[문서 발췌]
{context}

[질문]
{body.question}

[형식]
- 한 문장 핵심 답
- 필요하면 근거 출처 번호로 인용: [1]
"""
        answer = generate_answer_unified(prompt, body.model_name)
        return AskResp(answer=answer, used_chunks=len(topk), sources=sources)

    except HTTPException:
        raise
    except RuntimeError as milvus_error:
        raise HTTPException(503, f"Milvus 연결 대기/검색 실패: {milvus_error}")
    except Exception as e:
        raise HTTPException(500, f"질의 처리 중 오류: {e}")

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
def list_files(
    prefix: str = Query("", description="prefix로 필터링 (예: 'uploaded/')"),
):
    try:
        return {"files": MinIOStore().list_files(prefix=prefix)}
    except Exception as e:
        raise HTTPException(500, f"MinIO 파일 조회 실패: {e}")

@router.get("/file/{object_name}")
def get_file_url(
    object_name: str,
    method: str = Query("GET", pattern="^(GET|PUT)$"),
    minutes: int = Query(60, ge=1, le=7*24*60, description="URL 만료 시간(분)"),
    download_name: Optional[str] = Query(
        None, description="다운로드 파일명 지정 시 Content-Disposition 헤더 세팅"
    ),
):
    try:
        from datetime import timedelta
        minio = MinIOStore()

        response_headers = None
        if download_name:
            response_headers = {
                "response-content-disposition": f'attachment; filename="{download_name}"'
            }

        url = minio.presigned_url(
            object_name=object_name,
            method=method,
            expires=timedelta(minutes=minutes),
            response_headers=response_headers,
        )
        return {"url": url}
    except Exception as e:
        raise HTTPException(500, f"파일 사전서명 URL 생성 실패: {e}")

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
