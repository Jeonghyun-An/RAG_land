# app/api/llama_router.py
from __future__ import annotations

import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
from pydantic import BaseModel
import asyncio, json
from sse_starlette.sse import EventSourceResponse  # ✅ 요구사항: sse-starlette
from app.services import job_state

from app.services.file_parser import parse_pdf
from app.services.llama_model import generate_answer_unified
from app.services.minio_store import MinIOStore
from app.services.pdf_converter import convert_to_pdf, ConvertError
from app.services.chunker import smart_chunk_pages
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
    # (선택) 출처 제공
    sources: Optional[List[dict]] = None


# ---------- Helpers ----------
def index_pdf_to_milvus(
    job_id: str,
    file_path: str,
    minio_object: str | None = None,
    uploaded: bool = True,
    remove_local: bool = True,
    doc_id: str | None = None,
) -> None:
    try:
        job_state.update(job_id, status="parsing", step="parse_pdf:start")
        print(f"[INDEX] start: {file_path}")

        # 1) 페이지별 텍스트 파싱
        pages = parse_pdf(file_path, by_page=True)
        if not pages:
            raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다.")

        # 2) 토큰 기준 청킹
        model = get_embedding_model()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer and hasattr(tokenizer, "encode"):
            encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
        else:
            encode = lambda s: s.split()

        job_state.update(job_id, status="chunking", step="chunk:start", progress=35)
        chunks = smart_chunk_pages(pages, encode)
        if not chunks:
            raise RuntimeError("Chunking 결과가 비었습니다.")
        job_state.update(job_id, status="chunking", step="chunk:done", chunks=len(chunks), progress=50)

        # 3) doc_id 확정 (넘겨받은 값 > MinIO 객체명 > 파일명)
        if not doc_id:
            base_from_obj = os.path.splitext(os.path.basename(minio_object or ""))[0] if minio_object else None
            doc_id = base_from_obj or os.path.splitext(os.path.basename(file_path))[0]

        # 4) Milvus upsert
        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())
        job_state.update(job_id, status="embedding", step="embed:start", progress=60)
        store.insert(doc_id, chunks, embed_fn=embed)
        print(f"[INDEX] done: {file_path} (doc_id={doc_id}, chunks={len(chunks)})")
        job_state.update(job_id, status="indexing", step="milvus:inserted", progress=90)

        # 5) MinIO 원본 삭제는 "이번에 새로 올린(uploaded=True)" 경우에만
        if os.getenv("RAG_DELETE_AFTER_INDEX", "0") == "1" and minio_object and uploaded:
            try:
                MinIOStore().delete(minio_object)
                print(f"[CLEANUP] deleted from MinIO: {minio_object}")
                job_state.update(job_id, status="cleanup", step="minio:deleted", minio_object=minio_object, progress=95)
            except Exception as e:
                print(f"[CLEANUP] delete failed: {e}")
                job_state.update(job_id, status="cleanup", step=f"minio:delete_failed:{e!s}")


        # 6) 로컬 파일 정리 옵션
        if remove_local:
            try:
                os.remove(file_path)
            except Exception:
                pass
        job_state.complete(job_id, pages=len(pages), chunks=len(chunks), doc_id=(doc_id or os.path.basename(file_path)))
    except Exception as e:
        job_state.fail(job_id, str(e))
        raise

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
):
    # 1) 로컬 저장
    safe_name = os.path.basename(file.filename)
    local_path = os.path.join(UPLOAD_DIR, safe_name)
    content = await file.read()
    with open(local_path, "wb") as f:
        f.write(content)

    # 2) 비-PDF면 PDF로 변환
    try:
        pdf_path = convert_to_pdf(local_path)   # 이미 PDF면 그대로 반환
    except ConvertError as e:
        raise HTTPException(400, f"파일 변환 실패: {e}")
    except Exception as e:
        raise HTTPException(500, f"파일 변환 중 예외: {e}")

    # 3) MinIO 업로드 (원본 + 변환본 둘 다 올리고 싶으면 선택)
    minio = MinIOStore()

    # ✅ 기본 키: 파일명 그대로
    pdf_name = os.path.basename(pdf_path)               # 예: doc.pdf
    object_pdf = f"uploaded/{pdf_name}"                 # 예: uploaded/doc.pdf
    uploaded = True

    # 중복 체크: 이름 같고, 사이즈(바이트) 같으면 스킵
    if minio.exists(object_pdf):
        try:
            remote_size = minio.size(object_pdf)
        except Exception:
            remote_size = -1
        local_size = os.path.getsize(pdf_path)

        if remote_size == local_size and remote_size > -1:
            uploaded = False
            print(f"[UPLOAD] dedup hit: {object_pdf} (same name & size)")
        else:
            # 이름은 같지만 사이즈 다르면 충돌 회피용 새 키로 저장
            object_pdf = f"uploaded/{uuid.uuid4().hex}_{pdf_name}"
            minio.upload(pdf_path, object_name=object_pdf, content_type="application/pdf")
            print(f"[UPLOAD] name match but size differs -> stored as: {object_pdf}")
    else:
        minio.upload(pdf_path, object_name=object_pdf, content_type="application/pdf")
        print(f"[UPLOAD] stored: {object_pdf}")

    # doc_id는 오브젝트 파일명(확장자 제외)로 통일하면 충돌 안 남
    doc_id = os.path.splitext(os.path.basename(object_pdf))[0]

    # 인덱싱(백그라운드) - uploaded 플래그 전달
    job_id = uuid.uuid4().hex
    job_state.start(job_id, doc_id=doc_id, minio_object=object_pdf)
    job_state.update(job_id, status="uploaded", step="minio:ok", filename=safe_name, progress=10)
    # 백그라운드 호출 시 'uploaded'와 'doc_id'를 반드시 넘겨야 함
    background_tasks.add_task(index_pdf_to_milvus, pdf_path, object_pdf, uploaded, True, doc_id)


    return UploadResp(
        filename=safe_name,
        minio_object=object_pdf,
        indexed="background",
        job_id=job_id,
    )

@router.post("/ask", response_model=AskResp)
def ask_question(body: AskReq):
    try:
        # 1) 초기 넉넉히 검색 (예: 20)
        model = get_embedding_model()
        store = MilvusStoreV2(dim=model.get_sentence_embedding_dimension())
        cands = store.search(body.question, embed_fn=embed, topk=max(20, body.top_k*5))
        if not cands:
            raise HTTPException(404, "관련 문서를 찾지 못했습니다. 먼저 문서를 업로드/인덱싱 해주세요.")

        # 2) 리랭크 후 상위 k
        topk = rerank(body.question, cands, top_k=body.top_k)

        # 3) 임계값 컷오프(리랭커 스코어 기준, 필요시 조정)
        THRESH = float(os.getenv("RAG_SCORE_THRESHOLD", "0.2"))
        if "re_score" in topk[0] and topk[0]["re_score"] < THRESH:
            return AskResp(answer="문서에서 답을 확실히 찾기 어렵습니다.", used_chunks=0, sources=[])

        # 4) 컨텍스트 + 출처 만들기
        context = ""
        sources = []
        for i, c in enumerate(topk, 1):
            context += f"[{i}] (doc:{c['doc_id']} p.{c['page']} {c['section']})\n{c['chunk']}\n\n"
            sources.append({"id": i, "doc_id": c["doc_id"], "page": c["page"], "section": c["section"]})

        prompt = f"""아래 문서를 참고해서 질문에 정확히 답하세요. 근거가 없으면 모른다고 답합니다.

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

                # 종료 조건
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
        # 너무 길면 일부만 보여줌
        preview = files[:limit_preview]
        more = max(0, matched - len(preview))
        return {
            "status": "dry-run",
            "prefix": prefix,
            "matched": matched,
            "preview": preview,
            "more": more,
        }

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

    return {
        "status": "ok",
        "prefix": prefix,
        "matched": matched,
        "deleted": deleted,
        "failed": failed,
        "errors": errors,
    }
