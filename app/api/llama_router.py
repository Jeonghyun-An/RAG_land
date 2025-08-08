# app/api/llama_router.py
import os
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel

from app.services.chunker import chunk_text
from app.services.file_parser import parse_pdf
from app.services.llama_model import load_model, generate_answer
from app.services.milvus_store import MilvusStore
from app.services.minio_store import MinIOStore

router = APIRouter()

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- Schemas ----------
class GenerateReq(BaseModel):
    prompt: str
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"

class AskReq(BaseModel):
    question: str
    model_name: str ="meta-llama/Llama-3.2-1B-Instruct"
    top_k: int = 3


@router.get("/test")
def test():
    return {"status": "LLaMA router is working"}


@router.post("/generate")
def generate(body: GenerateReq):
    try:
        model, tokenizer = load_model(body.model_name)
        result = generate_answer(body.prompt, model, tokenizer)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 응답 생성 실패: {e}")


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), bg: BackgroundTasks = None):
    # 간단한 확장자 체크 (pdf만 허용)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    minio_store = MinIOStore()

    try:
        # 1) 로컬 저장
        safe_name = os.path.basename(file.filename)
        local_path = os.path.join(UPLOAD_DIR, safe_name)
        content = await file.read()
        with open(local_path, "wb") as f:
            f.write(content)

        # 2) MinIO 업로드 (유니크 오브젝트명)
        object_name = f"uploaded/{uuid.uuid4().hex}_{safe_name}"
        minio_store.upload(local_path, object_name=object_name, content_type="application/pdf")

        # 3) 텍스트 → chunk → Milvus 저장 (무거우면 백그라운드로)
        def _index_pdf(path: str):
            text = parse_pdf(path)
            if not text:
                # 파싱 실패나 빈 문서
                raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다.")
            chunks = chunk_text(text)
            if not chunks:
                raise RuntimeError("Chunking 결과가 비었습니다.")

            MilvusStore.wait_for_milvus()
            db = MilvusStore()
            db.add_texts(chunks)

        if bg is not None:
            # 비동기 인덱싱 (업로드 응답은 즉시 반환)
            bg.add_task(_index_pdf, local_path)
            return {"filename": safe_name, "minio_object": object_name, "indexed": "background"}
        else:
            # 동기 인덱싱 (요청이 끝날 때까지 대기)
            _index_pdf(local_path)
            return {"filename": safe_name, "minio_object": object_name, "indexed": "sync"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 처리 중 오류: {e}")


@router.post("/ask")
def ask_question(body: AskReq):
    try:
        MilvusStore.wait_for_milvus()
        db = MilvusStore()
        retrieved_chunks: List[str] = db.search(body.question, top_k=body.top_k)

        if not retrieved_chunks:
            # 콜드 스타트 등 대비
            raise HTTPException(status_code=404, detail="관련 문서를 찾지 못했습니다. 먼저 문서를 업로드/인덱싱 해주세요.")

        context = "\n\n".join(retrieved_chunks)
        prompt = f"""아래 문서를 참고해서 질문에 정확히 답하세요. 근거가 없으면 모른다고 답합니다.

[문서 발췌]
{context}

[질문]
{body.question}

[답변]
"""

        model, tokenizer = load_model(body.model_name)
        answer = generate_answer(prompt, model, tokenizer)
        return {"answer": answer, "used_chunks": len(retrieved_chunks)}

    except HTTPException:
        raise
    except RuntimeError as milvus_error:
        raise HTTPException(status_code=503, detail=f"Milvus 연결 대기/검색 실패: {milvus_error}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 처리 중 오류: {e}")


@router.get("/files")
def list_files():
    minio_store = MinIOStore()
    try:
        files = minio_store.list_files()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MinIO 파일 조회 실패: {e}")
    
@router.get("/file/{object_name}")
def get_file(object_name: str):
    minio_store = MinIOStore()
    try:
        url = minio_store.presigned_url(object_name, method="GET")
        return {"url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 다운로드 URL 생성 실패: {e}")
    