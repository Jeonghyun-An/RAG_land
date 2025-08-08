import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.services.chunker import chunk_text
from app.services.file_parser import parse_pdf
from app.services.llama_model import load_model, generate_answer
from app.services.milvus_store import MilvusStore
from app.services.minio_store import MinIOStore

router = APIRouter()
minio_store = MinIOStore()
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/test")
def test():
    return {"status": "LLaMA router is working"}


@router.post("/generate")
def generate(prompt: str, model_name: str = "ko-llama3-luxia-8b"):
    try:
        model, tokenizer = load_model(model_name)
        result = generate_answer(prompt, model, tokenizer)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # 2. MinIO 업로드
        minio_store.upload(file_path,object_name="uploaded/" + file.filename)    

        # 3. 텍스트 → chunk → Milvus 저장
        text = parse_pdf(file_path)
        chunks = chunk_text(text)

        MilvusStore.wait_for_milvus()
        db = MilvusStore()
        db.add_texts(chunks)

        return {"filename": file.filename, "chunks": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 처리 중 오류 발생: {e}")


@router.post("/ask")
def ask_question(question: str, model_name: str = "ko-llama3-luxia-8b"):
    try:
        MilvusStore.wait_for_milvus()
        db = MilvusStore()
        retrieved_chunks = db.search(question, top_k=3)

        context = "\n\n".join(retrieved_chunks)
        prompt = f"""다음 문서를 참고하여 질문에 답하세요:

{context}

질문: {question}
답변:"""

        model, tokenizer = load_model(model_name)
        answer = generate_answer(prompt, model, tokenizer)

        return {"answer": answer}

    except RuntimeError as milvus_error:
        raise HTTPException(status_code=404, detail=str(milvus_error))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files")
def list_files():
    try:
        files = minio_store.list_files()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MinIO 파일 조회 실패: {e}")
