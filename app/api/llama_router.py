# app/api/llama_router.py
import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.services.chunker import chunk_text
from app.services.file_parser import parse_pdf
from app.services.llama_model import load_model, generate_answer
from app.services.milvus_store import MilvusStore

router = APIRouter()

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
UPLOAD_DIR = "data"

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 텍스트 추출 → chunk → Milvus 저장
    text = parse_pdf(file_path)
    chunks = chunk_text(text)
    db = MilvusStore()
    db.add_texts(chunks)

    return {"filename": file.filename, "chunks": len(chunks)}

@router.post("/ask")
def ask_question(question: str, model_name: str = "ko-llama3-luxia-8b"):
    try:
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
