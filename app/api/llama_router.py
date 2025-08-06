# app/api/llama_router.py
from fastapi import APIRouter, UploadFile, File
from app.services.llama_model import load_model, generate_answer

router = APIRouter()

@router.get("/test")
def test():
    return {"status": "LLaMA router is working"}

@router.post("/generate")
def generate(prompt: str):
    model, tokenizer = load_model()
    result = generate_answer(prompt, model, tokenizer)
    return {"response": result}

UPLOAD_DIR = "data"

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {"filename": file.filename}
    