# app/main.py
from fastapi import FastAPI
from app.api.llama_router import router as llama_router

app = FastAPI()
app.include_router(llama_router, prefix="/llama")

@app.get("/")
def root():
    return {"message": "LLaMA RAG Chatbot is running"}
