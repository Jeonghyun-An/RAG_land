# app/services/embedding_model.py
import os
from sentence_transformers import SentenceTransformer

# 전역 모델 인스턴스
embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "jhgan/ko-sbert-nli"))
