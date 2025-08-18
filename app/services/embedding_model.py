# app/services/embedding_model.py
import os
from functools import lru_cache
from typing import List

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sbert-nli")

@lru_cache
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers가 필요합니다. "
            "컨테이너에서 `pip install sentence-transformers sentencepiece`로 설치하세요."
        ) from e
    return SentenceTransformer(DEFAULT_MODEL)

def embed(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()
