# app/services/embedding_model.py
import os
from functools import lru_cache
from typing import List

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sbert-nli")
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "128"))  # ← 필요시 .env로 조절

@lru_cache
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers가 필요합니다. "
            "컨테이너에서 `pip install sentence-transformers sentencepiece`로 설치하세요."
        ) from e
    m = SentenceTransformer(DEFAULT_MODEL)
    # 최대 길이 강제(경고 방지용)
    try:
        m.max_seq_length = EMBED_MAX_TOKENS
    except Exception:
        pass
    return m

def embed(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    try:
        if hasattr(model, "max_seq_length"):
            # 모델이 max_seq_length 속성을 지원하면 그걸 사용
            model.max_seq_length = EMBED_MAX_TOKENS
    except Exception:
        pass
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # ← IP 사용 시 cos와 동일하게
        # truncate=True               # ← 길면 안전하게 자르기
    ).tolist()

# (옵션) 다른 모듈에서 편하게 쓰라고 헬퍼 추가
def get_sentence_embedding_dimension() -> int:
    return get_embedding_model().get_sentence_embedding_dimension()
