# embedding_model.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Tuple

# -----------------------------
# 환경 변수
# -----------------------------
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # 범용 성능/품질 밸런스 좋음
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "128"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# "cuda" | "cpu" | None(자동)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", None)

# 토크나이저 멀티스레딩 경고 억제
os.environ.setdefault("TOKENIZERS_PARALLELISM", os.getenv("TOKENIZERS_PARALLELISM", "false"))

# CPU 점유 과다 방지
EMBED_NUM_THREADS = int(os.getenv("EMBED_NUM_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(EMBED_NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(EMBED_NUM_THREADS))


def _pick_device() -> str:
    if EMBEDDING_DEVICE:
        return EMBEDDING_DEVICE
    try:
        import torch  # noqa
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _limit_threads() -> None:
    try:
        import torch
        torch.set_num_threads(max(1, EMBED_NUM_THREADS))
    except Exception:
        pass


# -----------------------------
# 로더
# -----------------------------
@lru_cache
def _load_embedding_impl() -> Tuple[object, str]:
    """
    1) sentence-transformers (권장)
    2) FlagEmbedding(BGE-M3) (선택)
    반환: (impl, kind)  kind in {"st","flag"}
    """
    device = _pick_device()
    _limit_threads()

    # 1) sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(DEFAULT_MODEL, device=device)
        try:
            m.max_seq_length = EMBED_MAX_TOKENS
        except Exception:
            pass
        if device == "cuda":
            # 가능한 경우만 half로 전환 (대부분 안전)
            try:
                m = m.half()
            except Exception:
                pass
        return m, "st"
    except Exception:
        pass

    # 2) FlagEmbedding (BGEM3)
    try:
        from FlagEmbedding import BGEM3FlagModel
        use_fp16 = (_pick_device() == "cuda")
        m = BGEM3FlagModel(DEFAULT_MODEL, use_fp16=use_fp16, device=_pick_device())
        return m, "flag"
    except Exception as e:
        raise RuntimeError(
            "임베딩 모델 로드 실패: sentence-transformers 또는 FlagEmbedding 중 하나가 필요합니다. "
            "requirements 및 CUDA/드라이버를 확인하세요."
        ) from e


def get_embedding_model():
    model, _ = _load_embedding_impl()
    return model


# -----------------------------
# API
# -----------------------------
def embed(texts: List[str]) -> List[List[float]]:
    """문장/청크 임베딩 반환. normalize=True 고정."""
    if not texts:
        return []
    model, kind = _load_embedding_impl()

    if kind == "st":
        # sentence-transformers 경로
        try:
            if hasattr(model, "max_seq_length"):
                model.max_seq_length = EMBED_MAX_TOKENS
        except Exception:
            pass

        vecs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
        )
        return vecs.tolist()

    # FlagEmbedding 경로
    outs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        max_length=EMBED_MAX_TOKENS,
    )
    vecs = outs["dense_vecs"]
    try:
        return vecs.tolist()
    except Exception:
        return list(vecs)


def get_sentence_embedding_dimension() -> int:
    model, kind = _load_embedding_impl()
    if kind == "st":
        try:
            return model.get_sentence_embedding_dimension()
        except Exception:
            pass
    # FlagEmbedding일 경우
    try:
        dim = getattr(model, "embedding_size", None)
        if isinstance(dim, int):
            return dim
    except Exception:
        pass
    # 안전 폴백(BGE-M3는 1024)
    return 1024
