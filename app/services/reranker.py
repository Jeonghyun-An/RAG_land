# reranker.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple

# -----------------------------
# 환경 변수
# -----------------------------
# 우선순위: 기본은 CE 먼저(설치 지옥 회피), 필요 시 "flag,ce"로 바꾸면 FlagEmbedding 우선
RERANKER_BACKENDS = os.getenv("RERANKER_BACKENDS", "ce,flag").lower()  # 예: "ce,flag" | "flag,ce"

# FlagEmbedding 전용 모델(설치되어 있을 때만 사용)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# CrossEncoder(문자열) 전용 경량 모델(기본값 충분히 성능/속도 괜찮음)
CE_FALLBACK_MODEL = os.getenv("CE_FALLBACK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", None)  # "cuda" | "cpu" | None(자동)
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "64"))


def _pick_reranker_device() -> str:
    if RERANKER_DEVICE:
        return RERANKER_DEVICE
    try:
        import torch  # noqa
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# -----------------------------
# 로더
# -----------------------------
@lru_cache
def _load_ce() -> Tuple[Any, str]:
    """CrossEncoder 우선 로드(가벼움/설치 쉬움)."""
    device = _pick_reranker_device()
    from sentence_transformers import CrossEncoder  # 가벼움
    model_name = CE_FALLBACK_MODEL
    ce = CrossEncoder(model_name, device=device, trust_remote_code=True)
    return ce, "ce"


@lru_cache
def _load_flag() -> Tuple[Any, str]:
    """FlagEmbedding 리랭커 로드(설치되어 있을 때만)."""
    device = _pick_reranker_device()
    from FlagEmbedding import FlagReranker
    use_fp16 = (device == "cuda")
    r = FlagReranker(RERANKER_MODEL, use_fp16=use_fp16, device=device)
    return r, "flag"


@lru_cache
def _load_reranker_impl() -> Tuple[Any | None, str]:
    """백엔드 우선순위에 따라 시도."""
    order = [x.strip() for x in RERANKER_BACKENDS.split(",") if x.strip()]
    if not order:
        order = ["ce", "flag"]

    for backend in order:
        try:
            if backend == "ce":
                return _load_ce()
            if backend == "flag":
                return _load_flag()
        except Exception:
            continue

    return None, "none"


# -----------------------------
# API
# -----------------------------
def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    candidates: [{"chunk": "...", ...}, ...]
    re_score 필드를 채워서 내림차순 정렬 후 top_k 반환.
    로더 실패 시에는 안전 폴백으로 상위 일부 그대로 반환.
    """
    if not candidates:
        return []

    ranker, kind = _load_reranker_impl()
    if ranker is None or kind == "none":
        return candidates[:top_k]

    # (query, chunk) 페어 만들기
    pairs = [(query, c.get("chunk", "") or "") for c in candidates]

    # 점수 계산
    try:
        if kind == "flag":
            scores = ranker.compute_score(pairs, batch_size=RERANK_BATCH_SIZE)
        else:  # "ce"
            try:
                import torch
                with torch.inference_mode():
                    scores = ranker.predict(pairs, convert_to_numpy=True).tolist()
            except Exception:
                scores = ranker.predict(pairs, convert_to_numpy=True).tolist()
    except Exception:
        # 계산 중 실패해도 전체 파이프라인은 살림
        return candidates[:top_k]

    # 주입 & 정렬
    for c, s in zip(candidates, scores):
        try:
            c["re_score"] = float(s)
        except Exception:
            c["re_score"] = 0.0

    candidates.sort(key=lambda x: x.get("re_score", 0.0), reverse=True)
    return candidates[:top_k]


__all__ = ["rerank"]
