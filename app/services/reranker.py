# app/services/reranker.py
from __future__ import annotations

import os
import traceback
from functools import lru_cache
from typing import Any, Dict, List, Tuple

# -----------------------------
# 환경 변수
# -----------------------------
RERANKER_BACKENDS = os.getenv("RERANKER_BACKENDS", "ce,flag").lower()  # "ce,flag" | "flag,ce"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
CE_FALLBACK_MODEL = os.getenv("CE_FALLBACK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# "cuda" | "cpu" | "auto"(기본)
RERANKER_DEVICE_ENV = (os.getenv("RERANKER_DEVICE") or "auto").lower()

def _get_batch_env() -> int:
    # 둘 다 지원 (오탈자/기존값 호환)
    v = os.getenv("RERANKER_BATCH_SIZE") or os.getenv("RERANK_BATCH_SIZE") or "64"
    try:
        return max(1, int(v))
    except Exception:
        return 64

RERANKER_BATCH_SIZE = _get_batch_env()


def _pick_device() -> str:
    if RERANKER_DEVICE_ENV in ("cuda", "cpu"):
        if RERANKER_DEVICE_ENV == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
            print("[RERANK] Requested CUDA but not available → fallback to CPU")
            return "cpu"
        return "cpu"
    # auto
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# -----------------------------
# 로더
# -----------------------------
@lru_cache
def _load_ce() -> Tuple[Any, str]:
    """CrossEncoder 경량 리랭커 (설치 쉬움, 기본값 우선)."""
    device = _pick_device()
    from sentence_transformers import CrossEncoder
    print(f"[RERANK] Loading CE model='{CE_FALLBACK_MODEL}' device='{device}'")
    ce = CrossEncoder(CE_FALLBACK_MODEL, device=device, trust_remote_code=True)
    return ce, "ce"


@lru_cache
def _load_flag() -> Tuple[Any, str]:
    """FlagEmbedding 리랭커 (설치된 경우에만)."""
    device = _pick_device()
    from FlagEmbedding import FlagReranker
    use_fp16 = (device == "cuda")
    print(f"[RERANK] Loading FlagReranker model='{RERANKER_MODEL}' device='{device}' fp16={use_fp16}")
    r = FlagReranker(RERANKER_MODEL, use_fp16=use_fp16, device=device)
    return r, "flag"


@lru_cache
def _load_reranker_impl() -> Tuple[Any | None, str]:
    order = [x.strip() for x in RERANKER_BACKENDS.split(",") if x.strip()] or ["ce", "flag"]

    for backend in order:
        try:
            if backend == "flag":
                try:
                    import importlib
                    if importlib.util.find_spec("FlagEmbedding") is None:
                        raise ModuleNotFoundError("FlagEmbedding not installed")
                except Exception as e:
                    print(f"[RERANK] FlagEmbedding not available: {e}")
                    continue
                return _load_flag()
            if backend == "ce":
                return _load_ce()
        except Exception as e:
            print(f"[RERANK] backend '{backend}' load failed: {e}\n{traceback.format_exc()}")
            continue

    print("[RERANK] No reranker available, using None")
    return None, "none"


# -----------------------------
# API
# -----------------------------
def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    candidates: [{"chunk": "...", ...}, ...]
    re_score 필드를 채워서 내림차순 정렬 후 top_k 반환.
    로더 실패/계산 실패 시에는 안전 폴백으로 상위 일부 그대로 반환.
    """
    if not candidates:
        return []

    ranker, kind = _load_reranker_impl()
    if ranker is None or kind == "none":
        return candidates[:top_k]

    pairs = [(query, c.get("chunk", "") or "") for c in candidates]

    try:
        if kind == "flag":
            scores = ranker.compute_score(pairs, batch_size=RERANKER_BATCH_SIZE)
        else:  # "ce"
            try:
                import torch
                with torch.inference_mode():
                    scores = ranker.predict(pairs, convert_to_numpy=True).tolist()
            except Exception:
                scores = ranker.predict(pairs, convert_to_numpy=True).tolist()
    except Exception as e:
        print(f"[RERANK] scoring failed: {e}\n{traceback.format_exc()}")
        return candidates[:top_k]

    for c, s in zip(candidates, scores):
        try:
            c["re_score"] = float(s)
        except Exception:
            c["re_score"] = 0.0

    candidates.sort(key=lambda x: x.get("re_score", 0.0), reverse=True)
    return candidates[:top_k]


__all__ = ["rerank"]
