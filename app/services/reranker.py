# app/services/reranker.py
from __future__ import annotations
import os, traceback
from functools import lru_cache
from typing import List, Dict, Any, Tuple

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_BACKENDS = [x.strip() for x in os.getenv("RERANKER_BACKENDS", "flag,ce").split(",") if x.strip()]
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "32"))

def _strip_meta_line(s: str) -> str:
    if not s: return ""
    # 필요 시 chunk 앞 메타 제거 로직을 재사용
    return s.strip()

@lru_cache
def _load_flag_reranker():
    try:
        from FlagEmbedding import FlagReranker
        use_fp16 = (RERANKER_DEVICE == "cuda")
        print(f"[RERANK] Loading FLAG model='{RERANKER_MODEL}' device='{RERANKER_DEVICE}' fp16={use_fp16}")
        model = FlagReranker(RERANKER_MODEL, use_fp16=use_fp16, device=RERANKER_DEVICE)
        return model
    except Exception as e:
        print(f"[RERANK] Flag load failed: {e}\n{traceback.format_exc()}")
        return None

@lru_cache
def _load_ce():
    try:
        from sentence_transformers import CrossEncoder
        print(f"[RERANK] Loading CE model='{RERANKER_MODEL}' device='{RERANKER_DEVICE}'")
        model = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)
        return model
    except Exception as e:
        print(f"[RERANK] CE load failed: {e}\n{traceback.format_exc()}")
        return None

def _score_flag(pairs: List[Tuple[str, str]]) -> List[float]:
    model = _load_flag_reranker()
    if model is None:
        raise RuntimeError("Flag reranker not available")
    # normalize=True → 0~1 근처 점수
    return model.compute_score(pairs, normalize=True, batch_size=RERANKER_BATCH_SIZE)

def _score_ce(pairs: List[Tuple[str, str]]) -> List[float]:
    model = _load_ce()
    if model is None:
        raise RuntimeError("CE reranker not available")
    # CE는 자유 스케일(음수/양수 혼재). 그대로 사용하고 컷오프에서 emb 백업 조건 병행.
    return model.predict(pairs, batch_size=RERANKER_BATCH_SIZE).tolist()

def _score_pairs(pairs: List[Tuple[str, str]]) -> Tuple[str, List[float]]:
    last_err = None
    for backend in RERANKER_BACKENDS:
        try:
            if backend == "flag":
                return "flag", _score_flag(pairs)
            if backend == "ce":
                return "ce", _score_ce(pairs)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"reranker backends failed: {last_err}")

def rerank(query: str, cands: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    if not cands:
        return []

    pairs = [(query, _strip_meta_line(c.get("chunk") or "")) for c in cands]
    backend, scores = _score_pairs(pairs)

    # 점수 부착
    for c, s in zip(cands, scores):
        c["re_score"] = float(s)
        c["re_backend"] = backend

    # 내림차순 정렬
    cands.sort(key=lambda x: x.get("re_score", -1e9), reverse=True)
    return cands[:max(1, top_k)]

def preload_reranker():
    """
    앱 시작 시 리랭커 모델을 미리 로드합니다.
    첫 요청 시 발생하는 모델 로딩 지연을 제거합니다.
    """
    for backend in RERANKER_BACKENDS:
        try:
            if backend == "flag":
                model = _load_flag_reranker()
                if model:
                    print(f"[RERANK] FLAG reranker preloaded")
            elif backend == "ce":
                model = _load_ce()
                if model:
                    print(f"[RERANK] CE reranker preloaded")
        except Exception as e:
            print(f"[RERANK] Failed to preload {backend}: {e}")
    
    print("[RERANK] Reranker preload complete")