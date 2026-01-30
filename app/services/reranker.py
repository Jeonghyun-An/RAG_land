# app/services/reranker.py
from __future__ import annotations
import os, traceback
import logging
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Union, Iterable
import numpy as np

logger = logging.getLogger(__name__)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_BACKENDS = [x.strip() for x in os.getenv("RERANKER_BACKENDS", "flag,ce").split(",") if x.strip()]
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda")
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "64"))

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

def ensure_list(scores: Union[float, np.ndarray, List[float], Any]) -> List[float]:
    """FlagReranker/CE의 scores를 항상 List[float]로 보장"""
    if np.isscalar(scores) or isinstance(scores, (float, np.floating)):
        return [float(scores)]
    if isinstance(scores, np.ndarray):
        if scores.ndim == 0:
            return [float(scores)]
        return scores.flatten().tolist()
    if isinstance(scores, list):
        return [float(s) for s in scores]
    raise ValueError(f"Unexpected scores type from reranker: {type(scores)}")

def _score_flag(pairs: List[Tuple[str, str]]) -> List[float]:
    model = _load_flag_reranker()
    if model is None:
        raise RuntimeError("Flag reranker not available")
    raw_scores = model.compute_score(pairs, normalize=True, batch_size=RERANKER_BATCH_SIZE)
    return ensure_list(raw_scores)

def _score_ce(pairs: List[Tuple[str, str]]) -> List[float]:
    model = _load_ce()
    if model is None:
        raise RuntimeError("CE reranker not available")
    raw_scores = model.predict(pairs, batch_size=RERANKER_BATCH_SIZE)
    return ensure_list(raw_scores)


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
            logger.warning(f"[RERANK] Backend {backend} failed: {e}")
            continue
    raise RuntimeError(f"All reranker backends failed: {last_err}")

def rerank(
    query: str,
    cands: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    if not cands:
        return []

    # Early return for trivial cases
    if len(cands) == 0:
        return []
    if len(cands) == 1:
        c = cands[0]
        c["re_score"] = float(c.get("score", 0.5))
        c["re_backend"] = "single_fallback"
        return cands

    pairs = [(query, _strip_meta_line(c.get("chunk") or "")) for c in cands]
    pairs = [p for p in pairs if p[1]]  # 빈 청크 제거

    if not pairs:
        logger.warning("[RERANK] All pairs empty after stripping → fallback to original scores")
        for c in cands:
            c["re_score"] = float(c.get("score", 0.0))
            c["re_backend"] = "fallback_empty"
        cands.sort(key=lambda x: x.get("re_score", -1e9), reverse=True)
        return cands[:max(1, top_k)]

    MAX_BATCH_SIZE = 100

    if len(pairs) > MAX_BATCH_SIZE:
        logger.info(f"[RERANK] Large input ({len(pairs)}), splitting into batches of {MAX_BATCH_SIZE}")
        all_scores = []
        backend = None
        for i in range(0, len(pairs), MAX_BATCH_SIZE):
            batch_pairs = pairs[i:i + MAX_BATCH_SIZE]
            batch_cands = cands[i:i + MAX_BATCH_SIZE]
            try:
                b, batch_scores = _score_pairs(batch_pairs)
                batch_scores = ensure_list(batch_scores)
                if backend is None:
                    backend = b
                all_scores.extend(batch_scores)
            except Exception as e:
                logger.error(f"[RERANK] Batch {i//MAX_BATCH_SIZE + 1} failed: {e} → fallback")
                all_scores.extend([c.get("score", 0.0) for c in batch_cands])
        scores = all_scores
    else:
        try:
            backend, scores = _score_pairs(pairs)
            scores = ensure_list(scores)
        except Exception as e:
            logger.error(f"[RERANK] Full scoring failed: {e} → fallback to embedding scores")
            for c in cands:
                c["re_score"] = float(c.get("score", 0.0))
                c["re_backend"] = "fallback_error"
            cands.sort(key=lambda x: x.get("re_score", -1e9), reverse=True)
            return cands[:max(1, top_k)]

    # 스코어 부착
    for c, s in zip(cands, scores):
        c["re_score"] = float(s)
        c["re_backend"] = backend or "unknown"

    # 내림차순 정렬
    cands.sort(key=lambda x: x.get("re_score", -1e9), reverse=True)

    result = cands[:max(1, top_k)]

    if result:
        top3 = [round(c.get("re_score", 0), 4) for c in result[:3]]
        logger.info(f"[RERANK] Success | backend={backend} | top_k={top_k} | top3_scores={top3} | total={len(cands)}")

    return result

def preload_reranker():
    """
    앱 시작 시 리랭커 모델을 미리 로드
    첫 요청 시 발생하는 모델 로딩 지연 제거
    """
    print("[RERANK] Preloading reranker models...")

    for backend in RERANKER_BACKENDS:
        try:
            if backend == "flag":
                model = _load_flag_reranker()
                if model:
                    print(f"[RERANK] FLAG reranker preloaded")
                    # 테스트 스코어링으로 워밍업
                    try:
                        test_pairs = [("test query", "test document")]
                        _ = model.compute_score(test_pairs, normalize=True)
                        print(f"[RERANK] FLAG reranker warmed up")
                    except Exception as e:
                        print(f"[RERANK] FLAG warmup failed: {e}")

            elif backend == "ce":
                model = _load_ce()
                if model:
                    print(f"[RERANK] CE reranker preloaded")
                    # 테스트 스코어링으로 워밍업
                    try:
                        test_pairs = [("test query", "test document")]
                        _ = model.predict(test_pairs)
                        print(f"[RERANK] CE reranker warmed up")
                    except Exception as e:
                        print(f"[RERANK] CE warmup failed: {e}")

        except Exception as e:
            print(f"[RERANK] Failed to preload {backend}: {e}")

    print("[RERANK] Reranker preload complete")

# 배치 처리 유틸리티
def rerank_in_batches(
    query: str,
    cands: List[Dict[str, Any]],
    top_k: int = 5,
    batch_size: int = None,
) -> List[Dict[str, Any]]:
    """
    대량의 후보를 배치로 나누어 리랭킹
    
    Args:
        query: 검색 쿼리
        cands: 후보 청크 리스트
        top_k: 최종 반환할 상위 개수
        batch_size: 배치 크기 (None이면 RERANKER_BATCH_SIZE 사용)
    
    Returns:
        리랭킹된 상위 top_k개
    """
    if not cands:
        return []

    if batch_size is None:
        batch_size = RERANKER_BATCH_SIZE

    # 청크가 배치 크기보다 작으면 일반 리랭킹
    if len(cands) <= batch_size * 2:
        return rerank(query, cands, top_k)

    # 배치로 나누어 처리
    all_scored = []
    for i in range(0, len(cands), batch_size):
        batch = cands[i : i + batch_size]
        scored_batch = rerank(query, batch, top_k=len(batch))
        all_scored.extend(scored_batch)

    # 전체 재정렬
    all_scored.sort(key=lambda x: x.get("re_score", -1e9), reverse=True)

    return all_scored[: max(1, top_k)]