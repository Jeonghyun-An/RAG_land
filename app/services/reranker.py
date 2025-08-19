from typing import List, Dict, Any
import os
MODEL_ID = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
try:
    from sentence_transformers import CrossEncoder
    _ranker = CrossEncoder(MODEL_ID, trust_remote_code=True)
except Exception:
    _ranker = None

def rerank(query: str, candidates: List[Dict[str,Any]], top_k=3):
    if not _ranker or not candidates:
        return candidates[:top_k]
    pairs = [(query, c["chunk"]) for c in candidates]
    scores = _ranker.predict(pairs, convert_to_numpy=True).tolist()
    for c, s in zip(candidates, scores):
        c["re_score"] = float(s)
    candidates.sort(key=lambda x: x["re_score"], reverse=True)
    return candidates[:top_k]
