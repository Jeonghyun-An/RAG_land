# app/services/eval_logger.py
from __future__ import annotations
import json, os, datetime
from typing import List, Dict, Any, Optional
from app.services.minio_store import MinIOStore

# 저장 위치: minio "eval/logs/YYYY/MM/DD/qa-<date>.jsonl"
LOG_PREFIX = os.getenv("EVAL_LOG_PREFIX", "eval/logs")

def _today_paths():
    now = datetime.datetime.utcnow()
    y, m, d = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")
    fname = f"qa-{y}{m}{d}.jsonl"
    key = f"{LOG_PREFIX}/{y}/{m}/{d}/{fname}"
    return key

def strip_meta_line(txt: str) -> str:
    if not txt: return ""
    if txt.startswith("META:"):
        nl = txt.find("\n")
        return txt[nl+1:].strip() if nl != -1 else ""
    return txt.strip()

def to_contexts_from_sources(sources: List[Dict[str, Any]], max_ctx: int = 10) -> List[str]:
    ctxs = []
    for s in sources[:max_ctx]:
        ck = s.get("chunk") or ""
        ctxs.append(strip_meta_line(ck))
    # 빈 문자열 제거
    return [c for c in ctxs if c]

def log_qa_event(
    question: str,
    answer: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    ground_truths: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    RAGAS 포맷에 맞춰 1행 JSONL append:
    {question, answer, contexts: [..], ground_truths: [...], meta: {...}}
    """
    m = MinIOStore()
    key = _today_paths()
    rec = {
        "question": question or "",
        "answer": answer or "",
        "contexts": to_contexts_from_sources(sources or []),
        "ground_truths": ground_truths or [],   # 운영로그는 보통 []
        "meta": meta or {},
    }
    line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
    # MinIO에는 append가 없어 안전한 "read+write" 방식으로 덮어쓰기
    try:
        if m.exists(key):
            prev = m.get_bytes(key)
            m.upload_bytes(prev + line, key, content_type="application/jsonl", length=len(prev)+len(line))
        else:
            m.upload_bytes(line, key, content_type="application/jsonl", length=len(line))
    except Exception as e:
        # 로깅 실패는 서비스에 영향 주지 않도록 무시
        print(f"[EVAL_LOG] warn: {e}")
    return key
