# app/services/job_state.py
"""
Job 상태 관리 모듈
- 기존 llama_router 인터페이스 유지
- 새로운 java_router/dev_router도 지원
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import os

# In-memory 상태 (재시작 시 초기화)
_JOBS: Dict[str, Dict[str, Any]] = {}

STATUS_ORDER = [
    "queued", "uploaded",
    "parsing", "chunking",
    "embedding", "indexing",
    "cleanup",
    "done", "error"
]

PROGRESS_PCT = {
    "queued": 0,
    "uploaded": 10,
    "parsing": 25,
    "chunking": 40,
    "embedding": 70,
    "indexing": 90,
    "cleanup": 95,
    "done": 100,
    "error": 0,
}

def _phase(status: str) -> str:
    if status in ("done",):
        return "done"
    if status in ("error",):
        return "error"
    return "pending" if status in ("queued", "uploaded") else "running"

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _persist(job: Dict[str, Any]) -> None:
    """옵션: MinIO에 상태 JSON으로 저장"""
    if os.getenv("RAG_JOB_STATE_PERSIST", "").lower() != "minio":
        return
    try:
        from app.services.minio_store import MinIOStore
        key = f"jobs/{job['job_id']}.json"
        MinIOStore().put_json(key, job)
    except Exception:
        pass

def start(
    job_id: str, 
    doc_id: Optional[str] = None,  # 기존 llama_router용
    minio_object: Optional[str] = None,  # 기존 llama_router용
    data_id: Optional[str] = None,  # 새 java_router/dev_router용
    file_id: Optional[str] = None  # 새 java_router/dev_router용
) -> None:
    """
    Job 시작 (기존 + 새 인터페이스 모두 지원)
    
    기존: start(job_id, doc_id=..., minio_object=...)
    신규: start(job_id, data_id=..., file_id=...)
    """
    # 호환성: data_id가 없으면 doc_id 사용
    final_doc_id = data_id or doc_id or ""
    final_minio = file_id or minio_object or ""
    
    _JOBS[job_id] = {
        "job_id": job_id,
        "doc_id": final_doc_id,
        "data_id": final_doc_id,  # 양쪽 키 모두 제공
        "minio_object": final_minio,
        "file_id": final_minio,  # 양쪽 키 모두 제공
        "status": "queued",
        "phase": "pending",
        "progress": PROGRESS_PCT["queued"],
        "created_at": _now(),
        "updated_at": _now(),
        "steps": [],
        "metrics": {},
        "error": None,
        "parse_yn": "L",  # java_router용
    }
    _persist(_JOBS[job_id])
    
    # data_id로도 조회 가능하도록 (java_router/dev_router용)
    if final_doc_id and final_doc_id != job_id:
        _JOBS[final_doc_id] = _JOBS[job_id]

def update(job_id: str, status: Optional[str] = None, progress: Optional[int] = None, **fields) -> None:
    """Job 상태 업데이트"""
    j = _JOBS.get(job_id)
    if not j:
        return
    if status:
        j["status"] = status
        j["phase"] = _phase(status)
        if progress is None:
            progress = PROGRESS_PCT.get(status, j.get("progress", 0))
    if progress is not None:
        j["progress"] = max(0, min(100, int(progress)))
    j["updated_at"] = _now()
    if "step" in fields:
        # step 로그는 타임스탬프와 함께 누적
        step = fields.pop("step")
        j["steps"].append({"ts": _now(), "step": step})
    if fields:
        j.update(fields)
    _persist(j)

def complete(job_id: str, pages: Optional[int] = None, chunks: Optional[int] = None, **metrics) -> None:
    """
    Job 완료
    
    기존: complete(job_id, pages=..., chunks=..., doc_id=..., ...)
    신규: complete(job_id, pages=..., chunks=...)
    """
    j = _JOBS.get(job_id)
    if not j:
        return
    j["status"] = "done"
    j["phase"] = "done"
    j["progress"] = 100
    j["updated_at"] = _now()
    j["parse_yn"] = "S"  # Success (java_router용)
    
    # 메트릭 저장
    if pages is not None:
        j["pages"] = pages
        metrics["pages"] = pages
    if chunks is not None:
        j["chunks"] = chunks
        metrics["chunks"] = chunks
    
    if metrics:
        j["metrics"] = metrics
    _persist(j)

def fail(job_id: str, error: str) -> None:
    """
    Job 실패
    
    기존: fail(job_id, message)
    신규: fail(job_id, error)
    
    둘 다 지원
    """
    j = _JOBS.get(job_id)
    if not j:
        return
    j["status"] = "error"
    j["phase"] = "error"
    j["progress"] = 0
    j["updated_at"] = _now()
    j["error"] = error
    j["parse_yn"] = "F"  # Failed (java_router용)
    _persist(j)

def get(identifier: str) -> Optional[Dict[str, Any]]:
    """
    Job 상태 조회 (job_id 또는 doc_id/data_id로)
    """
    return _JOBS.get(identifier)

def list_jobs(status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Job 목록 조회"""
    items = list(_JOBS.values())
    if status:
        items = [x for x in items if x.get("status") == status or x.get("phase") == status]
    items.sort(key=lambda x: x.get("updated_at",""), reverse=True)
    
    # 중복 제거 (job_id와 data_id가 같은 객체를 가리키는 경우)
    seen_ids = set()
    unique_items = []
    for item in items:
        jid = item.get("job_id")
        if jid and jid not in seen_ids:
            seen_ids.add(jid)
            unique_items.append(item)
    
    return unique_items[:limit]

def clear(job_id: str) -> None:
    """Job 상태 삭제"""
    j = _JOBS.get(job_id)
    if j:
        # data_id 매핑도 삭제
        doc_id = j.get("doc_id") or j.get("data_id")
        if job_id in _JOBS:
            del _JOBS[job_id]
        if doc_id and doc_id in _JOBS and doc_id != job_id:
            del _JOBS[doc_id]

def clear_all() -> None:
    """모든 Job 상태 삭제"""
    _JOBS.clear()