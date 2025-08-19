# app/services/job_state.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import os
import json

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
    # 그 외는 UI에선 pending/running으로 묶여 보일 수 있음
    return "pending" if status in ("queued", "uploaded") else "running"

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _persist(job: Dict[str, Any]) -> None:
    """옵션: MinIO에 상태 JSON으로 저장 (RAG_JOB_STATE_PERSIST=minio)"""
    if os.getenv("RAG_JOB_STATE_PERSIST", "").lower() != "minio":
        return
    try:
        from app.services.minio_store import MinIOStore
        key = f"jobs/{job['job_id']}.json"
        MinIOStore().put_json(key, job)
    except Exception:
        # 상태 저장 실패는 무시(로그만 남기고 싶다면 여기서 print)
        pass

def start(job_id: str, doc_id: str, minio_object: str) -> None:
    _JOBS[job_id] = {
        "job_id": job_id,
        "doc_id": doc_id,
        "minio_object": minio_object,
        "status": "queued",
        "phase": "pending",
        "progress": PROGRESS_PCT["queued"],
        "created_at": _now(),
        "updated_at": _now(),
        "steps": [],
        "metrics": {},
        "error": None,
    }
    _persist(_JOBS[job_id])

def update(job_id: str, status: Optional[str] = None, progress: Optional[int] = None, **fields) -> None:
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

def complete(job_id: str, **metrics) -> None:
    j = _JOBS.get(job_id)
    if not j:
        return
    j["status"] = "done"
    j["phase"] = "done"
    j["progress"] = 100
    j["updated_at"] = _now()
    if metrics:
        j["metrics"] = metrics
    _persist(j)

def fail(job_id: str, message: str) -> None:
    j = _JOBS.get(job_id)
    if not j:
        return
    j["status"] = "error"
    j["phase"] = "error"
    j["progress"] = 0
    j["updated_at"] = _now()
    j["error"] = message
    _persist(j)

def get(job_id: str) -> Optional[Dict[str, Any]]:
    return _JOBS.get(job_id)

def list_jobs(status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    items = list(_JOBS.values())
    if status:
        items = [x for x in items if x.get("status") == status or x.get("phase") == status]
    items.sort(key=lambda x: x.get("updated_at",""), reverse=True)
    return items[:limit]
