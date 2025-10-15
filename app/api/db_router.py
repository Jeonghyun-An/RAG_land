# app/api/db_router.py
from fastapi import APIRouter, HTTPException
from app.services.db_connector import DBConnector

router = APIRouter(prefix="/db", tags=["db"])

@router.get("/health")
def health_db():
    try:
        db = DBConnector()
        ok = db.test_connection()
        return {"status": "ok" if ok else "down"}
    except Exception as e:
        raise HTTPException(500, f"DB health check failed: {e}")

@router.get("/files/{data_id}")
def get_file_meta(data_id: str):
    db = DBConnector()
    meta = db.get_file_by_id(data_id)
    if not meta:
        raise HTTPException(404, f"data_id {data_id} not found")
    return meta
