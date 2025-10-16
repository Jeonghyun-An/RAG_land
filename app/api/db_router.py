from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.services.db_connector import DBConnector
import os, logging

router = APIRouter(prefix="/db", tags=["db"])

# ===== 1) 내부 토큰(선택) =====
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")  # compose에 넣으면 활성화

def require_internal_token(x_internal_token: Optional[str] = Header(None)) -> None:
    if INTERNAL_API_TOKEN and x_internal_token != INTERNAL_API_TOKEN:
        raise HTTPException(401, "unauthorized")

# ===== 2) 응답 스키마 =====
class DBHealth(BaseModel):
    status: str
    db_date: Optional[str] = None

class FileMeta(BaseModel):
    data_id: str
    data_title: Optional[str] = None
    data_code: Optional[str] = None
    data_code_detail: Optional[str] = None
    data_code_detail_sub: Optional[str] = None
    file_folder: Optional[str] = None
    file_id: Optional[str] = None
    parse_yn: Optional[str] = None
    ocr_failed_yn: Optional[str] = None
    minio_pdf_key: Optional[str] = None
    converted_pdf_path: Optional[str] = None
    rag_index_status: Optional[str] = None
    rag_last_indexed_at: Optional[str] = None
    chunk_count: Optional[int] = None
    milvus_doc_id: Optional[str] = None

# ===== 3) 엔드포인트 =====
@router.get("/health", response_model=DBHealth)
def health_db(_: None = Depends(require_internal_token)):
    try:
        db = DBConnector()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT CURRENT_DATE")
            row = cur.fetchone()
            cur.close()
        return DBHealth(status="ok", db_date=row[0] if row else None)
    except Exception:
        logging.exception("DB health check failed")
        # 내부 예외 내용은 숨김
        raise HTTPException(500, "DB health check failed")

@router.get("/files/{data_id}", response_model=FileMeta)
def get_file_meta(data_id: str, _: None = Depends(require_internal_token)):
    db = DBConnector()
    meta = db.get_file_by_id(data_id)
    if not meta:
        raise HTTPException(404, f"data_id {data_id} not found")
    return FileMeta(**meta)

@router.get("/status/{data_id}")
def get_status(data_id: str, _: None = Depends(require_internal_token)):
    db = DBConnector()
    sql = """
      SELECT data_id, parse_yn, ocr_failed_yn, rag_index_status,
             parse_start_dt, parse_end_dt, rag_last_indexed_at, chunk_count
        FROM data_master WHERE data_id=?
    """
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, (data_id,))
        row = cur.fetchone()
        cols = [c[0] for c in cur.description] if cur.description else []
        cur.close()
    if not row:
        raise HTTPException(404, f"data_id {data_id} not found")
    return dict(zip(cols, row))

@router.get("/ocr/pages/{data_id}")
def get_ocr_pages(
    data_id: str,
    limit: int = 10,
    offset: int = 0,
    _: None = Depends(require_internal_token),
):
    db = DBConnector()
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT page, text, parse_dt, upt_dt "
            "FROM ocr_text WHERE data_id=? ORDER BY page ASC "
            "LIMIT ? OFFSET ?",
            (data_id, limit, offset),
        )
        rows = cur.fetchall()
        cur.close()
    return [{"page": r[0], "text": r[1], "parse_dt": r[2], "upt_dt": r[3]} for r in rows]
