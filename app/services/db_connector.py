# app/services/db_connector.py
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Optional, Any, Dict, Tuple
import CUBRIDdb
from datetime import datetime

def _dsn() -> str:
    host = os.getenv("CUBRID_HOST", "211.219.26.15")
    port = os.getenv("CUBRID_PORT", "44000")
    db   = os.getenv("CUBRID_DB", "nuclear")
    return f"CUBRID:{host}:{port}:{db}:::"

class DBConnector:
    def __init__(self):
        self.user = os.getenv("CUBRID_USER", "nuclear")
        self.password = os.getenv("CUBRID_PASSWORD", "nuclear13!#")

    def connect(self):
        # autocommit 기본 OFF → commit 필요
        conn = CUBRIDdb.connect(_dsn(), user=self.user, password=self.password)
        # 문자셋 이슈 예방
        try:
            cur = conn.cursor()
            cur.execute("SET NAMES utf8")
            cur.close()
        except Exception:
            pass
        return conn

    @contextmanager
    def get_conn(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        finally:
            try: conn.close()
            except Exception: pass

    def test_connection(self) -> bool:
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            _ = cur.fetchone()
            cur.close()
        return True

    # ---- 도메인 메서드 ----
    def get_file_by_id(self, data_id: str | int) -> Optional[Dict[str, Any]]:
        """
        data_master 레코드 조회
        ✅ 추가: simulated_yn, converted_path 컬럼 (선택)
        """
        sql = """
        SELECT data_id, data_title, data_code, data_code_detail, data_code_detail_sub,
               file_folder, file_id, parse_yn, ocr_failed_yn,
               minio_pdf_key, minio_original_key, rag_index_status,
               chunk_count, parse_start_dt, parse_end_dt, milvus_doc_id
          FROM data_master
         WHERE data_id=?
        """
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, (data_id,))
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            cur.close()
        if not row:
            return None
        return dict(zip(cols, row))

    def update_parse_status(
        self,
        data_id: str | int,
        parse_yn: Optional[str] = None,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        ocr_failed: Optional[bool] = None,
        rag_status: Optional[str] = None,
    ):
        """
        파싱 상태 업데이트
        ✅ rag_status: "queued" | "running" | "done" | "error" | "self_ocr_required"
        """
        sets = []
        params: list[Any] = []
        if parse_yn is not None:
            sets.append("parse_yn=?"); params.append(parse_yn)
        if start_dt is not None:
            sets.append("parse_start_dt=?"); params.append(start_dt)
            # 진행중 표시
            if rag_status is None: rag_status = "running"
        if end_dt is not None:
            sets.append("parse_end_dt=?"); params.append(end_dt)
            # 완료/에러는 호출부에서 넘겨도 되고 여기서 Y면 done으로 표시
            if rag_status is None and parse_yn == "Y": rag_status = "done"
        if ocr_failed is not None:
            sets.append("ocr_failed_yn=?"); params.append("Y" if ocr_failed else "N")
        if rag_status is not None:
            sets.append("rag_index_status=?"); params.append(rag_status)

        if not sets:
            return

        sql = f"UPDATE data_master SET {', '.join(sets)} WHERE data_id=?"
        params.append(data_id)
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            cur.close()

    def insert_ocr_result(self, data_id: str | int, page: int, text: str,
                          head: str | None = None, middle: str | None = None, tail: str | None = None):
        """
        OCR 텍스트 저장 (upsert 유사)
        """
        # upsert 유사 로직
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT idx FROM ocr_text WHERE data_id=? AND page=?", (data_id, page))
            row = cur.fetchone()
            if row:
                cur.execute("""
                    UPDATE ocr_text
                       SET text=?, upt_dt=SYS_DATETIME, text_sc_head=?, text_sc_middle=?, text_sc_end=?
                     WHERE data_id=? AND page=?""",
                    (text, head, middle, tail, data_id, page))
            else:
                cur.execute("""
                    INSERT INTO ocr_text (data_id, page, text, parse_dt, text_sc_head, text_sc_middle, text_sc_end)
                    VALUES (?, ?, ?, SYS_DATETIME, ?, ?, ?)""",
                    (data_id, page, text, head, middle, tail))
            cur.close()

    def update_rag_completed(self, data_id: str | int, chunks: int | None = None, doc_id: str | None = None):
        """
        RAG 인덱싱 완료 처리
        ✅ chunk_count, milvus_doc_id 업데이트
        """
        sql = """
        UPDATE data_master
           SET parse_yn='Y',
               rag_index_status='done',
               rag_last_indexed_at=SYS_DATETIME,
               parse_end_dt=SYS_DATETIME,
               chunk_count=COALESCE(?, chunk_count),
               milvus_doc_id=COALESCE(?, milvus_doc_id)
         WHERE data_id=?
        """
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, (chunks, doc_id, data_id))
            cur.close()

    def update_simulated_flag(self, data_id: str | int, simulated: bool):
        """
        ✅ 신규: simulate_remote 모드 여부 기록
        컬럼이 없으면 예외 무시
        """
        try:
            sql = "UPDATE data_master SET simulated_yn=? WHERE data_id=?"
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, ('Y' if simulated else 'N', data_id))
                cur.close()
        except Exception as e:
            # 컬럼이 없거나 기타 오류 시 무시
            print(f"[DB] simulated_yn 업데이트 실패 (무시): {e}")

    def update_minio_keys(self, data_id: str | int, 
                          pdf_key: Optional[str] = None, 
                          original_key: Optional[str] = None):
        """
        ✅ 신규: MinIO 키 업데이트
        """
        sets = []
        params: list[Any] = []
        
        if pdf_key is not None:
            sets.append("minio_pdf_key=?")
            params.append(pdf_key)
        
        if original_key is not None:
            sets.append("minio_original_key=?")
            params.append(original_key)
        
        if not sets:
            return
        
        sql = f"UPDATE data_master SET {', '.join(sets)} WHERE data_id=?"
        params.append(data_id)
        
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            cur.close()

    def get_simulated_yn(self, data_id: str | int) -> Optional[str]:
        """
        ✅ 신규: simulated_yn 조회
        컬럼이 없으면 None 반환
        """
        try:
            sql = "SELECT simulated_yn FROM data_master WHERE data_id=?"
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                row = cur.fetchone()
                cur.close()
                return row[0] if row else None
        except Exception:
            return None