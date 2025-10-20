# app/services/db_connector.py
"""
CUBRID 데이터베이스 연결 및 작업 처리
- simulate_remote 모드에서는 DB 접근 최소화 및 에러 안전 처리
"""
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Optional, Any, Dict
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
        conn = CUBRIDdb.connect(_dsn(), user=self.user, password=self.password)
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
            try: 
                conn.close()
            except Exception: 
                pass

    def test_connection(self) -> bool:
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                _ = cur.fetchone()
                cur.close()
            return True
        except Exception as e:
            print(f"[DB] Connection test failed: {e}")
            return False

    # ==================== 3. OCR 성공 ====================
    def mark_ocr_success(self, data_id: str | int):
        """OCR 성공 마킹 (simulate_remote에서는 스킵)"""
        try:
            sql = """
            UPDATE data_master
               SET parse_yn = 'S',
                   parse_end_dt = SYS_DATETIME
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                cur.close()
        except Exception as e:
            print(f"[DB] mark_ocr_success failed (simulate_remote?): {e}")

    def insert_ocr_result(self, data_id: str | int, page: int, text: str):
        """OCR 결과 저장 (simulate_remote에서는 스킵)"""
        try:
            sql = """
            MERGE INTO ocr_text A USING DB_ROOT
                ON A.data_id = ? AND A.page = ?
                WHEN MATCHED THEN
                    UPDATE SET
                        text = ?,
                        upt_dt = SYS_DATETIME
                WHEN NOT MATCHED THEN
                    INSERT (data_id, page, text, parse_dt)
                    VALUES (?, ?, ?, SYS_DATETIME)
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id, page, text, data_id, page, text))
                cur.close()
        except Exception as e:
            print(f"[DB] insert_ocr_result failed (simulate_remote?): {e}")

    # ==================== 4. OCR 실패 ====================
    def mark_ocr_failure(self, data_id: str | int, error_msg: str = None):
        """OCR 실패 마킹 (simulate_remote에서는 스킵)"""
        try:
            sql = """
            UPDATE data_master
               SET parse_yn = 'F',
                   parse_end_dt = SYS_DATETIME,
                   rag_index_status = 'error'
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                cur.close()
            
            if error_msg:
                self.insert_ocr_history(data_id, 'F', error_msg)
        except Exception as e:
            print(f"[DB] mark_ocr_failure failed (simulate_remote?): {e}")

    # ==================== 5. OCR 히스토리 ====================
    def insert_ocr_history(self, data_id: str | int, parse_yn: str, error_msg: str = None):
        """OCR 처리 이력 로그 (simulate_remote에서는 스킵)"""
        try:
            sql = """
            INSERT INTO ocr_history (
                data_id,
                parse_yn,
                parse_dt,
                error_msg
            ) VALUES (
                ?,
                ?,
                SYS_DATETIME,
                ?
            )
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id, parse_yn, error_msg))
                cur.close()
        except Exception as e:
            print(f"[DB] insert_ocr_history failed (simulate_remote?): {e}")

    # ==================== RAG 인덱싱 완료 ====================
    def update_rag_completed(self, data_id: str | int, chunks: int | None = None, doc_id: str | None = None):
        """RAG 인덱싱 완료 처리 (simulate_remote에서는 스킵)"""
        try:
            sql = """
            UPDATE data_master
               SET parse_yn = 'S',
                   rag_index_status = 'done',
                   rag_last_indexed_at = SYS_DATETIME,
                   parse_end_dt = SYS_DATETIME,
                   chunk_count = COALESCE(?, chunk_count),
                   milvus_doc_id = COALESCE(?, milvus_doc_id)
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (chunks, doc_id, data_id))
                cur.close()
            
            self.insert_ocr_history(data_id, 'S', None)
        except Exception as e:
            print(f"[DB] update_rag_completed failed (simulate_remote?): {e}")

    # ==================== 기존 메서드 (호환성 유지) ====================
    def update_parse_status(
        self,
        data_id: str | int,
        parse_yn: Optional[str] = None,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        ocr_failed: Optional[bool] = None,
        rag_status: Optional[str] = None,
    ):
        """기존 호환성 유지용 메서드 (simulate_remote에서는 스킵)"""
        try:
            sets = []
            params: list[Any] = []
            
            if parse_yn is not None:
                sets.append("parse_yn=?")
                params.append(parse_yn)
            if start_dt is not None:
                sets.append("parse_start_dt=?")
                params.append(start_dt)
                if rag_status is None: 
                    rag_status = "running"
            if end_dt is not None:
                sets.append("parse_end_dt=?")
                params.append(end_dt)
                if rag_status is None and parse_yn in ('Y', 'S'): 
                    rag_status = "done"
            if ocr_failed is not None:
                sets.append("ocr_failed_yn=?")
                params.append("Y" if ocr_failed else "N")
            if rag_status is not None:
                sets.append("rag_index_status=?")
                params.append(rag_status)

            if not sets:
                return

            sql = f"UPDATE data_master SET {', '.join(sets)} WHERE data_id=?"
            params.append(data_id)
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, tuple(params))
                cur.close()
        except Exception as e:
            print(f"[DB] update_parse_status failed (simulate_remote?): {e}")

    # ==================== MinIO 키 관리 ====================
    def update_simulated_flag(self, data_id: str | int, simulated: bool):
        """simulate_remote 모드 여부 기록 (simulate_remote에서는 스킵)"""
        try:
            sql = "UPDATE data_master SET simulated_yn=? WHERE data_id=?"
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, ('Y' if simulated else 'N', data_id))
                cur.close()
        except Exception as e:
            print(f"[DB] update_simulated_flag failed (simulate_remote?): {e}")

    def update_minio_keys(self, data_id: str | int, 
                          pdf_key: Optional[str] = None, 
                          original_key: Optional[str] = None):
        """MinIO 키 업데이트 (simulate_remote에서는 스킵)"""
        try:
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
        except Exception as e:
            print(f"[DB] update_minio_keys failed (simulate_remote?): {e}")

    def get_simulated_yn(self, data_id: str | int) -> Optional[str]:
        """simulated_yn 조회"""
        try:
            sql = "SELECT simulated_yn FROM data_master WHERE data_id=?"
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                row = cur.fetchone()
                cur.close()
                return row[0] if row else None
        except Exception as e:
            print(f"[DB] get_simulated_yn failed: {e}")
            return None
# ==================== 기본 조회 ====================
    def get_file_by_id(self, data_id: str | int) -> Optional[Dict[str, Any]]:
        """data_master 레코드 조회"""
        try:
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
        except Exception as e:
            print(f"[DB] get_file_by_id failed (simulate_remote?): {e}")
            return None

    # ==================== 1. 파일 변환 시 ====================
    def update_converted_file_path(self, data_id: str | int, file_folder: str, file_id: str):
        """변환된 파일 경로 업데이트 (simulate_remote에서는 스킵)"""
        try:
            sql = """
            UPDATE data_master
               SET file_folder = ?,
                   file_id = ?
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (file_folder, file_id, data_id))
                cur.close()
        except Exception as e:
            print(f"[DB] update_converted_file_path failed (simulate_remote?): {e}")

    # ==================== 2. OCR 시작 ====================
    def mark_ocr_start(self, data_id: str | int):
        """OCR 시작 마킹 (simulate_remote에서는 스킵)"""
        try:
            sql = """
            UPDATE data_master
               SET parse_yn = 'L',
                   parse_start_dt = SYS_DATETIME,
                   rag_index_status = 'running'
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                cur.close()
        except Exception as e:
            print(f"[DB] mark_ocr_start failed (simulate_remote?): {e}")

    #