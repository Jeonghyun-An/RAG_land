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

    # ==================== 기본 조회 ====================
    def get_file_by_id(self, data_id: str | int) -> Optional[Dict[str, Any]]:
        """
        data_master(osk_data) 레코드 조회
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

    # ==================== 1. 파일 변환 시 (doc -> pdf) ====================
    def update_converted_file_path(self, data_id: str | int, file_folder: str, file_id: str):
        """
        변환된 파일 경로 업데이트
        자바 요구사항: file_folder는 /COMMON/oskData/ 이후 경로
        
        예: file_folder = "2023/12/21", file_id = "converted.pdf"
        """
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

    # ==================== 2. OCR 추출 - 시작 ====================
    def mark_ocr_start(self, data_id: str | int):
        """
        OCR 시작: parse_yn='L', parse_start_dt=현재시간
        자바 규격: L = Loading (진행중)
        """
        sql = """
        UPDATE data_master
           SET parse_yn = 'L',
               parse_start_dt = SYS_DATETIME
         WHERE data_id = ?
        """
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, (data_id,))
            cur.close()

    # ==================== 3. OCR 추출 - 성공 ====================
    def mark_ocr_success(self, data_id: str | int):
        """
        OCR 성공: parse_yn='S', parse_end_dt=현재시간
        자바 규격: S = Success
        """
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

    def insert_ocr_result(self, data_id: str | int, page: int, text: str):
        """
        OCR 결과 저장 (MERGE - upsert)
        자바 규격: data_id + page로 upsert
        """
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
            # MERGE 구문: ON 조건 (data_id, page), UPDATE/INSERT 각각
            cur.execute(sql, (data_id, page, text, data_id, page, text))
            cur.close()

    # ==================== 4. OCR 추출 - 실패 ====================
    def mark_ocr_failure(self, data_id: str | int, error_msg: str = None):
        """
        OCR 실패: parse_yn='F', parse_end_dt=현재시간
        자바 규격: F = Failure
        """
        sql = """
        UPDATE data_master
           SET parse_yn = 'F',
               parse_end_dt = SYS_DATETIME
         WHERE data_id = ?
        """
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, (data_id,))
            cur.close()
        
        # 실패 로그 기록
        if error_msg:
            self.insert_ocr_history(data_id, 'F', error_msg)

    # ==================== 5. OCR 히스토리 로그 ====================
    def insert_ocr_history(self, data_id: str | int, parse_yn: str, error_msg: str = None):
        """
        OCR 처리 이력 로그
        자바 규격: osk_ocr_hist 테이블
        
        parse_yn: 'S' (성공) | 'F' (실패) | 'L' (진행중)
        """
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

    # ==================== RAG 인덱싱 완료 (기존 + 확장) ====================
    def update_rag_completed(self, data_id: str | int, chunks: int | None = None, doc_id: str | None = None):
        """
        RAG 인덱싱 완료 처리
        - OCR 성공 처리 포함 (parse_yn='S')
        - chunk_count, milvus_doc_id 업데이트
        """
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
        
        # 성공 로그 기록
        self.insert_ocr_history(data_id, 'S', None)

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
        """
        기존 호환성 유지용 메서드
        새로운 코드에서는 위의 전용 메서드 사용 권장
        """
        sets = []
        params: list[Any] = []
        
        if parse_yn is not None:
            sets.append("parse_yn=?"); params.append(parse_yn)
        if start_dt is not None:
            sets.append("parse_start_dt=?"); params.append(start_dt)
            if rag_status is None: rag_status = "running"
        if end_dt is not None:
            sets.append("parse_end_dt=?"); params.append(end_dt)
            if rag_status is None and parse_yn in ('Y', 'S'): 
                rag_status = "done"
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

    # ==================== MinIO 키 관리 ====================
    def update_simulated_flag(self, data_id: str | int, simulated: bool):
        """
        simulate_remote 모드 여부 기록
        """
        try:
            sql = "UPDATE data_master SET simulated_yn=? WHERE data_id=?"
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, ('Y' if simulated else 'N', data_id))
                cur.close()
        except Exception as e:
            print(f"[DB] simulated_yn 업데이트 실패 (무시): {e}")

    def update_minio_keys(self, data_id: str | int, 
                          pdf_key: Optional[str] = None, 
                          original_key: Optional[str] = None):
        """
        MinIO 키 업데이트
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
        simulated_yn 조회
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