# app/services/db_connector.py
"""
CUBRID 데이터베이스 연결 및 작업 처리
- osk_data, osk_ocr_data, osk_ocr_hist 테이블 사용
- osk_data_sc 테이블 사용 (SC 문서용)
- 페이지 단위 수정 지원
"""
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Optional, Any, Dict, List, Tuple
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

    # ==================== 1. 파일 변환 시 (doc -> pdf) ====================
    def update_converted_file_path(self, data_id: str | int, file_folder: str, file_id: str):
        """
        파일 변환 시 경로 업데이트
        - file_folder: /COMMON/oskData/ 이후 경로
        - file_id: 파일명
        """
        try:
            sql = """
            UPDATE osk_data
               SET file_folder = ?,
                   file_id = ?
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (file_folder, file_id, data_id))
                cur.close()
            print(f"[DB] ✅ Updated file path for data_id={data_id}: {file_folder}/{file_id}")
        except Exception as e:
            print(f"[DB] ❌ update_converted_file_path failed: {e}")

    # ==================== 2. OCR 추출 시작 ====================
    def mark_ocr_start(self, data_id: str | int):
        """
        OCR 시작 시 상태 업데이트
        - parse_yn = 'L' (Loading)
        - parse_start_dt = SYSDATETIME
        """
        try:
            sql = """
            UPDATE osk_data
               SET parse_yn = 'L',
                   parse_start_dt = SYSDATETIME
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                cur.close()
            print(f"[DB] ✅ Marked OCR start for data_id={data_id} (parse_yn='L')")
        except Exception as e:
            print(f"[DB] ❌ mark_ocr_start failed: {e}")

    # ==================== 3. OCR 성공 ====================
    def mark_ocr_success(self, data_id: str | int):
        """
        OCR 성공 시 상태 업데이트
        - parse_yn = 'S' (Success)
        - parse_end_dt = SYSDATETIME
        """
        try:
            sql = """
            UPDATE osk_data
               SET parse_yn = 'S',
                   parse_end_dt = SYSDATETIME
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                cur.close()
            print(f"[DB] ✅ Marked OCR success for data_id={data_id} (parse_yn='S')")
        except Exception as e:
            print(f"[DB] ❌ mark_ocr_success failed: {e}")

    def insert_ocr_result(self, data_id: str | int, page: int, text: str):
        """
        OCR 결과 저장 (MERGE 쿼리 사용)
        - 기존 데이터가 있으면 UPDATE
        - 없으면 INSERT
        """
        try:
            sql = """
            MERGE INTO osk_ocr_data A USING (SELECT ? AS data_id, ? AS page FROM db_root) B
                ON A.data_id = B.data_id AND A.page = B.page
                WHEN MATCHED THEN
                    UPDATE SET
                        text = ?,
                        parse_dt = SYSDATETIME
                WHEN NOT MATCHED THEN
                    INSERT (data_id, page, text, parse_dt)
                    VALUES (?, ?, ?, SYSDATETIME)
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id, page, text, data_id, page, text))
                cur.close()
            print(f"[DB] ✅ Inserted OCR result: data_id={data_id}, page={page}")
        except Exception as e:
            print(f"[DB] ❌ insert_ocr_result failed: {e}")

    # ==================== 4. OCR 실패 ====================
    def mark_ocr_failure(self, data_id: str | int, error_msg: str = None):
        """
        OCR 실패 시 상태 업데이트
        - parse_yn = 'F' (Failed)
        - parse_end_dt = SYSDATETIME
        """
        try:
            sql = """
            UPDATE osk_data
               SET parse_yn = 'F',
                   parse_end_dt = SYSDATETIME
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                cur.close()
            print(f"[DB] ✅ Marked OCR failure for data_id={data_id} (parse_yn='F')")
            
            # 히스토리 로그
            if error_msg:
                self.insert_ocr_history(data_id, 'F', error_msg)
        except Exception as e:
            print(f"[DB] ❌ mark_ocr_failure failed: {e}")

    # ==================== 5. OCR 히스토리 로그 ====================
    def insert_ocr_history(self, data_id: str | int, parse_yn: str, error_msg: str = None):
        """
        OCR 처리 이력 로그
        - parse_yn: 'S' (성공) 또는 'F' (실패)
        - error_msg: 실패 시 에러 메시지
        """
        try:
            sql = """
            INSERT INTO osk_ocr_hist (
                data_id,
                parse_yn,
                parse_dt,
                error_msg
            ) VALUES (
                ?,
                ?,
                SYSDATETIME,
                ?
            )
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id, parse_yn, error_msg))
                cur.close()
            print(f"[DB] ✅ Logged OCR history: data_id={data_id}, parse_yn={parse_yn}")
        except Exception as e:
            print(f"[DB] ❌ insert_ocr_history failed: {e}")

    # ==================== 6. OCR 텍스트 가져오기 (Manual OCR용) ====================
    def get_ocr_text_by_data_id(self, data_id: str | int) -> List[Tuple[int, str]]:
        """
        data_id에 해당하는 OCR 텍스트를 DB에서 가져오기
        자바가 osk_ocr_data에 저장한 수동 OCR 결과 조회
        
        Args:
            data_id: 문서 ID
        
        Returns:
            [(page_no, text), ...] 형태의 페이지별 텍스트 리스트
            페이지 번호 순으로 정렬됨
        """
        try:
            sql = """
            SELECT page, text
            FROM osk_ocr_data
            WHERE data_id = ?
            ORDER BY page ASC
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                rows = cur.fetchall()
                cur.close()
            
            if not rows:
                print(f"[DB] ⚠️  No OCR text found for data_id={data_id}")
                return []
            
            result = [(int(row[0]), str(row[1] or '').strip()) for row in rows]
            print(f"[DB] ✅ Retrieved {len(result)} pages of OCR text for data_id={data_id}")
            return result
            
        except Exception as e:
            print(f"[DB] ❌ get_ocr_text_by_data_id failed: {e}")
            return []
    
    def get_ocr_page_count(self, data_id: str | int) -> int:
        """
        data_id의 OCR 페이지 수 조회
        
        Returns:
            페이지 수
        """
        try:
            sql = """
            SELECT COUNT(*) 
            FROM osk_ocr_data
            WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                count = cur.fetchone()[0]
                cur.close()
            return int(count)
        except Exception as e:
            print(f"[DB] get_ocr_page_count failed: {e}")
            return 0
    
    def verify_ocr_data_exists(self, data_id: str | int) -> bool:
        """
        OCR 데이터가 DB에 존재하는지 확인
        
        Returns:
            True if exists, False otherwise
        """
        count = self.get_ocr_page_count(data_id)
        return count > 0

    # ==================== 7. 특정 페이지 OCR 텍스트 가져오기 ====================
    def get_ocr_text_by_page(self, data_id: str | int, page: int) -> Optional[str]:
        """
        특정 페이지의 OCR 텍스트 조회
        사용자가 페이지별로 수정한 경우 사용
        
        Args:
            data_id: 문서 ID
            page: 페이지 번호
        
        Returns:
            해당 페이지의 텍스트 (없으면 None)
        """
        try:
            sql = """
            SELECT text
            FROM osk_ocr_data
            WHERE data_id = ? AND page = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id, page))
                row = cur.fetchone()
                cur.close()
            
            if row:
                return str(row[0] or '').strip()
            return None
            
        except Exception as e:
            print(f"[DB] ❌ get_ocr_text_by_page failed: {e}")
            return None

    # ==================== 8. 기본 조회 ====================
    def get_file_by_id(self, data_id: str | int) -> Optional[Dict[str, Any]]:
        """osk_data 레코드 조회"""
        try:
            sql = """
            SELECT data_id, data_title, data_code, data_code_detail, data_code_detail_sub,
                   file_folder, file_id, parse_yn, parse_start_dt, parse_end_dt,
                   link_data_id, main_yn
              FROM osk_data
             WHERE data_id = ?
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
            print(f"[DB] get_file_by_id failed: {e}")
            return None

    # ==================== 9. RAG 인덱싱 완료 (청킹/임베딩 완료) ====================
    def update_rag_completed(self, data_id: str | int):
        """
        RAG 인덱싱 완료 처리
        - parse_yn = 'S'
        - parse_end_dt = SYSDATETIME
        """
        try:
            sql = """
            UPDATE osk_data
               SET parse_yn = 'S',
                   parse_end_dt = SYSDATETIME
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                cur.close()
            
            # 성공 히스토리 로그
            self.insert_ocr_history(data_id, 'S', None)
            print(f"[DB] RAG indexing completed for data_id={data_id}")
        except Exception as e:
            print(f"[DB] update_rag_completed failed: {e}")

    # ==================== 10. RAG 인덱싱 에러 ====================
    def update_rag_error(self, data_id: str | int, error_msg: str):
        """
        RAG 인덱싱 에러 처리
        - parse_yn = 'F'
        - parse_end_dt = SYSDATETIME
        """
        try:
            sql = """
            UPDATE osk_data
               SET parse_yn = 'F',
                   parse_end_dt = SYSDATETIME
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                cur.close()
            
            # 실패 히스토리 로그
            self.insert_ocr_history(data_id, 'F', error_msg)
            print(f"[DB] RAG indexing error logged for data_id={data_id}")
        except Exception as e:
            print(f"[DB] update_rag_error failed: {e}")

    # ==================== 11. SC 문서 조회 (신규 추가) ====================
    def get_sc_document(self, data_id: str | int) -> Optional[Dict[str, Any]]:
        """
        osk_data_sc 테이블에서 SC 문서 조회
        
        Args:
            data_id: 문서 ID
        
        Returns:
            {
                'sc_id': int,
                'data_id': str,
                'preface_text': str,
                'conclusion_text': str,
                'upt_dt': datetime,
                'sc_title': str,
                'reciever_agency': str,
                'sc_code': str,
                'send_date': str,
            }
            문서가 없으면 None 반환
        """
        try:
            sql = """
            SELECT sc_id, data_id, contents_text, conclusion_text, upt_dt,
            sc_title, receiver_agency, sc_code, send_date
            FROM osk_data_sc
            WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (data_id,))
                row = cur.fetchone()
                cols = [d[0] for d in cur.description]
                cur.close()
            
            if not row:
                print(f"[DB] ⚠️  No SC document found for data_id={data_id}")
                return None
            
            result = dict(zip(cols, row))
            print(f"[DB] ✅ Retrieved SC document: sc_id={result.get('sc_id')}, data_id={data_id}")
            return result
            
        except Exception as e:
            print(f"[DB] ❌ get_sc_document failed: {e}")
            return None
    
    def get_sc_combined_text(self, data_id: str | int) -> Optional[str]:
        """
        SC 문서의 preface + contents + conclusion을 하나로 합쳐서 반환
        
        Args:
            data_id: 문서 ID
        
        Returns:
            합쳐진 전체 텍스트 (없으면 None)
        """
        doc = self.get_sc_document(data_id)
        if not doc:
            return None
        
        # 3개 컬럼 합치기 (빈 문자열은 제외)
        parts = []
        
        preface = (doc.get('preface_text') or '').strip()
        if preface:
            parts.append(preface)
        
        contents = (doc.get('contents_text') or '').strip()
        if contents:
            parts.append(contents)
        
        conclusion = (doc.get('conclusion_text') or '').strip()
        if conclusion:
            parts.append(conclusion)
        
        if not parts:
            print(f"[DB] ⚠️  SC document has no text content: data_id={data_id}")
            return None
        
        # 두 줄바꿈으로 구분하여 합치기
        combined = "\n\n".join(parts)
        print(f"[DB] ✅ Combined SC text: {len(combined)} chars from {len(parts)} sections")
        return combined
    
    def get_sc_document_with_structure(self, data_id: str | int) -> Optional[Dict[str, Any]]:
        """
        SC 문서를 구조화된 형태로 반환 (새로운 필드 포함)
        
        Args:
            data_id: 문서 ID
        
        Returns:
            {
                'metadata': {
                    'sc_code': str,           # 문서번호
                    'receiver_agency': str,   # 수신처
                    'sc_title': str,          # 제목
                    'send_date': str,         # 발신일
                    'sender_file_id': str,    # 발신 파일 ID
                    'sc_id': int,
                    'data_id': str,
                    'upt_dt': datetime
                },
                'header': str,                # 문서번호 + 수신처 + 제목 + 발신일 조합
                'contents': str,              # 본문
                'conclusion': str,            # 맺음말
                'full_text': str              # 전체 텍스트 (header + contents + conclusion)
            }
        """
        doc = self.get_sc_document(data_id)
        if not doc:
            return None
        
        # 메타데이터 추출
        metadata = {
            'sc_code': (doc.get('sc_code') or '').strip(),
            'receiver_agency': (doc.get('receiver_agency') or '').strip(),
            'sc_title': (doc.get('sc_title') or '').strip(),
            'send_date': (doc.get('send_date') or '').strip(),
            'sender_file_id': (doc.get('sender_file_id') or '').strip(),
            'sc_id': doc.get('sc_id'),
            'data_id': doc.get('data_id'),
            'upt_dt': doc.get('upt_dt')
        }
        
        # 헤더 구성 (문서번호 + 수신처 + 제목 + 발신일)
        header_parts = []
        
        if metadata['sc_code']:
            header_parts.append(f"문서번호: {metadata['sc_code']}")
        
        if metadata['receiver_agency']:
            header_parts.append(f"수신처: {metadata['receiver_agency']}")
        
        if metadata['sc_title']:
            header_parts.append(f"제목: {metadata['sc_title']}")
        
        if metadata['send_date']:
            header_parts.append(f"발신일: {metadata['send_date']}")
        
        header = "\n".join(header_parts) if header_parts else ""
        
        # 본문 및 맺음말 추출
        contents = (doc.get('contents_text') or '').strip()
        conclusion = (doc.get('conclusion_text') or '').strip()
        
        # 전체 텍스트 조합
        full_text_parts = []
        if header:
            full_text_parts.append(header)
        if contents:
            full_text_parts.append(contents)
        if conclusion:
            full_text_parts.append(conclusion)
        
        full_text = "\n\n".join(full_text_parts)
        
        if not full_text:
            print(f"[DB] ⚠️  SC document has no text content: data_id={data_id}")
            return None
        
        result = {
            'metadata': metadata,
            'header': header,
            'contents': contents,
            'conclusion': conclusion,
            'full_text': full_text
        }
        
        return result

    def update_file_id_only(self, data_id: str | int, new_file_id: str) -> None:
        """
        osk_data.file_id만 변경. file_folder는 변경하지 않음.
        CUBRID에서는 위치 홀더로 '?' 사용.
        """
        try:
            sql = """
            UPDATE osk_data
               SET file_id = ?
             WHERE data_id = ?
            """
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, (new_file_id, data_id))
                cur.close()
            print(f"[DB] ✅ Updated file_id only: data_id={data_id} -> {new_file_id}")
        except Exception as e:
            print(f"[DB] ❌ update_file_id_only failed: {e}")
            
    # ==================== 12. 카테고리별 문서 목록 조회 (신규 추가) ====================
    def fetch_docs_by_code(
        self,
        data_code: str | None = None,
        data_code_detail: str | None = None,
        data_code_detail_sub: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        osk_data 에서 카테고리 기준으로 data_id 목록을 가져온다.
        RAG 대상(parse_yn='S')이면서 삭제되지 않은 문서만 조회.
        
        Args:
            data_code: 대분류 코드
            data_code_detail: 중분류 코드
            data_code_detail_sub: 소분류 코드
        
        Returns:
            [{"data_id": str, "data_title": str, ...}, ...]
        """
        try:
            sql = """
                SELECT data_id, data_title, data_code, data_code_detail, data_code_detail_sub
                FROM osk_data
                WHERE 1=1
            """
            params: List[Any] = []

            # RAG 대상으로 올라간 문서만 필터
            sql += " AND parse_yn = 'S'"
            
            # 삭제되지 않은 문서만 필터
            sql += " AND (del_yn IS NULL OR del_yn != 'Y')"

            if data_code:
                sql += " AND data_code = ?"
                params.append(data_code)

            if data_code_detail:
                sql += " AND data_code_detail = ?"
                params.append(data_code_detail)

            if data_code_detail_sub:
                sql += " AND data_code_detail_sub = ?"
                params.append(data_code_detail_sub)
            
            sql += " ORDER BY data_id"

            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                cur.close()
            
            result = [dict(zip(cols, row)) for row in rows]
            print(f"[DB] ✅ fetch_docs_by_code: {len(result)} documents (code={data_code}, detail={data_code_detail}, sub={data_code_detail_sub})")
            return result
            
        except Exception as e:
            print(f"[DB] ❌ fetch_docs_by_code failed: {e}")
            return []

    # ==================== 호환성 유지 메서드 ====================
    def update_parse_status(
        self,
        data_id: str | int,
        parse_yn: Optional[str] = None,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        ocr_failed: Optional[bool] = None,
        rag_status: Optional[str] = None,
    ):
        """기존 코드 호환성 유지용 (실제로는 위의 메서드들 사용 권장)"""
        try:
            sets = []
            params: list[Any] = []
            
            if parse_yn is not None:
                sets.append("parse_yn=?")
                params.append(parse_yn)
            if start_dt is not None:
                sets.append("parse_start_dt=?")
                params.append(start_dt)
            if end_dt is not None:
                sets.append("parse_end_dt=?")
                params.append(end_dt)

            if not sets:
                return

            sql = f"UPDATE osk_data SET {', '.join(sets)} WHERE data_id=?"
            params.append(data_id)
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, tuple(params))
                cur.close()
        except Exception as e:
            print(f"[DB] update_parse_status failed: {e}")