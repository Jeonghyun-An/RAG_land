# app/services/milvus_store_v2.py
"""
Milvus 벡터 스토어 V2 - 개선 버전
주요 개선사항:
1. 동적 EF 파라미터 적용 (환경변수 반영)
2. 검색 성능 최적화
3. 통계 정보 개선
4. 기존 모든 기능 유지
"""
from __future__ import annotations

import os
import re
from typing import List, Tuple, Dict, Any, Callable, Optional

from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    MilvusException,
)

# 컬렉션 이름(환경변수로 덮어쓰기 가능)
COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks_v2")
SECTION_MAX = int(os.getenv("MILVUS_SECTION_MAX", "512"))
DOC_ID_MAX  = int(os.getenv("MILVUS_DOCID_MAX",  "256"))
CHUNK_MAX   = int(os.getenv("MILVUS_CHUNK_MAX",  "8192"))

# HNSW 인덱스 파라미터
INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "IP")
HNSW_M = int(os.getenv("MILVUS_HNSW_M", "16"))
HNSW_EFCON = int(os.getenv("MILVUS_HNSW_EFCON", "200"))

# 동적 검색 파라미터
EF_SEARCH = int(os.getenv("MILVUS_EF_SEARCH", "512"))      # 기본 ef
EF_PER_K = int(os.getenv("MILVUS_EF_PER_K", "10"))          # topk당 ef 증가량
EF_MAX = int(os.getenv("MILVUS_EF_MAX", "2048"))           # ef 최대값


def _safe_truncate_text(text: str, max_len: int) -> str:
    """
    텍스트를 UTF-8 바이트 길이 기준으로 안전하게 자르기.
    문자 수가 아닌 바이트 수를 기준으로 하여 Milvus VARCHAR 제한을 절대 넘지 않음.
    """
    if not text:
        return ""
    
    # 1단계: 바이트 길이 체크
    encoded = text.encode('utf-8', errors='ignore')
    if len(encoded) <= max_len:
        return text
    
    # 2단계: 바이트 기준으로 잘라야 함
    # 안전 마진 (10% 여유)
    target_bytes = int(max_len * 0.9)
    
    # 바이트 단위로 자르되, UTF-8 문자 경계를 지킴
    truncated_bytes = encoded[:target_bytes]
    
    # UTF-8 디코딩 시도 (불완전한 멀티바이트 문자 제거)
    try:
        truncated = truncated_bytes.decode('utf-8', errors='ignore')
    except:
        # 혹시 모를 에러에 대비
        truncated = text[:int(max_len * 0.5)]
    
    # 3단계: 문장 경계에서 자르기 시도
    if len(truncated) > target_bytes * 0.5:  # 절반 이상 남아있으면
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?'),
            truncated.rfind('。'),
        )
        if last_sentence_end > len(truncated) * 0.5:
            truncated = truncated[:last_sentence_end + 1]
    
    # 4단계: 최종 바이트 길이 검증 및 강제 자르기
    final_encoded = truncated.encode('utf-8', errors='ignore')
    while len(final_encoded) > max_len and truncated:
        # 10% 씩 줄이기
        cut_point = int(len(truncated) * 0.9)
        truncated = truncated[:cut_point]
        final_encoded = truncated.encode('utf-8', errors='ignore')
    
    return truncated.strip()


def _vmax(field):
    """
    Milvus 2.2.x는 field.params 형태가 다를 수 있어 방어적으로 추출
    - field.params.get("max_length")
    - field.params.get("type_params", {}).get("max_length")
    - getattr(field, "max_length", 0)
    """
    try:
        p = getattr(field, "params", {}) or {}
        if isinstance(p, dict):
            if "max_length" in p:
                return int(p["max_length"])
            tp = p.get("type_params") or {}
            if "max_length" in tp:
                return int(tp["max_length"])
    except Exception:
        pass
    try:
        return int(getattr(field, "max_length", 0) or 0)
    except Exception:
        return 0
    

def _get_schema_limits(col: Collection) -> dict:
    """컬렉션 스키마에서 VARCHAR 필드의 실제 max_length 추출"""
    f = {x.name: x for x in col.schema.fields}
    return {
        "doc_id":  _vmax(f.get("doc_id"))  or DOC_ID_MAX,
        "section": _vmax(f.get("section")) or SECTION_MAX,
        "chunk":   _vmax(f.get("chunk"))   or CHUNK_MAX,
    }


class MilvusStoreV2:
    """
    메타데이터가 포함된 RAG용 Milvus 스토어(V2)
      - 스키마:
          id (INT64, auto_id, primary)
          doc_id (VARCHAR 256)
          seq (INT64)
          page (INT64)
          section (VARCHAR 512)
          chunk (VARCHAR 8192)
          embedding (FLOAT_VECTOR dim={dim})
      - 인덱스: HNSW + IP (Milvus 2.2.x 에서 COSINE 미지원 → IP 사용)
      - 임베딩은 normalize_embeddings=True로 인코딩하여 IP == cosine로 동작
      
    개선사항:
      - 동적 EF 파라미터 적용 (topk에 따라 자동 조절)
      - 검색 성능 로깅 강화
    """

    def __init__(self, dim: int, name: Optional[str] = None):
        self.dim = int(dim)
        self.collection_name = name or COLLECTION

        host = os.getenv("MILVUS_HOST", "milvus")
        port = os.getenv("MILVUS_PORT", "19530")
        if not connections.has_connection("default"):
            connections.connect(alias="default", host=host, port=port)

        force_reset = os.getenv("RAG_RESET_COLLECTION", "0") == "1"

        # 컬렉션 존재 여부 확인
        if utility.has_collection(self.collection_name):
            col = Collection(self.collection_name)
            # 스키마/차원 불일치 시 재생성 or 에러
            if force_reset or self._schema_mismatch(col, self.dim):
                print(f"⚠️ drop & recreate collection: {self.collection_name} (force_reset={force_reset})")
                utility.drop_collection(self.collection_name)
                self.col = self._create_collection()
            else:
                self.col = col
                self._ensure_index()
        else:
            self.col = self._create_collection()

        # load 시도 (인덱스 없거나 비어있을 수 있으므로 예외 무시)
        try:
            self.col.load()
        except MilvusException as e:
            print(f"load skipped: {e}")

    # ---------------- internal ----------------

    def _schema_mismatch(self, col: Collection, expect_dim: int) -> bool:
        """컬렉션 스키마(특히 embedding dim) 불일치 시 True"""
        try:
            fdict = {f.name: f for f in col.schema.fields}
            # 필수 필드 체크
            required = ("doc_id", "seq", "page", "section", "chunk", "embedding")
            if any(r not in fdict for r in required):
                return True
            emb = fdict["embedding"]
            # dim 읽기 (버전에 따라 params 또는 속성)
            emb_dim = emb.params.get("dim") if hasattr(emb, "params") else getattr(emb, "dim", None)
            if int(emb_dim or 0) != int(expect_dim):
                return True
            
            def vmax(field):
                try:
                    return int(field.params.get("max_length"))
                except Exception:
                    try:
                        return int(getattr(field, "max_length", 0))
                    except Exception:
                        return 0
            
            # 환경변수 기준으로 스키마 불일치 체크
            current_doc_id_max = vmax(fdict["doc_id"])
            current_section_max = vmax(fdict["section"])
            current_chunk_max = vmax(fdict["chunk"])
            
            # 현재 스키마와 환경변수 비교
            if current_doc_id_max != DOC_ID_MAX:
                print(f" Schema mismatch: doc_id max_length {current_doc_id_max} != {DOC_ID_MAX}")
                return True
            if current_section_max != SECTION_MAX:
                print(f" Schema mismatch: section max_length {current_section_max} != {SECTION_MAX}")
                return True
            if current_chunk_max != CHUNK_MAX:
                print(f" Schema mismatch: chunk max_length {current_chunk_max} != {CHUNK_MAX}")
                return True
            
            return False
        except Exception as e:
            print(f" Schema check error: {e}")
            return True

    def _create_collection(self) -> Collection:
        """컬렉션 및 인덱스 생성"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=DOC_ID_MAX),
            FieldSchema(name="seq", dtype=DataType.INT64),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=SECTION_MAX),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=CHUNK_MAX),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields, description="RAG chunks with metadata (v2)")
        col = Collection(self.collection_name, schema)

        # 인덱스 생성 (HNSW + IP)
        col.create_index(
            field_name="embedding",
            index_params={
                "index_type": INDEX_TYPE,
                "metric_type": METRIC_TYPE,
                "params": {"M": HNSW_M, "efConstruction": HNSW_EFCON},
            },
        )
        print(f"Created collection: {self.collection_name} (dim={self.dim}, index={INDEX_TYPE}, metric={METRIC_TYPE})")
        return col

    def _ensure_index(self) -> None:
        """컬렉션만 있고 인덱스 없는 상태 보강"""
        try:
            if not getattr(self.col, "indexes", []):
                self.col.create_index(
                    field_name="embedding",
                    index_params={
                        "index_type": INDEX_TYPE,
                        "metric_type": METRIC_TYPE,
                        "params": {"M": HNSW_M, "efConstruction": HNSW_EFCON},
                    },
                )
                print("Created missing index on existing collection")
        except Exception as e:
            print(f"ensure index failed: {e}")

    def _replace_doc_if_needed(self, doc_id: str) -> None:
        """같은 doc_id 문서를 교체(삭제 후 재삽입)하고 싶을 때 사용.
           RAG_REPLACE_DOC=1 이면 활성화.
        """
        if os.getenv("RAG_REPLACE_DOC", "1") != "1":
            return
        try:
            deleted = self._delete_by_doc_id(doc_id)
            print(f"Replaced doc: {doc_id} (deleted {deleted} chunks)")
        except Exception as e:
            print(f"replace_doc failed: {e}")

    # 동적 EF 계산
    def _calculate_ef(self, topk: int) -> int:
        """
        topk에 따라 동적으로 ef 파라미터 계산
        ef = min(EF_MAX, max(EF_SEARCH, topk * EF_PER_K))
        
        예시:
        - topk=20 → ef = max(384, 20*8) = 384
        - topk=80 → ef = max(384, 80*8) = 640
        - topk=300 → ef = min(2048, 300*8) = 2048
        """
        calculated_ef = max(EF_SEARCH, topk * EF_PER_K)
        final_ef = min(EF_MAX, calculated_ef)
        return final_ef

    # ---------------- public ----------------

    def insert(
        self, 
        doc_id: str, 
        chunks: List[Tuple[str, Dict[str, Any]]], 
        embed_fn: Callable[[List[str]], List[List[float]]],
    ) -> Dict[str, Any]:
        """
        중복 방지 & 안전 삽입:
        - RAG_SKIP_IF_EXISTS=1  : 같은 doc_id 존재 시 스킵
        - RAG_REPLACE_DOC=1     : 같은 doc_id 존재 시 삭제 후 삽입
        - RAG_DEDUP_MANIFEST=1  : MinIO docs/{doc_id}.json 에 sha256 기록/비교
        - RAG_UNIQUE_SUFFIX_ON_CONFLICT=1 : 충돌인데 REPLACE 아님 → doc_id__hash 로 새로 삽입
        
        **중요**: 텍스트 길이 제한을 임베딩 전에 적용하여 Milvus 에러 방지
        """
        out = {"inserted": 0, "skipped": False, "reason": None, "doc_id": doc_id}
        if not chunks:
            out["skipped"] = True
            out["reason"] = "empty_chunks"
            return out

        # -------- 0) 현재 상태 조회
        try:
            exists_cnt = self.count_by_doc(doc_id)
        except Exception:
            exists_cnt = 0

        SKIP_IF_EXISTS  = os.getenv("RAG_SKIP_IF_EXISTS", "0") == "1"
        REPLACE_DOC     = os.getenv("RAG_REPLACE_DOC", "0") == "1"
        USE_MANIFEST    = os.getenv("RAG_DEDUP_MANIFEST", "0") == "1"
        UNIQUE_SUFFIX   = os.getenv("RAG_UNIQUE_SUFFIX_ON_CONFLICT", "1") == "1"

        # -------- 1) 매니페스트(해시) 비교
        import hashlib
        
        # ===== 중요: 텍스트 길이 제한을 임베딩 전에 적용 =====
        # 먼저 텍스트 추출 및 안전하게 자르기
        raw_texts = [(c[0] or "") for c in chunks]
        safe_texts = [_safe_truncate_text(t, CHUNK_MAX) for t in raw_texts]
        
        # 해시 계산은 안전하게 잘린 텍스트 기준
        text_blob = "\n\n".join(safe_texts).encode("utf-8", errors="ignore")
        doc_hash = hashlib.sha256(text_blob).hexdigest()

        manifest_key = f"docs/{doc_id}.json"

        # 매니페스트 비교
        if USE_MANIFEST:
            try:
                from app.services.minio_store import MinIOStore
                prev = MinIOStore().get_json(manifest_key)
                if prev and prev.get("sha256") == doc_hash:
                    out["skipped"] = True
                    out["reason"] = "manifest_match"
                    return out
            except Exception:
                pass

        # -------- 2) 존재 여부 체크
        if exists_cnt > 0:
            if SKIP_IF_EXISTS:
                out["skipped"] = True
                out["reason"] = "exists"
                return out
            if REPLACE_DOC:
                self._replace_doc_if_needed(doc_id)
            elif UNIQUE_SUFFIX:
                doc_id = doc_id + "__" + doc_hash[:8]
                out["doc_id"] = doc_id

        # -------- 3) 메타데이터 추출 및 길이 제한 적용
        limits = _get_schema_limits(self.col)
        SEC_MAX = limits["section"]
        CHK_MAX = limits["chunk"]
        
        # 안전 마진을 위한 실제 제한값 (스키마보다 작게)
        SAFE_SECTION_LIMIT = min(SEC_MAX, 480)  # 512 - 마진
        SAFE_CHUNK_LIMIT = min(CHK_MAX, 8000)   # 8192 - 마진
        
        metas = [c[1] for c in chunks]
        pages = []
        sections = []
        seqs = list(range(len(chunks)))
        
        for m in metas:
            try:
                pages.append(int(m.get("page", 0)))
            except Exception:
                pages.append(0)
            
            # section 필드도 안전하게 자르기
            s = "" if m.get("section") is None else str(m.get("section"))
            sections.append(_safe_truncate_text(s, SAFE_SECTION_LIMIT))

        # -------- 4) 텍스트 최종 정리 (이미 safe_texts로 잘림)
        # 추가 안전장치: 다시 한번 체크
        final_texts = [_safe_truncate_text(t, SAFE_CHUNK_LIMIT) for t in safe_texts]
        
        # 디버그 로그
        truncated_count = sum(1 for orig, safe in zip(raw_texts, final_texts) if len(orig) > len(safe))
        if truncated_count > 0:
            print(f"Truncated {truncated_count}/{len(final_texts)} chunks to fit schema limits")

        # -------- 5) 임베딩 생성 (이제 안전하게 잘린 텍스트로)
        print(f"Embedding {len(final_texts)} chunks...")
        vecs = embed_fn(final_texts)
        
        if not vecs or len(vecs) != len(final_texts):
            raise RuntimeError("embedding failed: empty or count mismatch")
        
        dim0 = len(vecs[0])
        if dim0 != self.dim:
            raise RuntimeError(f"embedding dim mismatch: expect {self.dim}, got {dim0}")
        
        for i, v in enumerate(vecs):
            if len(v) != dim0:
                raise RuntimeError(f"embedding dim mismatch at {i}: {len(v)}")

        # -------- 6) doc_id 길이 제한
        if len(doc_id) > DOC_ID_MAX:
            suf = hashlib.sha256(doc_id.encode("utf-8", "ignore")).hexdigest()[:8]
            doc_id = doc_id[:(DOC_ID_MAX - 10)] + "__" + suf
            out["doc_id"] = doc_id

        # -------- 7) entities 생성 전 최종 검증
        # 스키마 제한 확인
        schema_limits = _get_schema_limits(self.col)
        actual_chunk_max = schema_limits["chunk"]
        actual_section_max = schema_limits["section"]
        
        # 최종 안전장치: 스키마 제한보다 작게 자르기
        final_texts = [_safe_truncate_text(t, actual_chunk_max - 100) for t in final_texts]
        sections = [_safe_truncate_text(s, actual_section_max - 32) for s in sections]
        
        # 검증: 모든 텍스트가 제한 내에 있는지 확인
        for i, t in enumerate(final_texts):
            byte_len = len(t.encode('utf-8', errors='ignore'))
            if byte_len > actual_chunk_max:
                print(f"⚠️ CRITICAL: Chunk {i} still exceeds limit! {byte_len} > {actual_chunk_max}")
                # 강제 자르기
                final_texts[i] = t[:actual_chunk_max - 100]
        
        # -------- 8) entities 생성 및 삽입
        entities = [
            [doc_id] * len(final_texts),   # doc_id
            seqs,                           # seq
            pages,                          # page
            sections,                       # section
            final_texts,                    # chunk (이미 안전하게 잘림)
            vecs,                           # embedding
        ]

        print(f"Inserting {len(final_texts)} chunks for doc_id={doc_id}...")
        mr = self.col.insert(entities)
        self.col.flush()
        
        try:
            self.col.load()
        except Exception:
            pass

        out["inserted"] = len(final_texts)

        # -------- 9) 매니페스트 기록(옵션)
        if USE_MANIFEST:
            try:
                from app.services.minio_store import MinIOStore
                MinIOStore().put_json(manifest_key, {
                    "doc_id": doc_id,
                    "sha256": doc_hash,
                    "chunks": len(final_texts),
                    "dim": self.dim,
                })
            except Exception:
                pass

        return out

    def delete_by_doc(self, doc_id: str) -> int:
        """doc_id로 문서 삭제 (간단한 방식)"""
        try:
            res = self.col.delete(expr=f'doc_id == "{doc_id}"')
            self.col.flush()
            return getattr(res, "delete_count", 0) or 0
        except Exception:
            return 0
        
    def _delete_by_doc_id(self, doc_id: str) -> int:
        """doc_id로 해당 문서의 PK(id)들을 조회한 뒤, PK in [...] 방식으로 삭제"""
        try:
            self.col.load()
        except Exception:
            pass

        # 1) doc_id로 PK(id) 조회
        safe = str(doc_id).replace('"', r'\"')
        rows = self.col.query(
            expr=f'doc_id == "{safe}"',
            output_fields=["id"],
        ) or []

        ids = [r["id"] for r in rows if "id" in r]
        if not ids:
            return 0

        # 2) PK로 삭제 (Milvus는 긴 리스트 삭제가 비효율적일 수 있으니 배치)
        BATCH = 16384  # 안전한 배치 크기
        for i in range(0, len(ids), BATCH):
            batch = ids[i : i + BATCH]
            self.col.delete(expr=f"id in {batch}")

        self.col.flush()
        return len(ids)
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """외부 호출용 alias"""
        return self._delete_by_doc_id(doc_id)
    
    def search(
        self, 
        query: str, 
        embed_fn: Callable[[List[str]], List[List[float]]], 
        topk: int = 20
    ) -> List[Dict[str, Any]]:
        """
        개선: 동적 ef 파라미터 적용
        IP metric + normalize 임베딩 기준으로 상위 topk 반환
        """
        if not query:
            return []
        qv = embed_fn([query])[0]

        try:
            self.col.load()
        except Exception:
            pass

        # 동적 ef 계산
        ef = self._calculate_ef(topk)

        res = self.col.search(
            data=[qv],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": ef}},  # 동적 ef 적용
            limit=topk,
            output_fields=["doc_id", "seq", "page", "section", "chunk"],
            consistency_level="Strong",
        )

        out: List[Dict[str, Any]] = []
        for hit in res[0]:
            ent = hit.entity
            out.append({
                "score": float(hit.distance),
                "doc_id": ent.get("doc_id"),
                "seq": int(ent.get("seq")),
                "page": int(ent.get("page")),
                "section": ent.get("section"),
                "chunk": ent.get("chunk"),
            })

        # 검색 로깅
        if out:
            print(f"Search: topk={topk}, ef={ef}, results={len(out)}, top_score={out[0]['score']:.3f}")
        
        return out
    
    def search_in_docs(
        self,
        query: str,
        embed_fn: Callable[[List[str]], List[List[float]]],
        doc_ids: List[str],
        topk: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        개선: 동적 ef 파라미터 적용
        특정 doc_id 목록 안에서만 검색하는 버전
        - doc_ids: osk_data.data_id 목록 (문자열)
        """
        if not query or not doc_ids:
            return []

        qv = embed_fn([query])[0]

        try:
            self.col.load()
        except Exception:
            pass

        # Milvus expr용으로 doc_id in ["...", "..."] 형태로 변환
        safe_ids = [str(d).replace('"', "").replace("\\", "") for d in doc_ids]
        expr = "doc_id in [" + ",".join(f'"{i}"' for i in safe_ids) + "]"

        # 동적 ef 계산
        ef = self._calculate_ef(topk)

        res = self.col.search(
            data=[qv],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": ef}},  # 동적 ef 적용
            limit=topk,
            expr=expr,  # 여기서 필터 적용
            output_fields=["doc_id", "seq", "page", "section", "chunk"],
            consistency_level="Strong",
        )

        out: List[Dict[str, Any]] = []
        for hit in res[0]:
            ent = hit.entity
            out.append({
                "score": float(hit.distance),
                "doc_id": ent.get("doc_id"),
                "seq": int(ent.get("seq")),
                "page": int(ent.get("page")),
                "section": ent.get("section"),
                "chunk": ent.get("chunk"),
            })

        # 검색 로깅
        if out:
            print(f"Search (filtered): topk={topk}, ef={ef}, doc_filter={len(doc_ids)}, results={len(out)}")
        
        return out

    def debug_search(
        self,
        query: str,
        embed_fn: Callable[[List[str]], List[List[float]]],
        topk: int = 5,
    ) -> List[Dict[str, Any]]:
        """디버그용 검색 (메타 정보 포함)"""
        results = self.search(query, embed_fn, topk)
        # 사용된 ef 값 로깅
        ef = self._calculate_ef(topk)
        print(f"[DEBUG] search with ef={ef} (topk={topk})")
        return results

    def count_by_doc(self, doc_id: str) -> int:
        """특정 doc_id의 청크 개수 조회"""
        try:
            self.col.load()
        except Exception:
            pass
        res = self.col.query(
            expr=f'doc_id == "{doc_id}"',
            output_fields=["doc_id"],
            limit=1
        )
        return len(res) if res else 0

    def stats(self) -> dict:
        """
        개선: 검색 파라미터 정보 추가
        컬렉션 상태 요약
        """
        try:
            num = self.col.num_entities
        except Exception:
            num = -1
        idx = []
        try:
            for ix in getattr(self.col, "indexes", []):
                params = {}
                try:
                    params = ix.params
                except Exception:
                    pass
                idx.append(params)
        except Exception:
            pass
        
        return {
            "collection": self.col.name,
            "num_entities": num,
            "dim": self.dim,
            "indexes": idx,
            "schema_fields": [f.name for f in self.col.schema.fields],
            # 검색 파라미터 정보
            "search_params": {
                "metric_type": METRIC_TYPE,
                "ef_search_base": EF_SEARCH,
                "ef_per_k": EF_PER_K,
                "ef_max": EF_MAX,
                "index_type": INDEX_TYPE,
                "hnsw_m": HNSW_M,
                "hnsw_efcon": HNSW_EFCON,
            },
        }

    def query_by_doc(self, doc_id: str, limit: int = 10) -> list[dict]:
        """특정 doc_id로 저장된 청크 확인"""
        expr = f'doc_id == "{doc_id}"'
        rows = self.col.query(
            expr=expr,
            output_fields=["doc_id", "seq", "page", "section", "chunk"],
            limit=limit
        )
        max_preview = int(os.getenv("DEBUG_PEEK_MAX_CHARS", "300"))
        return [
            {
                "doc_id": r.get("doc_id"),
                "seq": int(r.get("seq", -1)),
                "page": int(r.get("page", -1)),
                "section": r.get("section", ""),
                "chunk": (r.get("chunk", "")[:max_preview] + "..." 
                         if max_preview > 0 and len(r.get("chunk", "")) > max_preview 
                         else r.get("chunk", "")),
            }
            for r in rows
        ]

    def peek(self, limit: int = 10) -> list[dict]:
        """컬렉션의 일부 데이터 미리보기"""
        try:
            self.col.load()
        except Exception:
            pass
        
        rows = self.col.query(
            expr="id >= 0",
            output_fields=["id", "doc_id", "seq", "page", "section", "chunk"],
            limit=limit
        )
        
        max_preview = int(os.getenv("DEBUG_PEEK_MAX_CHARS", "300"))
        return [
            {
                "id": r.get("id"),
                "doc_id": r.get("doc_id"),
                "seq": int(r.get("seq", -1)),
                "page": int(r.get("page", -1)),
                "section": r.get("section", ""),
                "chunk": (r.get("chunk", "")[:max_preview] + "..." 
                         if max_preview > 0 and len(r.get("chunk", "")) > max_preview 
                         else r.get("chunk", "")),
            }
            for r in rows
        ]