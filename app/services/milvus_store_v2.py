# app/services/milvus_store_v2.py
from __future__ import annotations

import os
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
          page (INT64)
          section (VARCHAR 512)
          chunk (VARCHAR 8192)
          embedding (FLOAT_VECTOR dim={dim})
      - 인덱스: HNSW + IP (Milvus 2.2.x 에서 COSINE 미지원 → IP 사용)
      - 임베딩은 normalize_embeddings=True로 인코딩하여 IP == cosine로 동작
    """

    def __init__(self, dim: int, name: Optional[str] = None):
        self.dim = int(dim)
        self.collection_name = name or COLLECTION

        host = os.getenv("MILVUS_HOST", "milvus")
        port = os.getenv("MILVUS_PORT", "19530")
        if not connections.has_connection("default"):
            connections.connect(alias="default", host=host, port=port)

        force_reset = os.getenv("RAG_RESET_COLLECTION", "1") == "1"

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
            print(f"⚠️ load skipped: {e}")

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
            if vmax(fdict["doc_id"]) != DOC_ID_MAX: return True
            if vmax(fdict["section"]) != SECTION_MAX: return True
            if vmax(fdict["chunk"])   != CHUNK_MAX:  return True
            return False
        except Exception:
            return True

    def _create_collection(self) -> Collection:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length= DOC_ID_MAX),
            FieldSchema(name="seq", dtype=DataType.INT64),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length= SECTION_MAX),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length= CHUNK_MAX),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields, description="RAG chunks with metadata (v2)")
        col = Collection(self.collection_name, schema)

        # 인덱스 생성 (Milvus 2.2.x: COSINE 미지원 → IP)
        col.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        print(f"✅ created collection: {self.collection_name} (dim={self.dim})")
        return col

    def _ensure_index(self) -> None:
        """컬렉션만 있고 인덱스 없는 상태 보강"""
        try:
            if not getattr(self.col, "indexes", []):
                self.col.create_index(
                    field_name="embedding",
                    index_params={
                        "index_type": "HNSW",
                        "metric_type": "IP",
                        "params": {"M": 16, "efConstruction": 200},
                    },
                )
                print("✅ created missing index on existing collection")
        except Exception as e:
            print(f"⚠️ ensure index failed: {e}")

    def _replace_doc_if_needed(self, doc_id: str) -> None:
        """같은 doc_id 문서를 교체(삭제 후 재삽입)하고 싶을 때 사용.
           RAG_REPLACE_DOC=1 이면 활성화.
        """
        if os.getenv("RAG_REPLACE_DOC", "1") != "1":
            return
        try:
            deleted = self._delete_by_doc_id(doc_id)
            print(f"replaced doc: {doc_id} (deleted {deleted})")
        except Exception as e:
            print(f"⚠️ replace_doc failed: {e}")

    # ---------------- public ----------------

    def insert(self, doc_id: str, chunks: List[Tuple[str, Dict[str, Any]]], embed_fn: Callable[[List[str]], List[List[float]]], ) -> Dict[str, Any]:
        """
        중복 방지 & 안전 삽입:
        - RAG_SKIP_IF_EXISTS=1  : 같은 doc_id 존재 시 스킵
        - RAG_REPLACE_DOC=1     : 같은 doc_id 존재 시 삭제 후 삽입
        - RAG_DEDUP_MANIFEST=1  : MinIO docs/{doc_id}.json 에 sha256 기록/비교
        - RAG_UNIQUE_SUFFIX_ON_CONFLICT=1 : 충돌인데 REPLACE 아님 → doc_id__hash 로 새로 삽입
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
        # 해시 = 전체 텍스트 조인 sha256
        import hashlib
        texts = [c[0] for c in chunks]
        text_blob = "\n\n".join(texts).encode("utf-8", errors="ignore")
        doc_hash = hashlib.sha256(text_blob).hexdigest()
        manifest_key = f"docs/{doc_id}.json"
        manifest = None
        if USE_MANIFEST:
            try:
                from app.services.minio_store import MinIOStore
                m = MinIOStore()
                if m.exists(manifest_key):
                    manifest = m.get_json(manifest_key)
            except Exception:
                manifest = None

        # 해시가 같으면 스킵
        if manifest and manifest.get("sha256") == doc_hash:
            out["skipped"] = True
            out["reason"] = "same_hash"
            return out

        # -------- 2) 존재 정책 처리
        if exists_cnt > 0:
            if SKIP_IF_EXISTS and not manifest:
                # 단순 존재 스킵
                out["skipped"] = True
                out["reason"] = "exists_skip"
                return out

            if REPLACE_DOC:
                try:
                    deleted = self._delete_by_doc_id(doc_id)
                    print(f"replace existing doc_id={doc_id}: deleted {deleted} rows")
                except Exception as e:
                    raise RuntimeError(f"failed to replace existing doc_id={doc_id}: {e}")
            else:
                # REPLACE 아님 → 충돌 처리
                if UNIQUE_SUFFIX:
                    suffix = doc_hash[:8]
                    doc_id = f"{doc_id}__{suffix}"
                    out["doc_id"] = doc_id
                else:
                    out["skipped"] = True
                    out["reason"] = "exists_conflict"
                    return out

        # -------- 3) 메타 정규화
        limits = _get_schema_limits(self.col)
        SEC_MAX   = int(limits["section"])
        CHK_MAX = int(limits["chunk"])
        SAFE_VARCHAR_CAP = 512
        SEC_MAX = min(SEC_MAX if SEC_MAX>0 else SECTION_MAX, SAFE_VARCHAR_CAP)
        RAG_SEC_MAX = int(os.getenv("RAG_SECTION_MAX", 160))
        
        metas = [c[1] for c in chunks]
        pages = []
        sections = []
        for m in metas:
            try:
                pages.append(int(m.get("page", 0)))
            except Exception:
                pages.append(0)
            s = "" if m.get("section") is None else str(m.get("section"))
            s=s[:RAG_SEC_MAX]
            sections.append(s[:SEC_MAX])  #  실제 스키마 길이로 clamp
            
        # -------- 4) 텍스트 준비 (임베딩 전에 잘라서 "임베딩 내용 == 저장 내용")
        texts = [(c[0] or "") for c in chunks]
        texts = [t[:CHUNK_MAX] for t in texts]  #  하드 클램프
        
        seqs = list(range(len(texts)))

        # -------- 4) 임베딩 + 차원 검증
        vecs = embed_fn(texts)
        if not vecs or len(vecs) != len(texts):
            raise RuntimeError("embedding failed: empty or count mismatch")
        dim0 = len(vecs[0])
        if dim0 != self.dim:
            raise RuntimeError(f"embedding dim mismatch: expect {self.dim}, got {dim0}")
        for i, v in enumerate(vecs):
            if len(v) != dim0:
                raise RuntimeError(f"embedding dim mismatch at {i}: {len(v)}")

        # -------- 5) 리스트-컬럼 방식으로 삽입 (스키마: [id, doc_id, page, section, chunk, embedding])
        if len(doc_id) > DOC_ID_MAX:
            import hashlib
            suf = hashlib.sha256(doc_id.encode("utf-8","ignore")).hexdigest()[:8]
            doc_id = doc_id[:(DOC_ID_MAX-10)] + "__" + suf
        # 리스트로 묶어서 삽입
        entities = [
            [doc_id] * len(texts),   # doc_id
            seqs,                    # seq(in doc)
            pages,                   # page
            sections,                # section
            [t[:8192] for t in texts],  # chunk
            vecs,                    # embedding
        ]

        mr = self.col.insert(entities)
        self.col.flush()
        try:
            self.col.load()
        except Exception:
            pass

        out["inserted"] = len(texts)

        # -------- 6) 매니페스트 기록(옵션)
        if USE_MANIFEST:
            try:
                from app.services.minio_store import MinIOStore
                MinIOStore().put_json(manifest_key, {
                    "doc_id": doc_id,
                    "sha256": doc_hash,
                    "chunks": len(texts),
                    "dim": self.dim,
                })
            except Exception:
                pass

        return out

    def delete_by_doc(self, doc_id: str) -> int:
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
            # limit는 제거(모두 가져오기). 환경에 따라 한번에 너무 많으면 나눠서 조회 필요
        ) or []

        ids = [r["id"] for r in rows if "id" in r]
        if not ids:
            return 0

        # 2) PK로 삭제 (Milvus는 긴 리스트 삭제가 비효율적일 수 있으니 배치)
        BATCH = 16384  # 안전한 배치 크기
        for i in range(0, len(ids), BATCH):
            batch = ids[i : i + BATCH]
            # 리스트를 그대로 포맷팅하면 [1,2,3] 형태가 되어 expr로 사용 가능
            self.col.delete(expr=f"id in {batch}")

        self.col.flush()
        return len(ids)
    
    def search(self, query: str, embed_fn: Callable[[List[str]], List[List[float]]], topk: int = 20) -> List[Dict[str, Any]]:
        """IP metric + normalize 임베딩 기준으로 상위 topk 반환"""
        if not query:
            return []
        qv = embed_fn([query])[0]

        try:
            self.col.load()
        except Exception:
            pass

        res = self.col.search(
            data=[qv],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=topk,
            output_fields=["doc_id", "seq", "page", "section", "chunk"],
            consistency_level="Strong",  # 바로 insert한 것도 검색 반영
        )

        out: List[Dict[str, Any]] = []
        for hit in res[0]:
            ent = hit.entity
            out.append(
                {
                    "score": float(hit.distance),  # IP similarity (normalized → cosine과 동일하게 해석)
                    "doc_id": ent.get("doc_id"),
                    "seq": int(ent.get("seq")),
                    "page": int(ent.get("page")),
                    "section": ent.get("section"),
                    "chunk": ent.get("chunk"),
                }
            )
        return out
    
# ----------------카운트 관련----------------
    def count_by_doc(self, doc_id: str) -> int:
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

# 파일 상단 import 그대로 두고, 클래스 안에 아래 메서드들 추가

    def stats(self) -> dict:
        """컬렉션 상태 요약"""
        try:
            num = self.col.num_entities
        except Exception:
            num = -1
        idx = []
        try:
            for ix in getattr(self.col, "indexes", []):
                # Milvus 2.2.x 에서 index.params 구조가 다를 수 있어 방어적으로 추출
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
            "indexes": idx,
            "schema_fields": [f.name for f in self.col.schema.fields],
        }

    def query_by_doc(self, doc_id: str, limit: int = 10) -> list[dict]:
        """특정 doc_id로 저장된 청크 확인"""
        expr = f'doc_id == "{doc_id}"'
        rows = self.col.query(
            expr=expr,
            output_fields=["doc_id", "seq", "page", "section", "chunk"],
            limit=limit
        )
        return [
            {
                "doc_id": r.get("doc_id"),
                "page": int(r.get("page", -1)),
                "section": r.get("section", ""),
                # 잘림 길이를 라우터에서 제어하도록 환경변수 허용 (기본 300)
                "chunk": (
                    (r.get("chunk") or "")[: int(os.getenv("DEBUG_PEEK_MAX_CHARS", "300"))]
                    if os.getenv("DEBUG_PEEK_MAX_CHARS", "300") != "0"
                    else (r.get("chunk") or "")
                ),
            }
            for r in rows
        ]

    def peek(self, limit: int = 5) -> list[dict]:
        """아무거나 몇 개 보기(샘플)"""
        rows = self.col.query(
            expr="page >= 0",
            output_fields=["doc_id", "seq", "page", "section", "chunk"],
            limit=limit
        )
        return [
            {
                "doc_id": r.get("doc_id"),
                "page": int(r.get("page", -1)),
                "section": r.get("section", ""),
                "chunk": (
                    (r.get("chunk") or "")[: int(os.getenv("DEBUG_PEEK_MAX_CHARS", "300"))]
                    if os.getenv("DEBUG_PEEK_MAX_CHARS", "300") != "0"
                    else (r.get("chunk") or "")
                ),
            }
            for r in rows
        ]
        
    def debug_search(self, query: str, embed_fn, topk: int = 5) -> list[dict]:
        """리랭크 전 순수 벡터 검색 결과 보기"""
        qv = embed_fn([query])[0]
        self.col.load()
        res = self.col.search(
            data=[qv],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=topk,
            output_fields=["doc_id", "seq", "page", "section", "chunk"]
        )
        out = []
        if res and res[0]:
            for h in res[0]:
                out.append({
                    "score_ip": float(h.distance),
                    "doc_id": h.entity.get("doc_id"),
                    "page": int(h.entity.get("page")),
                    "section": h.entity.get("section"),
                    "chunk": (
                         (h.entity.get("chunk") or "")[: int(os.getenv("DEBUG_PEEK_MAX_CHARS", "300"))]
                         if os.getenv("DEBUG_PEEK_MAX_CHARS", "300") != "0"
                         else (h.entity.get("chunk") or "")
                     ),
                })
        return out
    
    # def neighbors_by_seq(self, doc_id: str, center_seq: int, k_before: int = 2, k_after: int = 2) -> list[dict]:
    #     lo = max(0, int(center_seq) - int(k_before))
    #     hi = int(center_seq) + int(k_after)
    #     expr = f'doc_id == "{doc_id}" and seq >= {lo} and seq <= {hi}'
    #     rows = self.col.query(
    #         expr=expr,
    #         output_fields=["doc_id", "seq", "page", "section", "chunk"],
    #         limit= max(1, (k_before + k_after + 1)) * 5   # 안전 여유치
    #     )
    #     # seq 기준 정렬 + 중복 제거
    #     rows = sorted(rows, key=lambda r: int(r.get("seq", 0)))
    #     dedup = []
    #     seen = set()
    #     for r in rows:
    #         key = (r.get("doc_id"), int(r.get("seq", -1)))
    #         if key in seen: continue
    #         seen.add(key)
    #         dedup.append({
    #             "doc_id": r.get("doc_id"),
    #             "seq": int(r.get("seq", -1)),
    #             "page": int(r.get("page", -1)),
    #             "section": r.get("section", ""),
    #             "chunk": r.get("chunk") or "",
    #         })
    #     return dedup


    def neighbors_by_seq(self, doc_id: str, center_seq: int, k_before: int = 1, k_after: int = 1) -> list[dict]:
        """
        같은 doc_id 내에서 seq 기준으로 앞뒤 이웃을 포함해 묶음을 돌려준다.
        반환: [{doc_id, page, seq, section, chunk}, ...] (seq 오름차순)
        """
        try:
            self.col.load()
        except Exception:
            pass
        start = int(center_seq) - int(k_before)
        end   = int(center_seq) + int(k_after)
        expr = f'doc_id == "{doc_id}" && seq >= {start} && seq <= {end}'
        rows = self.col.query(
            expr=expr,
            output_fields=["doc_id", "page", "seq", "section", "chunk"],
            limit=10000
        )
        rows = rows or []
        rows.sort(key=lambda r: int(r.get("seq", 0)))
        # page/seq 정규화 + 미리보기 자르기(환경변수)
        out = []
        for r in rows:
            out.append({
                "doc_id": r.get("doc_id"),
                "page": int(r.get("page", -1)),
                "seq": int(r.get("seq", -1)),
                "section": r.get("section", ""),
                "chunk": (
                    (r.get("chunk") or "")[: int(os.getenv("DEBUG_PEEK_MAX_CHARS", "300"))]
                    if os.getenv("DEBUG_PEEK_MAX_CHARS", "300") != "0"
                    else (r.get("chunk") or "")
                ),
            })
        return out
    