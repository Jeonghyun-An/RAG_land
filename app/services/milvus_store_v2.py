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
            required = ("doc_id", "page", "section", "chunk", "embedding")
            if any(r not in fdict for r in required):
                return True
            emb = fdict["embedding"]
            # dim 읽기 (버전에 따라 params 또는 속성)
            emb_dim = emb.params.get("dim") if hasattr(emb, "params") else getattr(emb, "dim", None)
            if int(emb_dim or 0) != int(expect_dim):
                return True
            return False
        except Exception:
            return True

    def _create_collection(self) -> Collection:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=8192),
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
        if os.getenv("RAG_REPLACE_DOC", "0") != "1":
            return
        try:
            self.col.delete(expr=f'doc_id == "{doc_id}"')
            self.col.flush()
            print(f"ℹ️ replaced doc: {doc_id}")
        except Exception as e:
            print(f"⚠️ replace_doc failed: {e}")

    # ---------------- public ----------------

    def insert(
        self,
        doc_id: str,
        chunks: List[Tuple[str, Dict[str, Any]]],
        embed_fn: Callable[[List[str]], List[List[float]]],
) ->     Dict[str, Any]:
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
                    self.col.delete(expr=f'doc_id == "{doc_id}"')
                    self.col.flush()
                    # 계속 진행해서 새로 삽입
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
        metas = [c[1] for c in chunks]
        pages = []
        sections = []
        for m in metas:
            try:
                pages.append(int(m.get("page", 0)))
            except Exception:
                pages.append(0)
            sections.append(str(m.get("section", ""))[:512])

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
        entities = [
            [doc_id] * len(texts),   # doc_id
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
            output_fields=["doc_id", "page", "section", "chunk"],
            consistency_level="Strong",  # 바로 insert한 것도 검색 반영
        )

        out: List[Dict[str, Any]] = []
        for hit in res[0]:
            ent = hit.entity
            out.append(
                {
                    "score": float(hit.distance),  # IP similarity (normalized → cosine과 동일하게 해석)
                    "doc_id": ent.get("doc_id"),
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
            limit=100000
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
            output_fields=["doc_id", "page", "section", "chunk"],
            limit=limit
        )
        # rows: List[Dict]
        # 텍스트가 길 수 있으니 미리 자르기(디버그 가독성)
        out = []
        for r in rows:
            out.append({
                "doc_id": r.get("doc_id"),
                "page": int(r.get("page", -1)),
                "section": r.get("section", ""),
                "chunk": (r.get("chunk") or "")[:300]
            })
        return out

    def peek(self, limit: int = 5) -> list[dict]:
        """아무거나 몇 개 보기(샘플)"""
        rows = self.col.query(
            expr="page >= 0",
            output_fields=["doc_id", "page", "section", "chunk"],
            limit=limit
        )
        out = []
        for r in rows:
            out.append({
                "doc_id": r.get("doc_id"),
                "page": int(r.get("page", -1)),
                "section": r.get("section", ""),
                "chunk": (r.get("chunk") or "")[:300]
            })
        return out

    def debug_search(self, query: str, embed_fn, topk: int = 5) -> list[dict]:
        """리랭크 전 순수 벡터 검색 결과 보기"""
        qv = embed_fn([query])[0]
        self.col.load()
        res = self.col.search(
            data=[qv],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=topk,
            output_fields=["doc_id", "page", "section", "chunk"]
        )
        out = []
        if res and res[0]:
            for h in res[0]:
                out.append({
                    "score_ip": float(h.distance),
                    "doc_id": h.entity.get("doc_id"),
                    "page": int(h.entity.get("page")),
                    "section": h.entity.get("section"),
                    "chunk": (h.entity.get("chunk") or "")[:300],
                })
        return out
