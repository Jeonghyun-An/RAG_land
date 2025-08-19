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

    def insert(self, doc_id: str, chunks: List[Tuple[str, Dict[str, Any]]], embed_fn: Callable[[List[str]], List[List[float]]]) -> None:
        """
        chunks: [(chunk_text, {"page":..., "section":..., "idx":...}), ...]
        embed_fn: 텍스트 리스트 → 벡터 리스트 (normalize_embeddings=True 권장)
        """
        if not chunks:
            return

        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]

        # (옵션) 동일 doc_id 교체
        self._replace_doc_if_needed(doc_id)

        # 임베딩
        vecs = embed_fn(texts)
        if not vecs:
            raise RuntimeError("❌ embedding result is empty.")
        # 차원 방어
        if len(vecs[0]) != self.dim:
            raise RuntimeError(f"❌ embedding dim mismatch: expected {self.dim}, got {len(vecs[0])}")

        # 삽입
        self.col.insert(
            {
                "doc_id": [doc_id] * len(texts),
                "page": [int(m.get("page", 0)) for m in metas],
                "section": [str(m.get("section", ""))[:512] for m in metas],
                "chunk": [t[:8192] for t in texts],
                "embedding": vecs,
            }
        )
        self.col.flush()
        try:
            self.col.load()
        except Exception:
            pass

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
