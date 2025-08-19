from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks_v2")

class MilvusStoreV2:
    def __init__(self, dim: int):
        host, port = os.getenv("MILVUS_HOST","milvus"), os.getenv("MILVUS_PORT","19530")
        if not connections.has_connection("default"):
            connections.connect(host=host, port=port)
        if not utility.has_collection(COLLECTION):
            fields = [
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("doc_id", DataType.VARCHAR, max_length=256),
                FieldSchema("page", DataType.INT64),
                FieldSchema("section", DataType.VARCHAR, max_length=512),
                FieldSchema("chunk", DataType.VARCHAR, max_length=8192),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields, description="RAG chunks with metadata")
            col = Collection(COLLECTION, schema)
            col.create_index("embedding", {
                "index_type":"HNSW","metric_type":"IP",
                "params":{"M":16,"efConstruction":200}
            })
        self.col = Collection(COLLECTION)
        try: self.col.load()
        except: pass

    def insert(self, doc_id: str, chunks: List[Tuple[str, Dict[str,Any]]], embed_fn):
        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        vecs  = embed_fn(texts)
        dim = len(vecs[0])
        self.col.insert({
            "doc_id":[doc_id]*len(texts),
            "page":[int(m["page"]) for m in metas],
            "section":[m.get("section","") for m in metas],
            "chunk":texts,
            "embedding":vecs,
        })
        self.col.flush(); self.col.load()

    def search(self, query: str, embed_fn, topk: int = 20):
        qv = embed_fn([query])[0]
        self.col.load()
        res = self.col.search(
            data=[qv], anns_field="embedding",
            param={"metric_type":"IP","params":{"ef":64}},
            limit=topk, output_fields=["doc_id","page","section","chunk"]
        )
        out=[]
        for h in res[0]:
            out.append({
                "distance": float(h.distance),  # COSINE distance or 1-cos? 드라이버 표준에 맞춰 사용
                "doc_id": h.entity.get("doc_id"),
                "page": int(h.entity.get("page")),
                "section": h.entity.get("section"),
                "chunk": h.entity.get("chunk"),
            })
        return out
