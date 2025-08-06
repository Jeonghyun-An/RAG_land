# app/services/milvus_store.py

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import numpy as np

class MilvusStore:
    def __init__(self, collection_name="rag_chunks", host="localhost", port="19530"):
        self.collection_name = collection_name
        self.embedding_dim = 768
        self.model = SentenceTransformer("jhgan/ko-sbert-nli")

        connections.connect(host=host, port=port)

        if not self.collection_exists():
            self.create_collection()
        self.collection = Collection(self.collection_name)

    def collection_exists(self):
        return self.collection_name in [c.name for c in Collection.list()]

    def create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
        ]
        schema = CollectionSchema(fields, description="RAG chunks collection")
        Collection(name=self.collection_name, schema=schema).create_index(field_name="embedding", index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        })

    def add_texts(self, texts: list[str]):
        embeddings = self.model.encode(texts).tolist()
        self.collection.insert([texts, embeddings])
        self.collection.flush()

    def search(self, query: str, top_k=3) -> list[str]:
        embedding = self.model.encode([query])[0].tolist()
        self.collection.load()
        result = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["chunk"]
        )
        return [hit.entity.get("chunk") for hit in result[0]]
