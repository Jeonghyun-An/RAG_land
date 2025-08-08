import os
import time
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, MilvusException
)
from app.services.embedding_model import embedding_model


class MilvusStore:
    def __init__(self, collection_name="rag_chunks"):
        self.collection_name = collection_name
        self.embedding_dim = 768  # or embedding_model.get_sentence_embedding_dimension()
        self.model = embedding_model

        host = os.getenv("MILVUS_HOST", "milvus")
        port = os.getenv("MILVUS_PORT", "19530")

        try:
            connections.connect(alias="default", host=host, port=port)
            print(f"✅ Milvus 연결 성공: {host}:{port}")
        except Exception as e:
            print(f"❌ Milvus 연결 실패: {e}")
            raise

        if not self.collection_exists():
            self.create_collection()

        self.collection = Collection(self.collection_name)

        # 🔒 데이터가 있을 때만 load 수행
        if self.collection.num_entities > 0:
            self.collection.load()

    def collection_exists(self):
        return self.collection_name in [c.name for c in Collection.list()]

    def create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
        ]
        schema = CollectionSchema(fields, description="RAG chunks collection")
        collection = Collection(name=self.collection_name, schema=schema)

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"✅ 컬렉션 생성 완료: {self.collection_name}")

    def add_texts(self, texts: list[str]):
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
        if not embeddings:
            raise ValueError("❌ 임베딩 결과가 없습니다.")

        self.collection.insert([texts, embeddings])
        self.collection.flush()
        self.collection.load()  # 📌 새로 insert했으면 다시 load

    def search(self, query: str, top_k=3) -> list[str]:
        if self.collection.num_entities == 0:
            raise RuntimeError("❌ Milvus 컬렉션에 데이터가 없습니다. 먼저 파일을 업로드하세요.")

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

    @staticmethod
    def wait_for_milvus(timeout=30):
        host = os.getenv("MILVUS_HOST", "milvus")
        port = os.getenv("MILVUS_PORT", "19530")
        for i in range(timeout):
            try:
                connections.connect(alias="default", host=host, port=port)
                _ = Collection.list()
                print("✅ Milvus가 준비되었습니다.")
                return
            except Exception:
                print(f"⏳ Milvus 연결 재시도 중... ({i + 1}/{timeout})")
                time.sleep(1)
        raise RuntimeError("❌ Milvus가 준비되지 않았습니다.")
