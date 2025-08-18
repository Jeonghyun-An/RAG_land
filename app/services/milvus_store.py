# app/services/milvus_store.py
from __future__ import annotations

import os
import time
from typing import List, Optional

from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, MilvusException, utility
)
from pymilvus.orm.mutation import MutationResult

from app.services.embedding_model import get_embedding_model, embed


class MilvusStore:
    def __init__(self, collection_name: str = "rag_chunks"):
        self.collection_name = collection_name

        # 임베딩 차원은 모델에서 직접 가져오자 (지연 로딩 안전)
        self._model = None  # lazy
        try:
            self._model = get_embedding_model()
            self.embedding_dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            # 모델 설치/로딩 실패해도 서버는 떠야 하므로 conservative fallback
            self.embedding_dim = 768

        host = os.getenv("MILVUS_HOST", "milvus")
        port = os.getenv("MILVUS_PORT", "19530")

        try:
            if not connections.has_connection("default"):
                connections.connect(alias="default", host=host, port=port)
            print(f"✅ Milvus 연결 성공: {host}:{port}")
        except Exception as e:
            print(f"❌ Milvus 연결 실패: {e}")
            raise

        # 컬렉션 없으면 생성
        if not utility.has_collection(self.collection_name):
            self._create_collection()

        self.collection = Collection(self.collection_name)

        # 데이터가 있을 때만 load 수행
        try:
            if self.collection.num_entities > 0:
                self.collection.load()
        except MilvusException as e:
            print(f"⚠️ load 스킵 (사유: {e})")

    # ---- 내부 유틸 ----
    def _create_collection(self) -> None:
        """
        rag_chunks 스키마:
          - id: INT64, primary key, auto_id
          - chunk: VARCHAR(2048)
          - embedding: FLOAT_VECTOR(dim=embedding_dim)
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
        ]
        schema = CollectionSchema(fields, description="RAG chunks collection")
        collection = Collection(name=self.collection_name, schema=schema)

        # IVF_FLAT (L2) 인덱스
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"✅ 컬렉션 생성 완료: {self.collection_name} (dim={self.embedding_dim})")

    # ---- public API ----
    def add_texts(self, texts: List[str]) -> MutationResult:
        if not texts:
            raise ValueError("❌ texts가 비었습니다.")

        # Sentence-Transformers는 기본적으로 cosine 유사도를 잘 쓰므로
        # embed()에서 normalize=True로 단위벡터 → L2와 cosine이 거의 동일하게 작동
        vectors = embed(texts)  # List[List[float]]
        if not vectors:
            raise ValueError("❌ 임베딩 결과가 없습니다.")

        # 스키마에서 auto_id=True이므로 id는 넣지 않음. 순서는 (chunk, embedding)
        mr = self.collection.insert([texts, vectors])
        self.collection.flush()

        # 새로 insert했으면 재로드
        try:
            if self.collection.num_entities > 0:
                self.collection.load()
        except MilvusException:
            pass
        return mr

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not query:
            return []
        if getattr(self.collection, "num_entities", 0) == 0:
            raise RuntimeError("❌ Milvus 컬렉션에 데이터가 없습니다. 먼저 업서트하세요.")

        qvec = embed([query])[0]

        # 검색 전 보장 로드
        try:
            self.collection.load()
        except MilvusException:
            pass

        result = self.collection.search(
            data=[qvec],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["chunk"],
            consistency_level="Strong",   # 갓 insert한 데이터도 즉시 검색되게
        )
        return [hit.entity.get("chunk") for hit in result[0]]

    @staticmethod
    def wait_for_milvus(timeout: int = 30) -> None:
        host = os.getenv("MILVUS_HOST", "milvus")
        port = os.getenv("MILVUS_PORT", "19530")
        for i in range(timeout):
            try:
                if not connections.has_connection("default"):
                    connections.connect(alias="default", host=host, port=port)
                # 응답성 체크
                _ = utility.list_collections()
                print("✅ Milvus가 준비되었습니다.")
                return
            except Exception:
                print(f"⏳ Milvus 연결 재시도 중... ({i + 1}/{timeout})")
                time.sleep(1)
        raise RuntimeError("❌ Milvus가 준비되지 않았습니다.")
