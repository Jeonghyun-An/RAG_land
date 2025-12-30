# finetune/prepare_from_milvus.py
import json
import asyncio
from pathlib import Path
from typing import List, Dict
import random
import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "nuclearchat-milvus-1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks_v2")

async def extract_from_milvus(target_count: int = 3000):
    print(f" Connecting to Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    
    try:
        from pymilvus import connections, Collection
        
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        
        print(f" Connected to Milvus")
        
        collection = Collection(MILVUS_COLLECTION)
        collection.load()
        
        print(f" Collection: {MILVUS_COLLECTION}")
        print(f" Total entities: {collection.num_entities}")
        
        #  스키마 확인
        print(f"\n Schema fields:")
        for field in collection.schema.fields:
            print(f"  - {field.name}: {field.dtype}")
        
        #  쿼리 수정 (chunk_index 대신 id 또는 빈 expr 사용)
        total_entities = collection.num_entities
        limit = min(target_count, total_entities)
        
        print(f"\n Extracting {limit} chunks...")
        
        # 방법 1: id 필드 사용
        results = collection.query(
            expr=f"id >= 0",  #  chunk_index → id
            output_fields=["doc_id", "section", "chunk", "page"],
            limit=limit
        )
        
        print(f" Extracted {len(results)} chunks from Milvus")
        
        qa_pairs = []
        for i, chunk in enumerate(results):
            if i % 100 == 0:
                print(f"Processing... {i}/{len(results)}")
            
            qa_pairs.extend(generate_qa_from_chunk(chunk))
        
        print(f" Generated {len(qa_pairs)} QA pairs")
        
        connections.disconnect("default")
        return qa_pairs
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return []

def generate_qa_from_chunk(chunk: Dict) -> List[Dict]:
    """청크에서 QA 생성"""
    text = chunk.get('chunk', '').strip()
    section = chunk.get('section', '').strip()
    doc_id = chunk.get('doc_id', '')
    
    if len(text) < 50:
        return []
    
    qa_pairs = []
    
    # 패턴 1: 섹션 기반 질문
    if section:
        qa_pairs.append({
            "instruction": f"{section}에 대해 설명해주세요.",
            "input": "",
            "output": text[:600]
        })
        
        qa_pairs.append({
            "instruction": f"{section}의 내용은 무엇인가요?",
            "input": f"문서 ID: {doc_id}",
            "output": text[:500]
        })
    
    # 패턴 2: 정의/의미
    if any(kw in text for kw in ["정의", "의미", "이란", "refers to", "means"]):
        keyword = section if section else "이 개념"
        qa_pairs.append({
            "instruction": f"{keyword}의 정의는 무엇인가요?",
            "input": "",
            "output": text[:400]
        })
    
    # 패턴 3: 절차/방법
    if any(kw in text for kw in ["방법", "절차", "단계", "과정", "procedure", "method"]):
        keyword = section if section else "이 작업"
        qa_pairs.append({
            "instruction": f"{keyword}의 절차는 어떻게 되나요?",
            "input": "",
            "output": text[:500]
        })
    
    # 패턴 4: 기준/규정
    if any(kw in text for kw in ["기준", "한도", "제한", "규정", "criteria", "limit", "requirement"]):
        keyword = section if section else "이 항목"
        qa_pairs.append({
            "instruction": f"{keyword}의 기준은 무엇인가요?",
            "input": "",
            "output": text[:450]
        })
    
    # 패턴 5: 구성요소
    if any(kw in text for kw in ["구성", "포함", "요소", "부품", "component", "consists"]):
        keyword = section if section else "이 시스템"
        qa_pairs.append({
            "instruction": f"{keyword}의 구성요소는 무엇인가요?",
            "input": "",
            "output": text[:500]
        })
    
    # 패턴 6: 법 조항 (한국어)
    if "제" in text and "조" in text:
        qa_pairs.append({
            "instruction": f"관련 법 조항에 대해 설명해주세요.",
            "input": section,
            "output": text[:550]
        })
    
    # 패턴 7: IAEA 관련 (영문)
    if "IAEA" in text or "Safety Standards" in text:
        qa_pairs.append({
            "instruction": "IAEA 안전 기준에 대해 설명해주세요.",
            "input": section,
            "output": text[:500]
        })
    
    # 패턴 8: 일반 질문 (항상 포함)
    qa_pairs.append({
        "instruction": "다음 내용을 요약해주세요.",
        "input": section or doc_id,
        "output": text[:400]
    })
    
    return qa_pairs

def save_dataset(data: List[Dict], output_path: str):
    """JSONL 저장"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f" Saved {len(data)} examples to {output_path}")

async def main():
    print("="*60)
    print(" Extracting QA Data from Milvus")
    print("="*60)
    
    qa_pairs = await extract_from_milvus(target_count=3000)
    
    if not qa_pairs:
        print(" No data extracted")
        return
    
    # 중복 제거
    unique_qa = {}
    for qa in qa_pairs:
        key = f"{qa['instruction']}:{qa['output'][:100]}"
        if key not in unique_qa:
            unique_qa[key] = qa
    
    qa_pairs = list(unique_qa.values())
    print(f" After deduplication: {len(qa_pairs)} unique QA pairs")
    
    # Train/Test 분할
    random.shuffle(qa_pairs)
    split_idx = int(len(qa_pairs) * 0.9)
    
    train_data = qa_pairs[:split_idx]
    test_data = qa_pairs[split_idx:]
    
    # 저장
    save_dataset(train_data, "/workspace/data/nuclear_qa.jsonl")
    save_dataset(test_data, "/workspace/data/test_qa.jsonl")
    
    print("="*60)
    print(f" Final Statistics:")
    print(f"   Train: {len(train_data)} examples")
    print(f"   Test:  {len(test_data)} examples")
    print(f"   Total: {len(qa_pairs)} unique QA pairs")
    print("="*60)
    
    # 샘플 출력
    print("\n Sample QA Pairs:\n")
    for i, qa in enumerate(train_data[:3], 1):
        print(f"[{i}] Q: {qa['instruction']}")
        print(f"    A: {qa['output'][:100]}...")
        print()

if __name__ == "__main__":
    asyncio.run(main())