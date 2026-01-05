#!/usr/bin/env python3
# finetune/extract_from_milvus.py
"""
Milvus에서 선택된 문서의 청크 데이터를 추출하여 학습용 QA 데이터셋 생성

사용법:
    python finetune/extract_from_milvus.py --doc-ids DOC1 DOC2 --output data/training_data.jsonl

특징:
    - 원자력 도메인 특화 QA 샘플 생성
    - 법 조항, 매뉴얼, 기술문서별 맞춤 템플릿
    - 데이터 증강 및 품질 필터링
"""
import os
import json
import argparse
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
from pymilvus import connections, Collection

# Milvus 설정
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks_v2")

# QA 생성 패턴
QA_PATTERNS = {
    "law": {  # 법 조항
        "keywords": ["제", "조", "항", "호", "법", "규정", "기준"],
        "templates": [
            "{section}의 내용은 무엇인가요?",
            "{section}에 대해 설명해주세요.",
            "{topic}에 관한 법적 규정은?",
            "{topic}의 기준은 어떻게 되나요?"
        ]
    },
    "manual": {  # 매뉴얼/절차
        "keywords": ["절차", "방법", "단계", "수행", "작업", "점검"],
        "templates": [
            "{topic}의 절차를 설명해주세요.",
            "{topic}은 어떻게 수행하나요?",
            "{topic} 시 주의사항은?",
            "{topic}의 단계별 방법은?"
        ]
    },
    "technical": {  # 기술문서
        "keywords": ["설계", "계통", "구조", "원리", "기능", "성능"],
        "templates": [
            "{topic}의 원리를 설명해주세요.",
            "{topic}의 주요 기능은?",
            "{topic}은 어떻게 작동하나요?",
            "{topic}의 설계 특징은?"
        ]
    },
    "safety": {  # 안전 관련
        "keywords": ["안전", "사고", "대응", "비상", "방호", "보호"],
        "templates": [
            "{topic} 발생 시 대응 절차는?",
            "{topic}을 위한 안전 조치는?",
            "{topic}의 안전 기준은?",
            "{topic} 관련 주의사항은?"
        ]
    }
}


def connect_milvus():
    """Milvus 연결"""
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f" Milvus 연결 성공: {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f" Milvus 연결 실패: {e}")
        raise


def extract_chunks_by_doc(doc_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """특정 문서의 모든 청크 조회"""
    try:
        collection = Collection(MILVUS_COLLECTION)
        collection.load()

        # doc_id로 필터링하여 조회
        results = collection.query(
            expr=f'doc_id == "{doc_id}"',
            output_fields=["doc_id", "seq", "page", "section", "chunk"],
            limit=limit
        )

        print(f"  {doc_id}: {len(results)}개 청크 추출")
        return results

    except Exception as e:
        print(f"   {doc_id} 추출 실패: {e}")
        return []


def classify_chunk_type(chunk: Dict[str, Any]) -> str:
    """청크의 문서 유형 분류"""
    text = chunk.get('chunk', '').lower()
    section = chunk.get('section', '').lower()

    # 법 조항
    if any(kw in text for kw in QA_PATTERNS["law"]["keywords"]):
        if re.search(r'제\s*\d+\s*조', text) or re.search(r'제\s*\d+\s*항', text):
            return "law"

    # 안전 관련
    if any(kw in text for kw in QA_PATTERNS["safety"]["keywords"]):
        return "safety"

    # 매뉴얼/절차
    if any(kw in text for kw in QA_PATTERNS["manual"]["keywords"]):
        return "manual"

    # 기술문서
    if any(kw in text for kw in QA_PATTERNS["technical"]["keywords"]):
        return "technical"

    return "general"


def extract_topic(chunk: Dict[str, Any], chunk_type: str) -> str:
    """청크에서 핵심 주제 추출"""
    text = chunk.get('chunk', '')
    section = chunk.get('section', '')

    # 섹션 제목이 있으면 우선 사용
    if section and section not in ["Unknown", "META"]:
        return section.strip()

    # 법 조항: 조항 번호 추출
    if chunk_type == "law":
        match = re.search(r'(제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?)', text)
        if match:
            return match.group(1)

    # 첫 문장에서 명사구 추출 (간단한 휴리스틱)
    first_sentence = text.split('.')[0].split('\n')[0]
    if len(first_sentence) > 10 and len(first_sentence) < 100:
        return first_sentence.strip()

    return "관련 내용"


def generate_qa_samples(
    chunks: List[Dict[str, Any]],
    doc_id: str,
    augment: bool = True
) -> List[Dict[str, str]]:
    """청크로부터 QA 샘플 생성"""
    samples = []

    for chunk in chunks:
        text = chunk.get('chunk', '').strip()
        if not text or len(text) < 50:  # 너무 짧은 청크는 제외
            continue

        # 메타라인 제거
        text = re.sub(r'^META:.*?\n', '', text)

        chunk_type = classify_chunk_type(chunk)
        topic = extract_topic(chunk, chunk_type)
        section = chunk.get('section', '')
        page = chunk.get('page', 0)

        # 청크 타입별 템플릿 선택
        if chunk_type in QA_PATTERNS:
            templates = QA_PATTERNS[chunk_type]["templates"]
        else:
            templates = [
                "{topic}에 대해 설명해주세요.",
                "{topic}은 무엇인가요?",
                "{topic}의 내용을 요약해주세요."
            ]

        # 템플릿 적용하여 질문 생성
        for template in templates[:2]:  # 템플릿당 최대 2개
            instruction = template.format(topic=topic, section=section)

            # input 필드 구성 (문서 컨텍스트)
            input_parts = []
            if doc_id:
                input_parts.append(f"문서: {doc_id}")
            if section and section not in ["Unknown", "META"]:
                input_parts.append(f"섹션: {section}")
            if page:
                input_parts.append(f"페이지: {page}")

            input_text = ", ".join(input_parts)

            sample = {
                "instruction": instruction,
                "input": input_text,
                "output": text
            }

            samples.append(sample)

            # 데이터 증강
            if augment:
                # 질문 변형 생성
                variations = generate_question_variations(instruction, topic)
                for variation in variations:
                    samples.append({
                        "instruction": variation,
                        "input": input_text,
                        "output": text
                    })

    return samples


def generate_question_variations(original: str, topic: str) -> List[str]:
    """질문 변형 생성"""
    variations = []

    # 존댓말 <-> 반말
    if "주세요" in original:
        variations.append(original.replace("주세요", "줘"))
    elif "줘" in original:
        variations.append(original.replace("줘", "주세요"))

    # 의문형 변형
    if "무엇인가요?" in original:
        variations.append(original.replace("무엇인가요?", "뭔가요?"))
        variations.append(topic + "에 대한 정보를 알려주세요.")
    elif "어떻게" in original:
        variations.append(original.replace("어떻게", "어떤 방식으로"))

    # 간략 요청 추가
    if "설명해주세요" in original:
        variations.append(original.replace("설명해주세요", "간단히 설명해주세요"))

    return variations[:2]  # 최대 2개 변형


def filter_quality(samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """품질 필터링"""
    filtered = []

    for sample in samples:
        instruction = sample['instruction']
        output = sample['output']

        # 필터 조건
        if len(instruction) < 10:  # 너무 짧은 질문
            continue
        if len(output) < 30:  # 너무 짧은 답변
            continue
        if len(output) > 4000:  # 너무 긴 답변 (토큰 제한 고려)
            continue
        if instruction.count('?') > 3:  # 질문이 너무 복잡
            continue

        filtered.append(sample)

    return filtered


def deduplicate_samples(samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """중복 제거"""
    unique = []
    seen = set()

    for sample in samples:
        # instruction + output 앞부분으로 키 생성
        key = (sample['instruction'], sample['output'][:100])
        if key not in seen:
            seen.add(key)
            unique.append(sample)

    return unique


def save_jsonl(samples: List[Dict[str, str]], output_path: Path):
    """JSONL 형식으로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f" 데이터셋 저장 완료: {output_path}")
    print(f"   총 {len(samples)}개 샘플")


def print_statistics(samples: List[Dict[str, str]]):
    """데이터셋 통계 출력"""
    if not samples:
        return

    inst_lengths = [len(s['instruction']) for s in samples]
    out_lengths = [len(s['output']) for s in samples]

    print("\n 데이터셋 통계:")
    print(f"   총 샘플 수: {len(samples)}")
    print(f"   평균 질문 길이: {sum(inst_lengths) / len(inst_lengths):.1f} 문자")
    print(f"   평균 답변 길이: {sum(out_lengths) / len(out_lengths):.1f} 문자")
    print(f"   최대 답변 길이: {max(out_lengths)} 문자")
    print(f"   최소 답변 길이: {min(out_lengths)} 문자")

    # input 필드 사용 비율
    with_input = sum(1 for s in samples if s.get('input'))
    print(f"   컨텍스트 포함: {with_input}/{len(samples)} ({with_input/len(samples)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Milvus에서 학습 데이터 추출")
    parser.add_argument("--doc-ids", nargs="+", required=True, help="문서 ID 목록")
    parser.add_argument("--output", type=str, default="data/training_data.jsonl", help="출력 파일")
    parser.add_argument("--limit", type=int, default=500, help="문서당 최대 청크 수")
    parser.add_argument("--no-augment", action="store_true", help="데이터 증강 비활성화")

    args = parser.parse_args()

    print("="*80)
    print(" Milvus 학습 데이터 추출")
    print("="*80)
    print(f"문서 수: {len(args.doc_ids)}")
    print(f"문서당 최대 청크: {args.limit}")
    print(f"데이터 증강: {'비활성화' if args.no_augment else '활성화'}")
    print("="*80)

    # Milvus 연결
    connect_milvus()

    # 문서별 청크 추출
    all_samples = []

    for doc_id in args.doc_ids:
        print(f"\n처리 중: {doc_id}")
        chunks = extract_chunks_by_doc(doc_id, args.limit)

        if not chunks:
            continue

        # QA 샘플 생성
        samples = generate_qa_samples(chunks, doc_id, augment=not args.no_augment)
        print(f"   {len(samples)}개 QA 샘플 생성")

        all_samples.extend(samples)

    print(f"\n 총 {len(all_samples)}개 샘플 생성 (중복 포함)")

    # 품질 필터링
    print("\n 품질 필터링 중...")
    filtered = filter_quality(all_samples)
    print(f"   {len(all_samples)} → {len(filtered)}개 (제거: {len(all_samples) - len(filtered)})")

    # 중복 제거
    print("\n 중복 제거 중...")
    unique = deduplicate_samples(filtered)
    print(f"   {len(filtered)} → {len(unique)}개 (제거: {len(filtered) - len(unique)})")

    # 저장
    output_path = Path(args.output)
    save_jsonl(unique, output_path)

    # 통계 출력
    print_statistics(unique)

    # 샘플 출력
    print("\n 샘플 3개:")
    print("-"*80)
    for i, sample in enumerate(unique[:3]):
        print(f"\n[샘플 {i+1}]")
        print(f"질문: {sample['instruction']}")
        if sample.get('input'):
            print(f"입력: {sample['input']}")
        print(f"답변: {sample['output'][:200]}...")
        print("-"*80)

    print("\n 완료!")


if __name__ == "__main__":
    main()