# app/services/finetune_service.py
"""
파인튜닝 서비스
- Milvus에서 선택된 문서의 청크 데이터 추출
- QA 형식 학습 데이터셋 생성
- QLoRA 파인튜닝 실행
"""
import os
import json
import subprocess
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from datetime import datetime

from pymilvus import Collection, connections
from app.services.milvus_store_v2 import MilvusStoreV2
from app.services.embedding_model import get_sentence_embedding_dimension


# ==================== 데이터 추출 ====================

async def extract_training_data(
    doc_ids: List[str],
    output_path: Path,
    max_chunks_per_doc: int = 500
) -> int:
    """
    선택된 문서들로부터 학습 데이터 추출
    
    Args:
        doc_ids: 학습에 사용할 문서 ID 목록
        output_path: 출력 JSONL 파일 경로
        max_chunks_per_doc: 문서당 최대 청크 수
    
    Returns:
        생성된 학습 샘플 수
    """
    print(f"[FINETUNE-EXTRACT] Extracting data from {len(doc_ids)} documents...")
    
    # Milvus 연결
    dim = get_sentence_embedding_dimension()
    store = MilvusStoreV2(dim=dim)
    
    all_samples = []
    
    for doc_id in doc_ids:
        print(f"[FINETUNE-EXTRACT] Processing doc_id: {doc_id}")
        
        # 문서별 청크 조회
        try:
            chunks = store.query_by_doc(doc_id=doc_id, limit=max_chunks_per_doc)
            
            if not chunks:
                print(f"[FINETUNE-EXTRACT] Warning: No chunks found for {doc_id}")
                continue
            
            print(f"[FINETUNE-EXTRACT] Found {len(chunks)} chunks for {doc_id}")
            
            # QA 샘플 생성
            samples = _generate_qa_samples(doc_id, chunks)
            all_samples.extend(samples)
            
        except Exception as e:
            print(f"[FINETUNE-EXTRACT] Error processing {doc_id}: {e}")
            continue
    
    # JSONL 파일로 저장
    print(f"[FINETUNE-EXTRACT] Saving {len(all_samples)} samples to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"[FINETUNE-EXTRACT] Dataset created: {output_path}")
    return len(all_samples)


def _generate_qa_samples(doc_id: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    청크 데이터로부터 QA 형식 학습 샘플 생성
    
    전략:
    1. 각 청크를 컨텍스트로 사용
    2. 청크 내용 기반 질문-답변 쌍 생성
    3. 원자력 도메인 특화 프롬프트 템플릿 적용
    
    Returns:
        {"instruction": str, "input": str, "output": str} 형식의 샘플 리스트
    """
    samples = []
    
    for chunk in chunks:
        chunk_text = chunk.get('chunk', '')
        section = chunk.get('section', '')
        page = chunk.get('page', 0)
        
        if not chunk_text or len(chunk_text) < 50:
            continue
        
        # 청크 내용을 요약한 답변으로 사용
        # 실제로는 더 정교한 QA 생성 로직 필요 (LLM 기반 생성 등)
        
        # 패턴 1: 정의 설명 (청크에 "정의", "의미" 등이 포함된 경우)
        if any(keyword in chunk_text for keyword in ['정의', '의미', '개념', 'definition']):
            # 첫 문장에서 주제 추출 시도
            first_sentence = chunk_text.split('.')[0].split('\n')[0]
            if len(first_sentence) > 10:
                instruction = f"{first_sentence}에 대해 설명해주세요."
                samples.append({
                    "instruction": instruction,
                    "input": f"문서: {doc_id}, 섹션: {section}",
                    "output": chunk_text.strip()
                })
        
        # 패턴 2: 절차/방법 (청크에 "절차", "방법", "단계" 등이 포함된 경우)
        if any(keyword in chunk_text for keyword in ['절차', '방법', '단계', 'procedure', 'step']):
            instruction = f"{section}의 절차에 대해 설명해주세요."
            samples.append({
                "instruction": instruction,
                "input": f"문서: {doc_id}",
                "output": chunk_text.strip()
            })
        
        # 패턴 3: 기준/요구사항 (청크에 "기준", "요구사항", "한도" 등이 포함된 경우)
        if any(keyword in chunk_text for keyword in ['기준', '요구사항', '한도', '제한', 'requirement', 'limit']):
            instruction = f"{section}의 요구사항은 무엇인가요?"
            samples.append({
                "instruction": instruction,
                "input": f"문서: {doc_id}, 페이지: {page}",
                "output": chunk_text.strip()
            })
        
        # 패턴 4: 일반 QA (모든 청크에 대해)
        # 섹션 제목이 있으면 섹션 기반 질문 생성
        if section and section != "Unknown":
            instruction = f"{section}에 대해 설명해주세요."
            samples.append({
                "instruction": instruction,
                "input": f"문서: {doc_id}",
                "output": chunk_text.strip()
            })
        else:
            # 섹션이 없으면 첫 50자를 이용한 질문 생성
            preview = chunk_text[:50].strip()
            if preview:
                instruction = f"다음 내용에 대해 자세히 설명해주세요: {preview}..."
                samples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": chunk_text.strip()
                })
    
    # 중복 제거 (동일한 instruction-output 쌍)
    unique_samples = []
    seen = set()
    for sample in samples:
        key = (sample['instruction'], sample['output'][:100])
        if key not in seen:
            seen.add(key)
            unique_samples.append(sample)
    
    print(f"[FINETUNE-EXTRACT] Generated {len(unique_samples)} unique QA samples from {len(chunks)} chunks")
    return unique_samples


# ==================== QLoRA 학습 실행 ====================

async def run_qlora_training(
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    lora_r: int = 16,
    lora_alpha: int = 32,
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    progress_callback: Optional[Callable] = None
):
    """
    QLoRA 파인튜닝 실행
    
    실제 학습은 별도 Python 스크립트 (train_qlora.py) 실행
    Docker 컨테이너나 서브프로세스로 실행
    """
    print(f"[FINETUNE-TRAIN] Starting QLoRA training...")
    print(f"[FINETUNE-TRAIN] Model: {model_name}")
    print(f"[FINETUNE-TRAIN] Dataset: {dataset_path}")
    print(f"[FINETUNE-TRAIN] Output: {output_dir}")
    
    # 학습 스크립트 경로
    train_script = Path("/workspace/finetune/train_qlora.py")
    
    # 환경변수 설정
    env = os.environ.copy()
    env.update({
        "MODEL_NAME": model_name,
        "DATASET_PATH": str(dataset_path),
        "OUTPUT_DIR": str(output_dir),
        "LORA_R": str(lora_r),
        "LORA_ALPHA": str(lora_alpha),
        "NUM_EPOCHS": str(num_epochs),
        "BATCH_SIZE": str(batch_size),
        "LEARNING_RATE": str(learning_rate),
        "MAX_SEQ_LENGTH": "2048",
        "GRADIENT_ACCUMULATION": "8",
        "LORA_DROPOUT": "0.05"
    })
    
    # 학습 스크립트 실행 (블로킹)
    # 실제 운영에서는 별도 컨테이너나 비동기 프로세스로 실행 권장
    try:
        print(f"[FINETUNE-TRAIN] Executing: python {train_script}")
        
        # 서브프로세스로 학습 실행
        process = subprocess.Popen(
            ["python", str(train_script)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 실시간 로그 출력 및 진행률 파싱
        for line in process.stdout:
            print(line.strip())
            
            # 진행률 콜백 (로그에서 epoch/step 파싱)
            if progress_callback and "Epoch" in line:
                try:
                    # 예: "Epoch 1/3, Step 100/500"
                    if "Epoch" in line and "/" in line:
                        parts = line.split("Epoch")[1].split(",")
                        epoch_part = parts[0].strip()
                        current_epoch, total_epochs = map(int, epoch_part.split("/"))
                        
                        if "Step" in line:
                            step_part = parts[1].split("Step")[1].strip()
                            current_step, total_steps = map(int, step_part.split("/"))
                            progress_callback(current_epoch, total_epochs, current_step, total_steps)
                except:
                    pass
        
        # 프로세스 종료 대기
        return_code = process.wait()
        
        if return_code != 0:
            raise RuntimeError(f"Training failed with return code {return_code}")
        
        print(f"[FINETUNE-TRAIN] Training completed successfully")
        print(f"[FINETUNE-TRAIN] Model saved to: {output_dir}")
        
    except Exception as e:
        print(f"[FINETUNE-TRAIN] Training failed: {e}")
        raise


# ==================== 데이터 품질 검증 ====================

def validate_training_data(dataset_path: Path) -> Dict[str, Any]:
    """
    학습 데이터 품질 검증
    
    Returns:
        검증 결과 딕셔너리
    """
    print(f"[FINETUNE-VALIDATE] Validating dataset: {dataset_path}")
    
    stats = {
        "total_samples": 0,
        "avg_instruction_length": 0,
        "avg_output_length": 0,
        "min_output_length": float('inf'),
        "max_output_length": 0,
        "samples_with_input": 0,
        "unique_instructions": 0
    }
    
    instruction_lengths = []
    output_lengths = []
    unique_instructions = set()
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            stats["total_samples"] += 1
            
            instruction = sample.get('instruction', '')
            input_text = sample.get('input', '')
            output = sample.get('output', '')
            
            instruction_lengths.append(len(instruction))
            output_lengths.append(len(output))
            unique_instructions.add(instruction)
            
            if input_text:
                stats["samples_with_input"] += 1
            
            stats["min_output_length"] = min(stats["min_output_length"], len(output))
            stats["max_output_length"] = max(stats["max_output_length"], len(output))
    
    if stats["total_samples"] > 0:
        stats["avg_instruction_length"] = sum(instruction_lengths) / len(instruction_lengths)
        stats["avg_output_length"] = sum(output_lengths) / len(output_lengths)
        stats["unique_instructions"] = len(unique_instructions)
    
    print(f"[FINETUNE-VALIDATE] Validation results:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Unique instructions: {stats['unique_instructions']}")
    print(f"  Avg instruction length: {stats['avg_instruction_length']:.1f}")
    print(f"  Avg output length: {stats['avg_output_length']:.1f}")
    print(f"  Output length range: [{stats['min_output_length']}, {stats['max_output_length']}]")
    
    return stats