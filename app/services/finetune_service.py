# app/services/finetune_service.py
"""
파인튜닝 서비스 (완전 구현)
- 기존 extract_english_first.py / extract_structured_compliance.py 활용
- L40S 최적화 LoRA 학습 실행
- 실시간 진행률 업데이트
"""
import os
import json
import subprocess
import re
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from datetime import datetime


# ==================== 데이터 추출 (기존 스크립트 활용) ====================

async def extract_training_data(
    doc_ids: List[str],
    output_path: Path,
    strategy: str = "balanced",  # "english_first", "structured", "balanced"
    total_samples: int = 5000
) -> int:
    """
    기존 추출 스크립트를 활용한 학습 데이터 생성
    
    Args:
        doc_ids: 학습에 사용할 문서 ID 목록
        output_path: 출력 JSONL 파일 경로
        strategy: 추출 전략
            - "english_first": extract_english_first.py (70:20:10 언어 우선순위)
            - "structured": extract_structured_compliance.py (60:40 구조화/컴플라이언스)
            - "balanced": 둘 다 사용 (60% english_first + 40% structured)
        total_samples: 총 샘플 수
    
    Returns:
        생성된 학습 샘플 수
    """
    print(f"[FINETUNE-EXTRACT] Strategy: {strategy}")
    print(f"[FINETUNE-EXTRACT] Documents: {doc_ids}")
    print(f"[FINETUNE-EXTRACT] Target samples: {total_samples}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 전략별 실행
    if strategy == "english_first":
        return await _run_english_first_extraction(doc_ids, output_path, total_samples)
    
    elif strategy == "structured":
        return await _run_structured_extraction(doc_ids, output_path, total_samples)
    
    elif strategy == "balanced":
        return await _run_balanced_extraction(doc_ids, output_path, total_samples)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


async def _run_english_first_extraction(
    doc_ids: List[str],
    output_path: Path,
    total_samples: int
) -> int:
    """
    extract_english_first.py 실행
    70% English, 20% Korean Native, 10% Korean Translation
    """
    print(f"[FINETUNE-EXTRACT] Running extract_english_first.py...")
    
    script_path = Path("/workspace/finetune/extract_english_first.py")
    
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    # 명령어 구성
    cmd = [
        "python",
        str(script_path),
        "--doc-ids", *doc_ids,
        "--output-dir", str(output_path.parent),
        "--total-samples", str(total_samples),
        "--combined"
    ]
    
    # 실행
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        # combined 파일 찾기
        combined_file = output_path.parent / "training_combined.jsonl"
        
        if combined_file.exists():
            # 파일 이동
            import shutil
            shutil.move(str(combined_file), str(output_path))
            
            # 샘플 수 계산
            sample_count = 0
            with open(output_path, 'r') as f:
                for _ in f:
                    sample_count += 1
            
            print(f"[FINETUNE-EXTRACT] Generated {sample_count} samples")
            return sample_count
        else:
            raise FileNotFoundError(f"Output file not found: {combined_file}")
    
    except subprocess.CalledProcessError as e:
        print(f"[FINETUNE-EXTRACT] Error: {e.stderr}")
        raise


async def _run_structured_extraction(
    doc_ids: List[str],
    output_path: Path,
    total_samples: int
) -> int:
    """
    extract_structured_compliance.py 실행
    60% Structured Extraction, 40% Compliance Mapping
    """
    print(f"[FINETUNE-EXTRACT] Running extract_structured_compliance.py...")
    
    script_path = Path("/workspace/finetune/extract_structured_compliance.py")
    
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    # 명령어 구성
    cmd = [
        "python",
        str(script_path),
        "--doc-ids", *doc_ids,
        "--output", str(output_path),
        "--total-samples", str(total_samples)
    ]
    
    # 실행
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        # 샘플 수 계산
        sample_count = 0
        with open(output_path, 'r') as f:
            for _ in f:
                sample_count += 1
        
        print(f"[FINETUNE-EXTRACT] Generated {sample_count} samples")
        return sample_count
    
    except subprocess.CalledProcessError as e:
        print(f"[FINETUNE-EXTRACT] Error: {e.stderr}")
        raise


async def _run_balanced_extraction(
    doc_ids: List[str],
    output_path: Path,
    total_samples: int
) -> int:
    """
    Balanced: 60% english_first + 40% structured
    """
    print(f"[FINETUNE-EXTRACT] Running balanced extraction (60:40)...")
    
    # 임시 파일 경로
    temp_dir = output_path.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    english_file = temp_dir / "english_first.jsonl"
    structured_file = temp_dir / "structured.jsonl"
    
    # 1. English-first (60%)
    english_samples = int(total_samples * 0.6)
    await _run_english_first_extraction(doc_ids, english_file, english_samples)
    
    # 2. Structured (40%)
    structured_samples = int(total_samples * 0.4)
    await _run_structured_extraction(doc_ids, structured_file, structured_samples)
    
    # 3. 통합
    print(f"[FINETUNE-EXTRACT] Merging datasets...")
    
    combined_samples = []
    
    # English-first 읽기
    with open(english_file, 'r', encoding='utf-8') as f:
        for line in f:
            combined_samples.append(json.loads(line))
    
    # Structured 읽기
    with open(structured_file, 'r', encoding='utf-8') as f:
        for line in f:
            combined_samples.append(json.loads(line))
    
    # 셔플
    import random
    random.shuffle(combined_samples)
    
    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in combined_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 임시 파일 삭제
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"[FINETUNE-EXTRACT] Generated {len(combined_samples)} total samples")
    return len(combined_samples)


# ==================== LoRA 학습 실행 ====================

async def run_lora_training(
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    lora_r: int = 32,
    lora_alpha: int = 64,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    progress_callback: Optional[Callable] = None
):
    """
    L40S 최적화 LoRA 파인튜닝 실행
    
    Args:
        model_name: 베이스 모델 (Qwen/Qwen2.5-14B-Instruct)
        dataset_path: 학습 데이터 경로
        output_dir: 출력 디렉토리
        lora_r: LoRA rank (L40S: 32)
        lora_alpha: LoRA alpha (L40S: 64)
        num_epochs: Epoch 수
        batch_size: Batch size (L40S: 4)
        learning_rate: Learning rate
        progress_callback: 진행률 콜백 함수
    """
    print(f"[FINETUNE-TRAIN] Starting L40S optimized LoRA training...")
    print(f"[FINETUNE-TRAIN] Model: {model_name}")
    print(f"[FINETUNE-TRAIN] Dataset: {dataset_path}")
    print(f"[FINETUNE-TRAIN] Output: {output_dir}")
    
    # L40S 최적화 학습 스크립트
    train_script = Path("/workspace/finetune/train_lora_l40s.py")
    
    if not train_script.exists():
        # Fallback: QLoRA 스크립트
        train_script = Path("/workspace/finetune/train_qlora.py")
        print(f"[FINETUNE-TRAIN] Warning: Using QLoRA script (train_lora_l40s.py not found)")
    
    # 환경변수 설정
    env = os.environ.copy()
    env.update({
        # 모델 & 데이터
        "MODEL_NAME": model_name,
        "DATASET_PATH": str(dataset_path),
        "OUTPUT_DIR": str(output_dir),
        
        # L40S 최적 LoRA 설정
        "LORA_R": str(lora_r),
        "LORA_ALPHA": str(lora_alpha),
        "LORA_DROPOUT": "0.05",
        
        # L40S 최적 학습 설정
        "BATCH_SIZE": str(batch_size),
        "GRADIENT_ACCUMULATION": "4",
        "NUM_EPOCHS": str(num_epochs),
        "LEARNING_RATE": str(learning_rate),
        "MAX_SEQ_LENGTH": "4096",  # L40S: 4096
        "USE_GRAD_CHECKPOINT": "0",  # L40S: 메모리 충분
    })
    
    # 서브프로세스로 학습 실행
    try:
        print(f"[FINETUNE-TRAIN] Executing: python {train_script}")
        
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
            
            # 진행률 콜백
            if progress_callback:
                try:
                    # 진행률 파싱
                    # 예: "Epoch 1/3, Step 100/500 (20%), Loss: 1.234"
                    if "Epoch" in line and "/" in line:
                        epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
                        step_match = re.search(r'Step\s+(\d+)/(\d+)', line)
                        
                        if epoch_match and step_match:
                            current_epoch = int(epoch_match.group(1))
                            total_epochs = int(epoch_match.group(2))
                            current_step = int(step_match.group(1))
                            total_steps = int(step_match.group(2))
                            
                            # 전체 진행률 계산
                            epoch_progress = (current_epoch - 1) / total_epochs
                            step_progress = current_step / total_steps / total_epochs
                            overall_progress = (epoch_progress + step_progress) * 100
                            
                            progress_callback(overall_progress, line.strip())
                        
                        elif epoch_match:
                            # Step 없이 Epoch만 있는 경우
                            current_epoch = int(epoch_match.group(1))
                            total_epochs = int(epoch_match.group(2))
                            overall_progress = (current_epoch / total_epochs) * 100
                            
                            progress_callback(overall_progress, line.strip())
                
                except Exception as e:
                    print(f"[FINETUNE-TRAIN] Progress parsing error: {e}")
        
        # 프로세스 완료 대기
        return_code = process.wait()
        
        if return_code != 0:
            raise Exception(f"Training failed with return code {return_code}")
        
        print(f"[FINETUNE-TRAIN] Training completed successfully!")
        print(f"[FINETUNE-TRAIN] Output saved to: {output_dir}")
        
    except Exception as e:
        print(f"[FINETUNE-TRAIN] Training failed: {e}")
        raise