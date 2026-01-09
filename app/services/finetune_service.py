# app/services/finetune_service.py
"""
파인튜닝 서비스 (완전 구현 - 모듈화 강화 버전)
- 동적 경로 탐색으로 extract_english_first.py / extract_structured_compliance.py 활용
- L40S 최적화 LoRA 학습 실행
- 실시간 진행률 업데이트
- 프로젝트 구조 독립적 설계
"""
import os
import json
import subprocess
import re
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from datetime import datetime

def _env_get(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else default

# ==================== 경로 탐색 유틸리티 ====================

def find_project_root() -> Path:
    """
    프로젝트 루트 디렉토리 찾기
    현재 파일 위치: app/services/finetune_service.py
    프로젝트 루트: 2단계 상위
    """
    return Path(__file__).parent.parent.parent


def find_extraction_script(script_name: str) -> Path:
    """
    추출 스크립트 경로 동적 탐색 (Docker 환경 대응)
    
    탐색 순서:
    1. 환경변수 FINETUNE_SCRIPTS_DIR (최우선)
    2. /workspace/finetune/{script_name} (Docker 컨테이너)
    3. {project_root}/finetune/{script_name} (로컬)
    4. {project_root}/scripts/{script_name}
    5. {project_root}/{script_name}
    
    Args:
        script_name: 스크립트 파일명 (예: "extract_english_first.py")
    
    Returns:
        스크립트 경로 (Path 객체)
    
    Raises:
        FileNotFoundError: 스크립트를 찾을 수 없을 때
    """
    project_root = find_project_root()
    
    # 탐색 경로 목록
    search_paths = []
    
    # 1. 환경변수로 커스텀 경로 (최우선)
    custom_finetune_dir = os.getenv("FINETUNE_SCRIPTS_DIR")
    if custom_finetune_dir:
        search_paths.append(Path(custom_finetune_dir) / script_name)
    
    # 2. Docker 컨테이너 표준 경로 (/workspace/finetune)
    workspace_finetune = Path("/workspace/finetune") / script_name
    search_paths.append(workspace_finetune)
    
    # 3. 프로젝트 루트 기준 경로들
    search_paths.extend([
        project_root / "finetune" / script_name,
        project_root / "scripts" / script_name,
        project_root / script_name,
    ])
    
    # 4. /app/finetune 도 체크 (컨테이너에 마운트된 경우)
    app_finetune = Path("/app/finetune") / script_name
    if app_finetune.parent.exists():  # /app/finetune 디렉토리가 존재하면
        search_paths.insert(2, app_finetune)  # Docker 경로 다음에 추가
    
    for path in search_paths:
        if path.exists():
            print(f"[FINETUNE-SERVICE] Found script: {path}")
            return path
    
    # 찾지 못한 경우
    search_paths_str = "\n  - ".join(str(p) for p in search_paths)
    raise FileNotFoundError(
        f"Script '{script_name}' not found in any of these locations:\n  - {search_paths_str}\n\n"
        f"Solutions:\n"
        f"  1. Mount finetune folder in docker-compose.yml:\n"
        f"     volumes:\n"
        f"       - ./finetune:/app/finetune:ro\n"
        f"  2. Set environment variable:\n"
        f"     FINETUNE_SCRIPTS_DIR=/workspace/finetune\n"
        f"  3. Copy scripts to container:\n"
        f"     COPY ./finetune /workspace/finetune\n"
    )


def find_training_script(script_name: str = "train_lora_l40s.py") -> Path:
    """
    학습 스크립트 경로 동적 탐색
    
    Fallback: train_lora_l40s.py -> train_qlora.py
    """
    try:
        return find_extraction_script(script_name)
    except FileNotFoundError:
        if script_name == "train_lora_l40s.py":
            print(f"[FINETUNE-SERVICE] {script_name} not found, trying fallback: train_qlora.py")
            return find_extraction_script("train_qlora.py")
        raise


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
    
    # 동적 스크립트 경로 탐색
    script_path = find_extraction_script("extract_english_first.py")
    
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
            with open(output_path, 'r', encoding='utf-8') as f:
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
    
    # 동적 스크립트 경로 탐색
    script_path = find_extraction_script("extract_structured_compliance.py")
    
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
        with open(output_path, 'r', encoding='utf-8') as f:
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
    
    # ========== Finetune 컨테이너 사용 여부 확인 ==========
    use_finetune_container = os.getenv("USE_FINETUNE_CONTAINER", "1") == "1"
    finetune_container_name = os.getenv("FINETUNE_CONTAINER_NAME", "nuclear-finetune")
    
    if use_finetune_container:
        print(f"[FINETUNE-TRAIN] Using external finetune container: {finetune_container_name}")
        return await _run_training_in_container(
            container_name=finetune_container_name,
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            progress_callback=progress_callback
        )
    else:
        print(f"[FINETUNE-TRAIN] Using local training (not recommended)")
        return await _run_training_local(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            progress_callback=progress_callback
        )


async def _run_training_in_container(
    container_name: str,
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    lora_r: int,
    lora_alpha: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    progress_callback: Optional[Callable] = None
):
    """
    외부 Finetune 컨테이너에서 학습 실행 (Docker Python SDK 사용)
    """
    import docker
    
    print(f"[FINETUNE-TRAIN] Executing in container: {container_name}")
    
    try:
        # Docker 클라이언트 연결
        client = docker.from_env()
        container = client.containers.get(container_name)
        # 환경변수 우선순위:
        # 1) 호출 인자(model_name, dataset_path, output_dir, lora_r, lora_alpha, num_epochs, batch_size, learning_rate)
        # 2) Portainer/Stack에 설정된 OS env
        # 3) 최종 fallback
        env_vars = {
            "MODEL_NAME": model_name,
            "DATASET_PATH": str(dataset_path),
            "OUTPUT_DIR": str(output_dir),

            "LORA_R": str(lora_r),
            "LORA_ALPHA": str(lora_alpha),
            "LORA_DROPOUT": _env_get("LORA_DROPOUT", "0.05"),

            # 하드코딩 제거하고, 스택 env를 존중
            "BATCH_SIZE": str(batch_size),
            "GRADIENT_ACCUMULATION": _env_get("GRADIENT_ACCUMULATION", "16"),
            "NUM_EPOCHS": str(num_epochs),
            "LEARNING_RATE": str(learning_rate),

            "MAX_SEQ_LENGTH": _env_get("MAX_SEQ_LENGTH", "1024"),
            "USE_GRAD_CHECKPOINT": _env_get("USE_GRAD_CHECKPOINT", "1"),

            # 옵션: QLoRA/LoRA 스크립트에서 쓸 수도 있는 최적화 env
            "OPTIM": _env_get("OPTIM", "paged_adamw_8bit"),
            "PYTORCH_CUDA_ALLOC_CONF": _env_get(
                "PYTORCH_CUDA_ALLOC_CONF",
                "max_split_size_mb:128,"
            ),
        }


        def _sh(v: str) -> str:
            return "'" + v.replace("'", "'\"'\"'") + "'"

        env_string = " ".join([f"{k}={_sh(str(v))}" for k, v in env_vars.items()])
        train_path = _env_get("FINETUNE_TRAIN_SCRIPT", "/app/finetune/train_lora_l40s.py")
        print(f"[FINETUNE-TRAIN] Command: python {train_path}")
        command = f"bash -lc \"{env_string} python {_sh(train_path)}\""

        
        print(f"[FINETUNE-TRAIN] Full command: {command}")
        
        res = container.exec_run(command, stream=False, demux=False)
        if isinstance(res, tuple):
            exit_code, output = res
        else:
            exit_code, output = res.exit_code, res.output
        
        # 출력 디코딩 및 출력
        if output:
            output_text = output.decode('utf-8', errors='ignore')
            print(output_text)
            
            # 진행률 파싱 (전체 출력에서)
            if progress_callback:
                for line in output_text.split('\n'):
                    try:
                        if "Epoch" in line and "/" in line:
                            epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
                            step_match = re.search(r'Step\s+(\d+)/(\d+)', line)
                            
                            if epoch_match and step_match:
                                current_epoch = int(epoch_match.group(1))
                                total_epochs = int(epoch_match.group(2))
                                current_step = int(step_match.group(1))
                                total_steps = int(step_match.group(2))
                                
                                epoch_progress = (current_epoch - 1) / total_epochs
                                step_progress = current_step / total_steps / total_epochs
                                overall_progress = (epoch_progress + step_progress) * 100
                                
                                progress_callback(overall_progress, line)
                            
                            elif epoch_match:
                                current_epoch = int(epoch_match.group(1))
                                total_epochs = int(epoch_match.group(2))
                                overall_progress = (current_epoch / total_epochs) * 100
                                
                                progress_callback(overall_progress, line)
                    
                    except Exception as e:
                        print(f"[FINETUNE-TRAIN] Progress parsing error: {e}")
        
        # Exit code 확인
        if exit_code != 0:
            raise Exception(f"Training failed with exit code {exit_code}")
        
        print(f"[FINETUNE-TRAIN] Training completed successfully!")
        print(f"[FINETUNE-TRAIN] Output saved to: {output_dir}")
        
    except docker.errors.NotFound:
        error_msg = f"Container '{container_name}' not found. Please start nuclear-finetune stack first."
        print(f"[FINETUNE-TRAIN] Error: {error_msg}")
        raise Exception(error_msg)
    
    except docker.errors.APIError as e:
        error_msg = f"Docker API error: {e}"
        print(f"[FINETUNE-TRAIN] Error: {error_msg}")
        raise Exception(error_msg)
    
    except Exception as e:
        print(f"[FINETUNE-TRAIN] Training failed: {e}")
        raise


async def _run_training_local(
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    lora_r: int,
    lora_alpha: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    progress_callback: Optional[Callable] = None
):
    """
    로컬에서 학습 실행 (비권장 - 라이브러리 없을 수 있음)
    """
    print(f"[FINETUNE-TRAIN] Warning: Running training locally (may fail if peft not installed)")
    
    # L40S 최적화 학습 스크립트 (Fallback 포함)
    train_script = find_training_script("train_lora_l40s.py")
    
    # 환경변수 설정
    env = os.environ.copy()
    env.update({
        "MODEL_NAME": model_name,
        "DATASET_PATH": str(dataset_path),
        "OUTPUT_DIR": str(output_dir),

        "LORA_R": str(lora_r),
        "LORA_ALPHA": str(lora_alpha),
        "LORA_DROPOUT": _env_get("LORA_DROPOUT", "0.05"),

        "BATCH_SIZE": str(batch_size),
        "GRADIENT_ACCUMULATION": _env_get("GRADIENT_ACCUMULATION", "16"),
        "NUM_EPOCHS": str(num_epochs),
        "LEARNING_RATE": str(learning_rate),

        "MAX_SEQ_LENGTH": _env_get("MAX_SEQ_LENGTH", "1024"),
        "USE_GRAD_CHECKPOINT": _env_get("USE_GRAD_CHECKPOINT", "1"),

        "OPTIM": _env_get("OPTIM", "paged_adamw_8bit"),
        "PYTORCH_CUDA_ALLOC_CONF": _env_get(
            "PYTORCH_CUDA_ALLOC_CONF",
            "max_split_size_mb:128"
        ),
    })

    
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
            
            if progress_callback:
                try:
                    if "Epoch" in line and "/" in line:
                        epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
                        step_match = re.search(r'Step\s+(\d+)/(\d+)', line)
                        
                        if epoch_match and step_match:
                            current_epoch = int(epoch_match.group(1))
                            total_epochs = int(epoch_match.group(2))
                            current_step = int(step_match.group(1))
                            total_steps = int(step_match.group(2))
                            
                            epoch_progress = (current_epoch - 1) / total_epochs
                            step_progress = current_step / total_steps / total_epochs
                            overall_progress = (epoch_progress + step_progress) * 100
                            
                            progress_callback(overall_progress, line.strip())
                        
                        elif epoch_match:
                            current_epoch = int(epoch_match.group(1))
                            total_epochs = int(epoch_match.group(2))
                            overall_progress = (current_epoch / total_epochs) * 100
                            
                            progress_callback(overall_progress, line.strip())
                
                except Exception as e:
                    print(f"[FINETUNE-TRAIN] Progress parsing error: {e}")
        
        return_code = process.wait()
        
        if return_code != 0:
            raise Exception(f"Training failed with return code {return_code}")
        
        print(f"[FINETUNE-TRAIN] Training completed successfully!")
        print(f"[FINETUNE-TRAIN] Output saved to: {output_dir}")
        
    except Exception as e:
        print(f"[FINETUNE-TRAIN] Training failed: {e}")
        raise