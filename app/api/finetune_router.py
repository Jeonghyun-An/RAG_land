# app/api/finetune_router.py
"""
파인튜닝 API 엔드포인트 (완전 구현 v3 - 모듈화 강화)
- finetune_service.py 의존성 주입
- 추출 전략 선택 가능
- L40S 최적화 LoRA 파인튜닝
- 실시간 진행률 업데이트
- 상태 관리 강화
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import json
from pathlib import Path

# 파인튜닝 서비스 임포트
from app.services.finetune_service import (
    extract_training_data,
    run_lora_training
)

router = APIRouter(prefix="/finetune", tags=["finetune"])

# ==================== 설정 ====================

# 파인튜닝 작업 상태 저장소 (인메모리)
finetune_jobs: Dict[str, Dict[str, Any]] = {}

# 학습 결과 저장 경로
FINETUNE_OUTPUT_DIR = Path(os.getenv("FINETUNE_OUTPUT_DIR", "/workspace/output"))
FINETUNE_DATA_DIR = Path(os.getenv("FINETUNE_DATA_DIR", "/workspace/data"))

# 디렉토리 생성
FINETUNE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINETUNE_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Request/Response Models ====================

class FinetuneStartRequest(BaseModel):
    """파인튜닝 시작 요청"""
    doc_ids: List[str]
    
    # 데이터 추출 전략
    extraction_strategy: Optional[str] = "balanced"  # "english_first", "structured", "balanced"
    total_samples: Optional[int] = 5000
    
    # 모델 설정
    model_name: Optional[str] = None  # None이면 환경변수에서 읽음
    lora_r: Optional[int] = None  # None이면 환경변수에서 읽음
    lora_alpha: Optional[int] = None
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    output_name: Optional[str] = None


class FinetuneStatusResponse(BaseModel):
    """파인튜닝 상태 응답"""
    job_id: str
    status: str  # pending, extracting, training, completed, failed
    progress: float  # 0-100
    current_step: Optional[str] = None
    doc_ids: List[str]
    extraction_strategy: Optional[str] = None
    dataset_size: Optional[int] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    output_path: Optional[str] = None


class FinetuneModel(BaseModel):
    """파인튜닝된 모델 정보"""
    name: str
    path: str
    base_model: str
    created_at: str
    dataset_size: Optional[int] = None
    doc_ids: Optional[List[str]] = None
    extraction_strategy: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


# ==================== Endpoints ====================

@router.post("/start", response_model=Dict[str, str])
async def start_finetuning(
    request: FinetuneStartRequest,
    background_tasks: BackgroundTasks
):
    """
    파인튜닝 작업 시작
    
    추출 전략:
    - "english_first": 언어 우선순위 (70% English, 20% Korean Native, 10% Translation)
    - "structured": 구조화 추출 (60% JSON/QA, 40% Compliance)
    - "balanced": 균형 (60% english_first + 40% structured)
    
    프로세스:
    1. 선택된 전략으로 학습 데이터 생성
    2. L40S 최적화 LoRA 파인튜닝 실행
    3. 결과 모델 저장
    """
    if not request.doc_ids:
        raise HTTPException(400, "doc_ids가 비어있습니다")
    
    # 전략 검증
    valid_strategies = ["english_first", "structured", "balanced"]
    if request.extraction_strategy not in valid_strategies:
        raise HTTPException(
            400, 
            f"유효하지 않은 전략: {request.extraction_strategy}. "
            f"사용 가능: {', '.join(valid_strategies)}"
        )
    
    # Job ID 생성
    job_id = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 출력 이름 결정
    if not request.output_name:
        # model_name이 None일 수 있으므로 환경변수 기본값 사용
        default_model = os.getenv("MODEL_NAME", "Qwen2.5-7B-Instruct")
        model_short = (request.model_name or default_model).split('/')[-1]
        output_name = f"{model_short}_{request.extraction_strategy}_{job_id}"
    else:
        output_name = request.output_name
    
    # Job 상태 초기화
    finetune_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "current_step": "초기화 중...",
        "doc_ids": request.doc_ids,
        "extraction_strategy": request.extraction_strategy,
        "dataset_size": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "output_path": None,
        "config": request.dict()
    }
    
    # 백그라운드 작업 시작
    background_tasks.add_task(
        run_finetuning_pipeline,
        job_id=job_id,
        config=request
    )
    
    return {
        "job_id": job_id,
        "message": "파인튜닝 작업이 시작되었습니다.",
        "output_name": output_name
    }


@router.get("/status/{job_id}", response_model=FinetuneStatusResponse)
async def get_finetuning_status(job_id: str):
    """파인튜닝 작업 상태 조회"""
    if job_id not in finetune_jobs:
        raise HTTPException(404, f"작업을 찾을 수 없습니다: {job_id}")
    
    job = finetune_jobs[job_id]
    return FinetuneStatusResponse(**job)


@router.get("/jobs", response_model=List[FinetuneStatusResponse])
async def list_finetuning_jobs():
    """모든 파인튜닝 작업 목록"""
    return [FinetuneStatusResponse(**job) for job in finetune_jobs.values()]


@router.get("/models", response_model=List[FinetuneModel])
async def list_finetuned_models():
    """파인튜닝된 모델 목록 조회"""
    models = []
    
    if not FINETUNE_OUTPUT_DIR.exists():
        return models
    
    for model_dir in FINETUNE_OUTPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        
        metadata_file = model_dir / "finetune_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                models.append(FinetuneModel(
                    name=model_dir.name,
                    path=str(model_dir),
                    base_model=metadata.get("base_model", "unknown"),
                    created_at=metadata.get("created_at", "unknown"),
                    dataset_size=metadata.get("dataset_size"),
                    doc_ids=metadata.get("doc_ids"),
                    extraction_strategy=metadata.get("extraction_strategy"),
                    config=metadata.get("config")
                ))
            except Exception as e:
                print(f"메타데이터 읽기 실패: {model_dir.name} - {e}")
                continue
        else:
            models.append(FinetuneModel(
                name=model_dir.name,
                path=str(model_dir),
                base_model="unknown",
                created_at=datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat()
            ))
    
    models.sort(key=lambda x: x.created_at, reverse=True)
    return models


@router.delete("/job/{job_id}")
async def delete_finetuning_job(job_id: str):
    """파인튜닝 작업 삭제"""
    if job_id not in finetune_jobs:
        raise HTTPException(404, f"작업을 찾을 수 없습니다: {job_id}")
    
    job = finetune_jobs[job_id]
    
    if job["status"] in ["extracting", "training"]:
        raise HTTPException(400, "실행 중인 작업은 삭제할 수 없습니다")
    
    del finetune_jobs[job_id]
    return {"message": f"작업이 삭제되었습니다: {job_id}"}


# ==================== Background Task ====================

async def run_finetuning_pipeline(job_id: str, config: FinetuneStartRequest):
    """
    파인튜닝 파이프라인 실행 (백그라운드)
    
    단계:
    1. 선택된 전략으로 데이터 추출 (10-30%)
    2. L40S 최적화 LoRA 파인튜닝 (30-95%)
    3. 메타데이터 저장 및 완료 (95-100%)
    """
    
    # ========== 환경변수 기본값 적용 ==========
    model_name = config.model_name or os.getenv("FT_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    lora_r = config.lora_r or int(os.getenv("LORA_R", "8"))
    lora_alpha = config.lora_alpha or int(os.getenv("LORA_ALPHA", "32"))
    num_epochs = config.num_epochs or int(os.getenv("NUM_EPOCHS", "3"))
    batch_size = config.batch_size or int(os.getenv("FT_BATCH_SIZE", "1"))
    learning_rate = config.learning_rate or float(os.getenv("LEARNING_RATE", "2e-4"))
    
    print(f"[FINETUNE-PIPELINE] Config:")
    print(f"  Model: {model_name}")
    print(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
    print(f"  Training: epochs={num_epochs}, batch={batch_size}, lr={learning_rate}")
    
    def update_progress(progress: float, step: str):
        """진행률 업데이트 헬퍼"""
        finetune_jobs[job_id].update({
            "progress": progress,
            "current_step": step
        })
    
    try:
        # ========== 1단계: 데이터 추출 ==========
        update_progress(5.0, "데이터 추출 준비 중...")
        finetune_jobs[job_id]["status"] = "extracting"
        
        # 데이터셋 경로
        dataset_path = FINETUNE_DATA_DIR / f"{job_id}_training.jsonl"
        
        # 추출 실행
        update_progress(10.0, f"데이터 추출 중 (전략: {config.extraction_strategy})...")
        dataset_size = await extract_training_data(
            doc_ids=config.doc_ids,
            output_path=dataset_path,
            strategy=config.extraction_strategy,
            total_samples=config.total_samples
        )
        
        finetune_jobs[job_id]["dataset_size"] = dataset_size
        update_progress(30.0, f"데이터 추출 완료 ({dataset_size} 샘플)")
        
        # ========== 2단계: 파인튜닝 ==========
        finetune_jobs[job_id]["status"] = "training"
        update_progress(35.0, "파인튜닝 준비 중...")
        
        # 출력 디렉토리
        output_name = config.output_name or f"{model_name.split('/')[-1]}_{config.extraction_strategy}_{job_id}"
        output_dir = FINETUNE_OUTPUT_DIR / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 진행률 콜백 (35% ~ 95%)
        def training_progress_callback(train_progress: float, step_msg: str):
            # 학습 진행률을 전체 진행률로 변환 (35% ~ 95%)
            overall_progress = 35.0 + (train_progress / 100.0) * 60.0
            update_progress(overall_progress, f"학습 중: {step_msg}")
        
        # 파인튜닝 실행
        await run_lora_training(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            progress_callback=training_progress_callback
        )
        
        # ========== 3단계: 메타데이터 저장 ==========
        update_progress(96.0, "메타데이터 저장 중...")
        
        metadata = {
            "job_id": job_id,
            "base_model": model_name,
            "extraction_strategy": config.extraction_strategy,
            "dataset_size": dataset_size,
            "doc_ids": config.doc_ids,
            "config": {
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            "created_at": finetune_jobs[job_id]["started_at"],
            "completed_at": datetime.now().isoformat()
        }
        
        metadata_path = output_dir / "finetune_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # ========== 완료 ==========
        update_progress(100.0, "파인튜닝 완료!")
        
        finetune_jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "output_path": str(output_dir)
        })
        
        print(f"[FINETUNE-PIPELINE] Job {job_id} completed successfully!")
        print(f"[FINETUNE-PIPELINE] Output: {output_dir}")
        
    except Exception as e:
        # 오류 처리
        error_msg = f"파인튜닝 실패: {str(e)}"
        print(f"[FINETUNE-PIPELINE] {error_msg}")
        
        finetune_jobs[job_id].update({
            "status": "failed",
            "error": error_msg,
            "completed_at": datetime.now().isoformat()
        })
        
        raise