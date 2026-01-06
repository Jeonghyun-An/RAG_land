# app/api/finetune_router.py
"""
파인튜닝 API 엔드포인트 (완전 구현 v2)
- 기존 추출 스크립트 활용 (extract_english_first.py, extract_structured_compliance.py)
- 추출 전략 선택 가능
- L40S 최적화 LoRA 파인튜닝
- 실시간 진행률 업데이트
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import json
from pathlib import Path

router = APIRouter(prefix="/finetune", tags=["finetune"])

# 파인튜닝 작업 상태 저장소
finetune_jobs: Dict[str, Dict[str, Any]] = {}

# 학습 결과 저장 경로
FINETUNE_OUTPUT_DIR = Path(os.getenv("FINETUNE_OUTPUT_DIR", "/workspace/output"))
FINETUNE_DATA_DIR = Path(os.getenv("FINETUNE_DATA_DIR", "/workspace/data"))

# ==================== Request/Response Models ====================
class FinetuneStartRequest(BaseModel):
    """파인튜닝 시작 요청"""
    doc_ids: List[str]
    
    # 데이터 추출 전략
    extraction_strategy: Optional[str] = "balanced"  # "english_first", "structured", "balanced"
    total_samples: Optional[int] = 5000
    
    # 모델 설정
    model_name: Optional[str] = "Qwen/Qwen2.5-14B-Instruct"
    lora_r: Optional[int] = 32  # L40S: 32
    lora_alpha: Optional[int] = 64  # L40S: 64
    num_epochs: Optional[int] = 3
    batch_size: Optional[int] = 4  # L40S: 4
    learning_rate: Optional[float] = 2e-4
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
        raise HTTPException(400, f"유효하지 않은 전략: {request.extraction_strategy}. 가능한 값: {valid_strategies}")
    
    # Job ID 생성
    job_id = f"ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Output 이름 자동 생성
    if not request.output_name:
        strategy_abbr = {
            "english_first": "en",
            "structured": "st",
            "balanced": "bal"
        }[request.extraction_strategy]
        request.output_name = f"nuclear-ft-{strategy_abbr}-{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # 작업 상태 초기화
    finetune_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "current_step": "작업 준비 중...",
        "doc_ids": request.doc_ids,
        "extraction_strategy": request.extraction_strategy,
        "dataset_size": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "output_path": None,
        "config": request.dict()
    }
    
    # 백그라운드에서 파인튜닝 실행
    background_tasks.add_task(
        run_finetuning_pipeline,
        job_id,
        request
    )
    
    strategy_desc = {
        "english_first": "언어 우선순위 (70:20:10)",
        "structured": "구조화 추출 (60:40)",
        "balanced": "균형 전략 (60:40)"
    }[request.extraction_strategy]
    
    return {
        "job_id": job_id,
        "message": f"파인튜닝 작업이 시작되었습니다 (전략: {strategy_desc}, 문서 {len(request.doc_ids)}개)"
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
    3. 결과 저장 및 메타데이터 (95-100%)
    """
    job = finetune_jobs[job_id]
    
    try:
        # ========== Step 1: 데이터 추출 ==========
        job["status"] = "extracting"
        job["progress"] = 10.0
        
        strategy_desc = {
            "english_first": "언어 우선순위 추출",
            "structured": "구조화 추출",
            "balanced": "균형 전략 추출"
        }[config.extraction_strategy]
        
        job["current_step"] = f"{strategy_desc} 중... ({len(config.doc_ids)}개 문서)"
        
        print(f"[FINETUNE] Job {job_id}: Extracting data with strategy '{config.extraction_strategy}'...")
        
        from app.services.finetune_service import extract_training_data
        
        dataset_path = FINETUNE_DATA_DIR / f"{job_id}_dataset.jsonl"
        FINETUNE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # 기존 추출 스크립트 실행
        dataset_size = await extract_training_data(
            doc_ids=config.doc_ids,
            output_path=dataset_path,
            strategy=config.extraction_strategy,
            total_samples=config.total_samples
        )
        
        job["dataset_size"] = dataset_size
        job["progress"] = 30.0
        job["current_step"] = f"학습 데이터 생성 완료 ({dataset_size}개 샘플, 전략: {config.extraction_strategy})"
        
        print(f"[FINETUNE] Job {job_id}: Dataset created with {dataset_size} samples")
        
        # ========== Step 2: 파인튜닝 실행 ==========
        job["status"] = "training"
        job["progress"] = 35.0
        job["current_step"] = "L40S 최적화 LoRA 파인튜닝 시작..."
        
        print(f"[FINETUNE] Job {job_id}: Starting L40S optimized LoRA training...")
        
        from app.services.finetune_service import run_lora_training
        
        output_dir = FINETUNE_OUTPUT_DIR / config.output_name
        FINETUNE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 진행률 콜백
        def update_progress(progress_pct: float, log_line: str):
            mapped_progress = 35.0 + (progress_pct * 0.60)  # 35-95%
            job["progress"] = min(mapped_progress, 95.0)
            job["current_step"] = log_line[:200]
        
        # 학습 실행
        await run_lora_training(
            model_name=config.model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            progress_callback=update_progress
        )
        
        # ========== Step 3: 완료 처리 ==========
        job["status"] = "completed"
        job["progress"] = 100.0
        job["current_step"] = "파인튜닝 완료!"
        job["output_path"] = str(output_dir)
        job["completed_at"] = datetime.now().isoformat()
        
        print(f"[FINETUNE] Job {job_id}: Training completed successfully")
        
        # 메타데이터 저장
        metadata = {
            "base_model": config.model_name,
            "created_at": job["completed_at"],
            "dataset_size": dataset_size,
            "doc_ids": config.doc_ids,
            "extraction_strategy": config.extraction_strategy,
            "config": config.dict()
        }
        
        metadata_file = output_dir / "finetune_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[FINETUNE] Job {job_id}: Metadata saved")
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()
        job["progress"] = 0.0
        job["current_step"] = f"오류 발생: {str(e)}"
        
        print(f"[FINETUNE] Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()