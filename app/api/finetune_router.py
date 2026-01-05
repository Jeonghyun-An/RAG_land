# app/api/finetune_router.py
"""
파인튜닝 API 엔드포인트
- 선택된 문서로부터 학습 데이터 생성
- QLoRA 파인튜닝 작업 시작
- 학습 진행률 조회
- 파인튜닝된 모델 목록 조회
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import json
from pathlib import Path

router = APIRouter(prefix="/finetune", tags=["finetune"])

# 파인튜닝 작업 상태 저장소 (실제 운영에서는 Redis나 DB 사용)
finetune_jobs: Dict[str, Dict[str, Any]] = {}

# 학습 결과 저장 경로
FINETUNE_OUTPUT_DIR = Path(os.getenv("FINETUNE_OUTPUT_DIR", "/workspace/output"))
FINETUNE_DATA_DIR = Path(os.getenv("FINETUNE_DATA_DIR", "/workspace/data"))

# ==================== Request/Response Models ====================
class FinetuneStartRequest(BaseModel):
    """파인튜닝 시작 요청"""
    doc_ids: List[str]
    model_name: Optional[str] = "Qwen/Qwen2.5-14B-Instruct"
    lora_r: Optional[int] = 16
    lora_alpha: Optional[int] = 32
    num_epochs: Optional[int] = 3
    batch_size: Optional[int] = 2
    learning_rate: Optional[float] = 2e-4
    output_name: Optional[str] = None  # 파인튜닝 모델 이름 (자동 생성 가능)

class FinetuneStatusResponse(BaseModel):
    """파인튜닝 상태 응답"""
    job_id: str
    status: str  # pending, extracting, training, completed, failed
    progress: float  # 0-100
    current_step: Optional[str] = None
    doc_ids: List[str]
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
    config: Optional[Dict[str, Any]] = None

# ==================== Endpoints ====================

@router.post("/start", response_model=Dict[str, str])
async def start_finetuning(
    request: FinetuneStartRequest,
    background_tasks: BackgroundTasks
):
    """
    파인튜닝 작업 시작
    
    프로세스:
    1. 선택된 doc_ids로부터 Milvus 청크 데이터 추출
    2. QA 형식 학습 데이터셋 생성
    3. QLoRA 파인튜닝 실행 (백그라운드)
    4. 결과 모델 저장
    """
    if not request.doc_ids:
        raise HTTPException(400, "doc_ids가 비어있습니다")
    
    # Job ID 생성
    job_id = f"ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Output 이름 자동 생성
    if not request.output_name:
        request.output_name = f"nuclear-ft-{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # 작업 상태 초기화
    finetune_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "current_step": "작업 준비 중...",
        "doc_ids": request.doc_ids,
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
    
    return {
        "job_id": job_id,
        "message": f"파인튜닝 작업이 시작되었습니다 (문서 {len(request.doc_ids)}개)"
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
    
    # output 디렉토리의 각 하위 폴더를 파인튜닝 모델로 간주
    for model_dir in FINETUNE_OUTPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        
        # 메타데이터 파일 읽기
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
                    config=metadata.get("config")
                ))
            except Exception as e:
                print(f"메타데이터 읽기 실패: {model_dir.name} - {e}")
                continue
        else:
            # 메타데이터가 없으면 기본 정보만
            models.append(FinetuneModel(
                name=model_dir.name,
                path=str(model_dir),
                base_model="unknown",
                created_at=datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat()
            ))
    
    # 최신순 정렬
    models.sort(key=lambda x: x.created_at, reverse=True)
    return models

@router.delete("/job/{job_id}")
async def delete_finetuning_job(job_id: str):
    """파인튜닝 작업 삭제"""
    if job_id not in finetune_jobs:
        raise HTTPException(404, f"작업을 찾을 수 없습니다: {job_id}")
    
    job = finetune_jobs[job_id]
    
    # 실행 중인 작업은 삭제 불가
    if job["status"] in ["extracting", "training"]:
        raise HTTPException(400, "실행 중인 작업은 삭제할 수 없습니다")
    
    del finetune_jobs[job_id]
    return {"message": f"작업이 삭제되었습니다: {job_id}"}

# ==================== Background Task ====================

async def run_finetuning_pipeline(job_id: str, config: FinetuneStartRequest):
    """
    파인튜닝 파이프라인 실행 (백그라운드)
    
    단계:
    1. Milvus에서 청크 데이터 추출
    2. QA 데이터셋 생성
    3. 파인튜닝 실행
    4. 결과 저장
    """
    job = finetune_jobs[job_id]
    
    try:
        # ========== Step 1: 데이터 추출 ==========
        job["status"] = "extracting"
        job["progress"] = 10.0
        job["current_step"] = f"Milvus에서 {len(config.doc_ids)}개 문서 데이터 추출 중..."
        
        print(f"[FINETUNE] Job {job_id}: Extracting data from Milvus...")
        
        from app.services.finetune_service import extract_training_data
        
        dataset_path = FINETUNE_DATA_DIR / f"{job_id}_dataset.jsonl"
        FINETUNE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Milvus에서 데이터 추출
        dataset_size = await extract_training_data(
            doc_ids=config.doc_ids,
            output_path=dataset_path
        )
        
        job["dataset_size"] = dataset_size
        job["progress"] = 30.0
        job["current_step"] = f"학습 데이터 생성 완료 ({dataset_size}개 샘플)"
        
        print(f"[FINETUNE] Job {job_id}: Dataset created with {dataset_size} samples")
        
        # ========== Step 2: 파인튜닝 실행 ==========
        job["status"] = "training"
        job["progress"] = 40.0
        job["current_step"] = "QLoRA 파인튜닝 시작..."
        
        print(f"[FINETUNE] Job {job_id}: Starting QLoRA training...")
        
        from app.services.finetune_service import run_qlora_training
        
        output_dir = FINETUNE_OUTPUT_DIR / config.output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 학습 실행 (진행률 콜백 포함)
        def progress_callback(current_epoch: int, total_epochs: int, step: int, total_steps: int):
            """학습 진행률 업데이트 콜백"""
            epoch_progress = (current_epoch / total_epochs) * 60  # 40-100% 구간
            step_progress = (step / total_steps) * (60 / total_epochs)
            total_progress = 40 + epoch_progress + step_progress
            
            job["progress"] = min(total_progress, 95.0)
            job["current_step"] = f"Epoch {current_epoch}/{total_epochs}, Step {step}/{total_steps}"
        
        await run_qlora_training(
            model_name=config.model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            progress_callback=progress_callback
        )
        
        job["progress"] = 95.0
        job["current_step"] = "모델 저장 중..."
        
        # ========== Step 3: 메타데이터 저장 ==========
        metadata = {
            "job_id": job_id,
            "base_model": config.model_name,
            "created_at": datetime.now().isoformat(),
            "dataset_size": dataset_size,
            "doc_ids": config.doc_ids,
            "config": config.dict()
        }
        
        metadata_file = output_dir / "finetune_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # ========== Step 4: 완료 ==========
        job["status"] = "completed"
        job["progress"] = 100.0
        job["current_step"] = "파인튜닝 완료"
        job["completed_at"] = datetime.now().isoformat()
        job["output_path"] = str(output_dir)
        
        print(f"[FINETUNE] Job {job_id}: Completed successfully")
        print(f"[FINETUNE] Output: {output_dir}")
        
    except Exception as e:
        print(f"[FINETUNE] Job {job_id}: Failed - {e}")
        import traceback
        traceback.print_exc()
        
        job["status"] = "failed"
        job["current_step"] = "파인튜닝 실패"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()