# stt-service/app.py
"""
Faster-Whisper 기반 STT 마이크로서비스
원자력 분야 전문 용어 최적화 포함
"""
import os
import tempfile
import logging
from pathlib import Path
from typing import Literal, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# ==================== 로깅 설정 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 환경 변수 ====================
MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "medium")  # tiny, base, small, medium, large-v3
DEVICE = os.getenv("STT_DEVICE", "cuda")  # cuda, cpu
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")  # int8, float16, float32
DOWNLOAD_ROOT = os.getenv("HF_HOME", "/models")
BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "5"))
VAD_FILTER = os.getenv("STT_VAD_FILTER", "true").lower() == "true"

# ==================== 원자력 전문 용어 사전 ====================
NUCLEAR_GLOSSARY = [
    # 국제기구
    "IAEA", "International Atomic Energy Agency",
    "KINS", "Korea Institute of Nuclear Safety",
    "KINAC", "Korea Institute of Nuclear Nonproliferation And Control",
    
    # 핵심 개념
    "핵물질", "nuclear material", "safeguards", "안전조치",
    "PIV", "Physical Inventory Verification", "물리적 재고검증",
    "PIT", "Physical Inventory Taking", "실물재고조사",
    "MBA", "Material Balance Area", "물질수지구역",
    "KMP", "Key Measurement Point", "주요측정점",
    
    # 절차 및 보고
    "재고변동보고", "inventory change report", "ICR",
    "물질수지보고", "material balance report", "MBR",
    "시설부속서", "facility attachment",
    "설계정보질문서", "design information questionnaire", "DIQ",
    
    # 시설 및 장비
    "격납건물", "containment building",
    "원자로", "reactor", "노심", "core",
    "사용후연료", "spent fuel", "신연료", "fresh fuel",
    "CANDU", "중수로", "경수로", "PWR", "pressurized water reactor",
    
    # 측정 및 분석
    "선량한도", "dose limit", "밀리시버트", "millisievert", "mSv",
    "방사선작업종사자", "radiation worker",
    "방사선관리구역", "radiation controlled area",
    "오염도", "contamination level",
    
    # 안전 및 사고
    "LOCA", "loss of coolant accident", "냉각재상실사고",
    "ECCS", "emergency core cooling system", "비상노심냉각계통",
    "심층방호", "defence in depth", "defense in depth",
    "중대사고", "severe accident",
    
    # 법규
    "원자력안전법", "Nuclear Safety Act",
    "원자력시설등의방호및방사능방재대책법",
    "제57조", "article 57", "제58조", "article 58",
]

NUCLEAR_PROMPT = """원자력 안전, KINAC 규정, IAEA 안전조치, 핵물질 재고 관리, 
방사선 안전, 원자로 운영, 사용후연료 관리"""

# ==================== 전역 모델 ====================
_MODEL = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global _MODEL
    
    logger.info("="*60)
    logger.info("STT Service Starting")
    logger.info("="*60)
    logger.info(f"Model Size: {MODEL_SIZE}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Compute Type: {COMPUTE_TYPE}")
    logger.info(f"Download Root: {DOWNLOAD_ROOT}")
    logger.info(f"Beam Size: {BEAM_SIZE}")
    logger.info(f"VAD Filter: {VAD_FILTER}")
    logger.info("="*60)
    
    try:
        logger.info(f"📥 Loading Faster-Whisper {MODEL_SIZE}...")
        _MODEL = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=DOWNLOAD_ROOT,
            num_workers=4,  # 병렬 처리
        )
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    logger.info("STT Service Shutting Down")

# ==================== FastAPI 앱 ====================
app = FastAPI(
    title="Nuclear STT Service",
    description="Faster-Whisper 기반 원자력 분야 특화 음성인식 서비스",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 엔드포인트 ====================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Nuclear STT Service",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "status": "ready" if _MODEL else "loading"
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": DEVICE
    }


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("ko"),  # ko, en, auto
    task: Literal["transcribe", "translate"] = Form("transcribe"),
    use_nuclear_context: bool = Form(True),  # 원자력 컨텍스트 사용 여부
    return_segments: bool = Form(False),  # 상세 세그먼트 반환 여부
):
    """
    음성을 텍스트로 변환
    
    Parameters:
    -----------
    audio : UploadFile
        오디오 파일 (webm, mp3, wav, m4a, ogg 등)
    language : str
        언어 코드 (ko=한국어, en=영어, auto=자동감지)
    task : str
        transcribe (전사) 또는 translate (영어로 번역)
    use_nuclear_context : bool
        원자력 전문 용어 컨텍스트 사용 여부
    return_segments : bool
        타임스탬프 포함 상세 세그먼트 반환 여부
    
    Returns:
    --------
    {
        "text": str,              # 전체 텍스트
        "language": str,          # 감지된 언어
        "duration": float,        # 오디오 길이 (초)
        "segments": List[dict]    # (선택) 세그먼트 상세 정보
    }
    """
    
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    # Content-Type 검증
    if not audio.content_type:
        raise HTTPException(status_code=400, detail="Missing content type")
    
    # 파일 확장자 검증
    valid_extensions = {".webm", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
    file_ext = Path(audio.filename or "").suffix.lower()
    
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file_ext}. Supported: {valid_extensions}"
        )
    
    logger.info(f"Received audio: {audio.filename} ({audio.content_type})")
    
    tmp_path = None
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"Saved to temp: {tmp_path}")
        
        # Faster-Whisper 변환 파라미터 설정
        transcribe_kwargs = {
            "language": None if language == "auto" else language,
            "task": task,
            "beam_size": BEAM_SIZE,
            "vad_filter": VAD_FILTER,
        }
        
        # VAD 파라미터 (묵음 제거)
        if VAD_FILTER:
            transcribe_kwargs["vad_parameters"] = dict(
                min_silence_duration_ms=500,  # 0.5초 이상 묵음만 제거
                speech_pad_ms=400,  # 음성 앞뒤 패딩
            )
        
        # 원자력 컨텍스트 추가
        if use_nuclear_context:
            transcribe_kwargs["initial_prompt"] = NUCLEAR_PROMPT
            transcribe_kwargs["hotwords"] = " ".join(NUCLEAR_GLOSSARY)
        
        logger.info(f"Transcribing with params: {transcribe_kwargs}")
        
        # 변환 실행
        segments, info = _MODEL.transcribe(tmp_path, **transcribe_kwargs)
        
        # 결과 수집
        full_text_parts = []
        segment_list = []
        
        for seg in segments:
            text_clean = seg.text.strip()
            full_text_parts.append(text_clean)
            
            if return_segments:
                segment_list.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text_clean,
                })
        
        result_text = " ".join(full_text_parts).strip()
        
        logger.info(f"Transcription complete: {len(segment_list)} segments, "
                   f"language={info.language}, duration={info.duration:.1f}s")
        logger.info(f"Result preview: {result_text[:100]}...")
        
        response_data = {
            "text": result_text,
            "language": info.language,
            "duration": round(info.duration, 2),
        }
        
        if return_segments:
            response_data["segments"] = segment_list
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        
    finally:
        # 임시 파일 정리
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info(f"Cleaned up temp file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean temp file: {e}")


@app.post("/batch-transcribe")
async def batch_transcribe(
    audios: List[UploadFile] = File(...),
    language: str = Form("ko"),
    use_nuclear_context: bool = Form(True),
):
    """
    여러 오디오 파일을 일괄 변환
    
    Parameters:
    -----------
    audios : List[UploadFile]
        오디오 파일 목록
    language : str
        언어 코드
    use_nuclear_context : bool
        원자력 컨텍스트 사용 여부
    
    Returns:
    --------
    List[dict]: 각 파일의 변환 결과
    """
    
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if len(audios) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for idx, audio in enumerate(audios):
        logger.info(f"Processing batch {idx+1}/{len(audios)}: {audio.filename}")
        
        try:
            result = await transcribe_audio(
                audio=audio,
                language=language,
                task="transcribe",
                use_nuclear_context=use_nuclear_context,
                return_segments=False
            )
            
            results.append({
                "filename": audio.filename,
                "status": "success",
                "data": result
            })
            
        except Exception as e:
            logger.error(f"Failed to process {audio.filename}: {e}")
            results.append({
                "filename": audio.filename,
                "status": "error",
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})


@app.get("/models")
async def list_models():
    """사용 가능한 모델 정보"""
    return {
        "current_model": MODEL_SIZE,
        "available_models": ["tiny", "base", "small", "medium", "large-v3"],
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )