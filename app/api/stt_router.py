# app/api/stt_router.py
"""
STT 마이크로서비스 프록시 라우터
- 프론트엔드에서 /api/stt/* 요청을 stt-whisper 서비스로 전달
"""
import os
import logging
import httpx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Literal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt", tags=["STT"])

# STT 서비스 URL
STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://stt-whisper:8000")
STT_ENABLED = os.getenv("STT_ENABLED", "false").lower() == "true"
STT_TIMEOUT = float(os.getenv("STT_TIMEOUT", "30.0"))

logger.info(f"[STT Router] Service URL: {STT_SERVICE_URL}")
logger.info(f"[STT Router] Enabled: {STT_ENABLED}")


@router.get("/health")
async def stt_health():
    """STT 서비스 헬스 체크"""
    if not STT_ENABLED:
        return {"status": "disabled"}
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{STT_SERVICE_URL}/health")
            return response.json()
    except Exception as e:
        logger.error(f"[STT] Health check failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("ko"),
    task: Literal["transcribe", "translate"] = Form("transcribe"),
    use_nuclear_context: bool = Form(True),
    return_segments: bool = Form(False),
):
    """
    음성을 텍스트로 변환 (STT 서비스로 프록시)
    
    Parameters:
    -----------
    audio : UploadFile
        오디오 파일
    language : str
        언어 코드 (ko, en, auto)
    task : str
        transcribe 또는 translate
    use_nuclear_context : bool
        원자력 전문 용어 컨텍스트 사용
    return_segments : bool
        상세 세그먼트 반환 여부
    
    Returns:
    --------
    {
        "text": str,
        "language": str,
        "duration": float,
        "segments": List[dict] (선택)
    }
    """
    
    if not STT_ENABLED:
        raise HTTPException(status_code=503, detail="STT service is disabled")
    
    logger.info(f"[STT] Transcribe request: {audio.filename}, lang={language}")
    
    try:
        # 오디오 파일 읽기
        audio_content = await audio.read()
        
        # STT 서비스로 전달
        async with httpx.AsyncClient(timeout=STT_TIMEOUT) as client:
            files = {"audio": (audio.filename, audio_content, audio.content_type)}
            data = {
                "language": language,
                "task": task,
                "use_nuclear_context": str(use_nuclear_context).lower(),
                "return_segments": str(return_segments).lower(),
            }
            
            response = await client.post(
                f"{STT_SERVICE_URL}/transcribe",
                files=files,
                data=data,
            )
            
            if response.status_code != 200:
                logger.error(f"[STT] Service error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"STT service error: {response.text}"
                )
            
            result = response.json()
            logger.info(f"[STT] Success: {len(result.get('text', ''))} chars transcribed")
            
            return JSONResponse(content=result)
            
    except httpx.TimeoutException:
        logger.error("[STT] Request timeout")
        raise HTTPException(status_code=504, detail="STT service timeout")
        
    except Exception as e:
        logger.error(f"[STT] Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.get("/models")
async def list_stt_models():
    """사용 가능한 STT 모델 정보"""
    if not STT_ENABLED:
        raise HTTPException(status_code=503, detail="STT service is disabled")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{STT_SERVICE_URL}/models")
            return response.json()
    except Exception as e:
        logger.error(f"[STT] Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=str(e))