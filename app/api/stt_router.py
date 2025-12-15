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
    """음성을 텍스트로 변환"""
    if not STT_ENABLED:
        raise HTTPException(status_code=503, detail="STT service is disabled")
    
    logger.info(f"[STT] Transcribe request: {audio.filename}, lang={language}")
    
    try:
        audio_content = await audio.read()
        
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
            logger.info(f"[STT] Success: {len(result.get('text', ''))} chars")
            
            return JSONResponse(content=result)
            
    except httpx.TimeoutException:
        logger.error("[STT] Request timeout")
        raise HTTPException(status_code=504, detail="STT request timeout")
        
    except httpx.ConnectError as e:
        logger.error(f"[STT] Connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to STT service. Check if stt-whisper container is running."
        )
        
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