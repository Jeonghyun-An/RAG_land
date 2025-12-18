import os
import logging
import httpx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Literal
import asyncio
import json
import base64
from fastapi import WebSocket, WebSocketDisconnect

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
    
@router.websocket("/stream")
async def stt_stream(ws: WebSocket):
    """
    WebSocket STT streaming (pseudo-realtime):
    - client sends small audio chunks (base64)
    - server buffers for a few seconds and calls STT /transcribe
    - returns partial text progressively
    """
    await ws.accept()

    if not STT_ENABLED:
        await ws.send_json({"type": "error", "message": "STT service is disabled"})
        await ws.close(code=1011)
        return

    buffer = bytearray()
    last_flush = asyncio.get_event_loop().time()

    # 기본값들(클라가 첫 메시지로 세팅 가능)
    language = "ko"
    task = "transcribe"
    use_nuclear_context = True
    return_segments = False

    # flush 정책(조절 가능)
    FLUSH_SECONDS = float(os.getenv("STT_STREAM_FLUSH_SECONDS", "3.0"))  # 2~5 추천
    MAX_BUFFER_BYTES = int(os.getenv("STT_STREAM_MAX_BYTES", str(2 * 1024 * 1024)))  # 2MB

    async def flush_to_stt():
        nonlocal buffer, last_flush
        if not buffer:
            return None

        # stt-whisper가 webm/opus를 받을 수 있어야 함(현재 프론트가 webm을 보냄)
        audio_content = bytes(buffer)
        buffer = bytearray()
        last_flush = asyncio.get_event_loop().time()

        async with httpx.AsyncClient(timeout=STT_TIMEOUT) as client:
            files = {"audio": ("chunk.webm", audio_content, "audio/webm")}
            data = {
                "language": language,
                "task": task,
                "use_nuclear_context": str(use_nuclear_context).lower(),
                "return_segments": str(return_segments).lower(),
            }

            resp = await client.post(f"{STT_SERVICE_URL}/transcribe", files=files, data=data)
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, f"STT service error: {resp.text}")

            return resp.json()

    try:
        await ws.send_json({"type": "ready"})

        while True:
            msg = await ws.receive_text()
            payload = json.loads(msg)

            # 1) 설정 메시지
            if payload.get("type") == "config":
                language = payload.get("language", language)
                task = payload.get("task", task)
                use_nuclear_context = bool(payload.get("use_nuclear_context", use_nuclear_context))
                return_segments = bool(payload.get("return_segments", return_segments))
                await ws.send_json({"type": "config_ack"})
                continue

            # 2) 오디오 청크
            if payload.get("type") == "audio":
                b64 = payload.get("data")
                if not b64:
                    continue
                chunk = base64.b64decode(b64)
                buffer.extend(chunk)

                now = asyncio.get_event_loop().time()
                should_flush = (now - last_flush) >= FLUSH_SECONDS or len(buffer) >= MAX_BUFFER_BYTES

                if should_flush:
                    await ws.send_json({"type": "status", "state": "transcribing"})
                    result = await flush_to_stt()
                    text = (result or {}).get("text", "").strip()
                    if text:
                        await ws.send_json({"type": "partial", "text": text})
                    await ws.send_json({"type": "status", "state": "listening"})
                continue

            # 3) 종료 요청
            if payload.get("type") == "end":
                await ws.send_json({"type": "status", "state": "finalizing"})
                result = await flush_to_stt()
                text = (result or {}).get("text", "").strip()
                await ws.send_json({"type": "final", "text": text})
                await ws.close()
                return

    except WebSocketDisconnect:
        logger.info("[STT] WebSocket disconnected")
    except Exception as e:
        logger.error(f"[STT] WebSocket stream error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        try:
            await ws.close(code=1011)
        except Exception:
            pass
