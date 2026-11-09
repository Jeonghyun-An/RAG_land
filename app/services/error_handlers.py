# app/services/error_handlers.py
"""
오류 응답에서 내부 정보 숨기기 (보안대책 M13 '오류 시 내부 정보 비노출').

  - 5xx 응답의 상세 내용(예외 메시지·경로·접속 정보 등)은 서버 로그에만 남기고, 응답에는 일반 문구와
    오류 번호(error_id)만 보낸다. 로그에서 같은 번호로 원인을 찾는다.
  - 처리하지 못한 예외도 같은 방식으로 500 응답.
  - 관리자 경로(Gateway·API 인증값을 통과한 요청)와 ERROR_DETAIL_EXPOSE=1(개발용)은 상세 내용을 그대로 보낸다.
  - 4xx(입력 오류 안내 등)는 그대로 둔다.
"""
from __future__ import annotations

import logging
import os
import uuid

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.services.admin_gate import is_gated_path

logger = logging.getLogger(__name__)

GENERIC_MESSAGES = {
    500: "요청을 처리하는 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
    503: "서비스를 일시적으로 이용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
}


def _expose_detail(request: Request) -> bool:
    if os.getenv("ERROR_DETAIL_EXPOSE", "0").strip() == "1":
        return True
    return is_gated_path(request.url.path)


def _generic(status: int, request: Request, detail) -> JSONResponse:
    error_id = uuid.uuid4().hex[:12]
    logger.error("[ERROR %s] %s %s -> %s: %s", error_id, request.method, request.url.path, status, detail)
    message = GENERIC_MESSAGES.get(status, GENERIC_MESSAGES[500])
    return JSONResponse({"detail": message, "error_id": error_id}, status_code=status)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException):
        if exc.status_code >= 500 and not _expose_detail(request):
            return _generic(exc.status_code, request, exc.detail)
        return await http_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        if _expose_detail(request):
            logger.exception("[ERROR] %s %s", request.method, request.url.path)
            return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)
        error_id = uuid.uuid4().hex[:12]
        logger.exception("[ERROR %s] %s %s", error_id, request.method, request.url.path)
        return JSONResponse({"detail": GENERIC_MESSAGES[500], "error_id": error_id}, status_code=500)
