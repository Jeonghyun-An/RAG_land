# app/services/admin_gate.py
"""
관리자 전용 경로 보호 (Gateway 가 부여하는 인증값 확인) — 표준 라이브러리만 사용.

배경
  관리·개발·파인튜닝·디버그·API 문서 경로는 관리자 PC 에서만 접근해야 한다.
  Gateway(Nginx)는 관리자 IP 요청에만 비밀 헤더(X-Admin-Gate)를 붙이고, 그 외 요청에서는
  클라이언트가 보낸 같은 이름의 헤더를 제거한다. 이 모듈은 FastAPI 쪽에서 그 값을 한 번 더
  확인해, Gateway 를 거치지 않는 호출(같은 ragnet 의 다른 컨테이너 등)과 Gateway 설정 오류를
  막는다.

환경변수
  ADMIN_GATE_SECRET   Gateway 의 admin_gate*.map 에 넣은 값과 같은 비밀값(예: openssl rand -hex 32)
  ADMIN_GATE_ENFORCE  "1"(기본) 적용 / "0" 경고 로그만 남기고 통과(긴급 복구용)

비밀값이 비어 있으면 보호 대상 경로는 모두 503 으로 거부한다(미설정 상태로 열리지 않도록).
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import posixpath
import re
from typing import Tuple

logger = logging.getLogger(__name__)

API_BASE = "/llama"
GATED_NAMES = (
    "finetune", "admin", "dev", "debug", "milvus", "db",
    "docs", "openapi.json", "generate", "test", "files/purge",
)
GATED_PREFIXES = tuple(f"{API_BASE}/{n}" for n in GATED_NAMES)
_HEADER = b"x-admin-gate"


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _matches(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in GATED_PREFIXES)


def is_gated_path(path: str) -> bool:
    """보호 대상 경로 여부. 중복 슬래시·'..' 로 우회하지 못하도록 정리한 경로도 함께 검사한다."""
    path = path or ""
    collapsed = re.sub(r"/+", "/", path)
    return _matches(path) or _matches(collapsed) or _matches(posixpath.normpath(collapsed))


def evaluate(header_value: str) -> Tuple[bool, int, str]:
    """(통과 여부, 거부 시 상태코드, 메시지)."""
    enforce = _flag("ADMIN_GATE_ENFORCE", "1")
    secret = os.getenv("ADMIN_GATE_SECRET", "")

    if not secret:
        if enforce:
            return False, 503, "admin gate is not configured"
        logger.warning("[ADMIN_GATE] ADMIN_GATE_SECRET 미설정 — ADMIN_GATE_ENFORCE=0 이라 통과")
        return True, 200, ""

    if header_value and hmac.compare_digest(header_value.encode("utf-8"), secret.encode("utf-8")):
        return True, 200, ""

    if enforce:
        return False, 403, "forbidden"
    logger.warning("[ADMIN_GATE] 인증값 불일치 — ADMIN_GATE_ENFORCE=0 이라 통과")
    return True, 200, ""


class AdminGateMiddleware:
    """순수 ASGI 미들웨어(SSE·백그라운드 작업에 영향이 없도록 BaseHTTPMiddleware 를 쓰지 않음)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and is_gated_path(scope.get("path", "")):
            value = b""
            for key, val in scope.get("headers", []):
                if key == _HEADER:
                    value = val
                    break
            ok, status, message = evaluate(value.decode("latin-1"))
            if not ok:
                logger.warning("[ADMIN_GATE] 거부 %s %s -> %s", scope.get("method", ""), scope.get("path", ""), status)
                body = json.dumps({"detail": message}).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode("ascii"))],
                })
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)
