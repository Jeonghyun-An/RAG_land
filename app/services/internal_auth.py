# app/services/internal_auth.py
"""
홈페이지(WAS) 연계 API 내부 인증 토큰 확인 — 표준 라이브러리만 사용.

환경변수
  JAVA_SHARED_SECRET   WAS 와 공유하는 토큰(요청 헤더 X-Internal-Token)
  JAVA_TOKEN_REQUIRED  "1" 이면 토큰 필수(JAVA_SHARED_SECRET 이 비어 있어도 거부, 외부망 오픈 전 적용)
                       "0"(기본) 이면 기존 동작: 토큰을 설정한 경우에만 확인
"""
from __future__ import annotations

import hmac
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def token_required() -> bool:
    return os.getenv("JAVA_TOKEN_REQUIRED", "0").strip().lower() in {"1", "true", "yes", "y"}


def verify_internal_token(token: Optional[str], secret: Optional[str] = None) -> bool:
    """WAS 가 보낸 토큰 확인. 비교는 상수 시간으로 한다."""
    secret = os.getenv("JAVA_SHARED_SECRET", "") if secret is None else secret
    if not secret:
        if token_required():
            logger.warning("[JAVA] JAVA_TOKEN_REQUIRED=1 인데 JAVA_SHARED_SECRET 미설정 — 요청 거부")
            return False
        return True
    if not token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), secret.encode("utf-8"))
