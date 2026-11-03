# app/services/safe_io.py
"""
요청으로 전달되는 파일 경로·콜백 URL 검증 유틸 (표준 라이브러리만 사용).

- 경로: 처리 대상 파일은 공유 볼륨(SERVER_BASE_PATH 등) 하위여야 한다.
        절대경로·'..'·심볼릭 링크로 볼륨 밖을 가리키는 요청은 거부한다.
- 콜백: 완료 통보(webhook)는 허용된 내부 주소로만 전송한다.

환경변수
  SHARED_PATH_ENFORCE         "1"(기본) 거부 / "0" 경고 로그만 남기고 허용(긴급 복구용)
  SHARED_ALLOWED_ROOTS        허용 루트 추가(쉼표 구분). 기본: SERVER_BASE_PATH, /mnt/shared
  SHARED_PATH_STRICT_REALPATH "1" 심볼릭 링크를 따라간 실제 경로까지 검사 / "0"(기본) 경고만
  WEBHOOK_ENFORCE             "1"(기본) 거부 / "0" 경고 로그만 남기고 허용(긴급 복구용)
  WEBHOOK_ALLOWED_HOSTS       허용 호스트(쉼표 구분, "host" 또는 "host:port").
                              지정하면 이 목록만 허용. 미지정 시 내부 주소만 허용
                              (localhost, host.docker.internal, 사설·루프백 IP)
"""
from __future__ import annotations

import ipaddress
import logging
import os
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_SHARED_ROOT = "/mnt/shared"


class UnsafePathError(ValueError):
    """허용 루트 밖을 가리키는 경로."""


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _norm(p) -> str:
    # abspath 는 '..' 를 문자열 기준으로 정리한다(심볼릭 링크는 따라가지 않음)
    return os.path.normpath(os.path.abspath(str(p)))


def _is_under(candidate, root) -> bool:
    c, r = _norm(candidate), _norm(root)
    return c == r or c.startswith(r.rstrip(os.sep) + os.sep)


def shared_roots(base_path: str) -> List[str]:
    """허용 루트 목록: SERVER_BASE_PATH + 기본 공유 볼륨 + SHARED_ALLOWED_ROOTS."""
    extra = os.getenv("SHARED_ALLOWED_ROOTS", "").split(",")
    roots: List[str] = []
    for r in [base_path, DEFAULT_SHARED_ROOT, *extra]:
        r = (r or "").strip()
        if r and r not in roots:
            roots.append(r)
    return roots


def ensure_under_roots(candidate, roots: Iterable[str]) -> None:
    """candidate 가 roots 중 하나의 하위가 아니면 UnsafePathError."""
    roots = list(roots)
    enforce = _flag("SHARED_PATH_ENFORCE", "1")

    if "\x00" in str(candidate):
        raise UnsafePathError("NUL 문자가 포함된 경로")

    if not any(_is_under(candidate, r) for r in roots):
        msg = f"허용 경로 밖: {candidate} (허용: {', '.join(roots)})"
        if enforce:
            raise UnsafePathError(msg)
        logger.warning("[SAFE_IO] %s — SHARED_PATH_ENFORCE=0 이라 허용", msg)
        return

    # 심볼릭 링크로 허용 루트를 벗어나는지 검사(공유 볼륨 내 링크 구조를 고려해 기본은 경고만)
    real_c = os.path.realpath(str(candidate))
    if not any(_is_under(real_c, os.path.realpath(r)) for r in roots):
        msg = f"심볼릭 링크가 허용 경로 밖을 가리킴: {candidate} -> {real_c}"
        if enforce and _flag("SHARED_PATH_STRICT_REALPATH", "0"):
            raise UnsafePathError(msg)
        logger.warning("[SAFE_IO] %s", msg)


def resolve_source_path(path: str, file_id: str, base_path: str) -> Path:
    """
    convert-and-index 의 기존 경로 조합 규칙을 그대로 유지하고,
    결과 경로가 공유 볼륨 하위인지만 추가로 검증한다.
    """
    raw_path = Path(path)
    base = raw_path if raw_path.is_absolute() else Path(base_path) / raw_path
    # base가 폴더이거나 확장자가 없으면 file_id를 붙여 실제 파일 경로 구성
    full_path = base if base.suffix else (base / file_id)
    ensure_under_roots(full_path, shared_roots(base_path))
    return full_path


# ==================== 콜백 URL ====================
def webhook_url_permitted(url: str) -> bool:
    """콜백 전송 가능 여부. 거부 시 호스트만 경고 로그에 남긴다(쿼리·토큰 비노출)."""
    allowed, host = _webhook_url_allowed(url)
    if allowed:
        return True
    if _flag("WEBHOOK_ENFORCE", "1"):
        logger.warning("[WEBHOOK] 차단: 허용되지 않은 콜백 호스트 '%s'", host or "(해석 불가)")
        return False
    logger.warning("[WEBHOOK] 허용되지 않은 콜백 호스트 '%s' — WEBHOOK_ENFORCE=0 이라 전송", host or "(해석 불가)")
    return True


def _webhook_url_allowed(url: str):
    """(허용 여부, 호스트) 반환."""
    if not url:
        return False, ""
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        port = p.port
    except ValueError:
        return False, ""
    if p.scheme not in ("http", "https") or not host:
        return False, host
    if p.username or p.password:  # user:pass@host 형태 우회 방지
        return False, host

    allow = [h.strip().lower() for h in os.getenv("WEBHOOK_ALLOWED_HOSTS", "").split(",") if h.strip()]
    if allow:
        hostport = f"{host}:{port}" if port else host
        return (host in allow or hostport in allow), host

    if host in ("localhost", "host.docker.internal"):
        return True, host
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False, host  # 이름은 WEBHOOK_ALLOWED_HOSTS 에 등록해야 허용
    # 링크로컬(169.254.x.x 등)·미지정(0.0.0.0) 주소는 콜백 대상이 될 이유가 없어 제외
    return (ip.is_loopback or (ip.is_private and not ip.is_link_local and not ip.is_unspecified)), host
