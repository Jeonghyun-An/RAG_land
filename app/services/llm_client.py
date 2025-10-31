# app/services/llm_client.py
from __future__ import annotations

import os
from typing import List, Optional
from openai import OpenAI

DEFAULT_BASE = os.getenv("OPENAI_BASE_URL", "http://vllm-a4000:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "not-used")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))  # 초

def _client(base_url: Optional[str] = None) -> OpenAI:
    return OpenAI(base_url=base_url or DEFAULT_BASE, api_key=API_KEY, timeout=OPENAI_TIMEOUT)

GEN_TEMP = float(os.getenv("GEN_TEMP", "0.0"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.9"))
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "320"))
GEN_REP_PEN = float(os.getenv("GEN_REP_PEN", "1.12"))  # vLLM이 무시해도 안전

def chat_complete(model_name: str, prompt: str,
                  temperature: float = GEN_TEMP,
                  max_tokens: int = GEN_MAX_TOKENS,
                  top_p: float = GEN_TOP_P,
                  stop: list[str] | None = None) -> str:
    c = _client()
    r = c.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop
    )
    return r.choices[0].message.content

def chat_complete_on(base_url: str, model_name: str, prompt: str,
                     temperature: float = GEN_TEMP,
                     max_tokens: int = GEN_MAX_TOKENS,
                     top_p: float = GEN_TOP_P,
                     stop: list[str] | None = None) -> str:
    c = _client(base_url)
    r = c.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop
    )
    return r.choices[0].message.content

def list_vllm_models(base_url: Optional[str] = None) -> List[str]:
    """해당 base_url(vLLM 서버)의 served name 목록. 실패 시 빈 리스트."""
    try:
        c = _client(base_url)
        out = c.models.list()
        return [m.id for m in getattr(out, "data", [])]
    except Exception:
        return []
    

def get_openai_client(base_url: Optional[str] = None) -> OpenAI:
    """
    vLLM(OpenAI 호환 API) 클라이언트 헬퍼
    - /ask 라우터 등에서 직접 호출하기 위해 공개 래퍼로 제공
    """
    return _client(base_url)

