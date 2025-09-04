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

def chat_complete(model_name: str, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    c = _client()
    r = c.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content

def chat_complete_on(base_url: str, model_name: str, prompt: str,
                     temperature: float = 0.2, max_tokens: int = 512) -> str:
    c = _client(base_url)
    r = c.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
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
