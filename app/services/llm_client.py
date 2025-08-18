# app/services/llm_client.py
import os
from openai import OpenAI

_base = os.getenv("OPENAI_BASE_URL", "http://vllm:8000/v1")
_key  = os.getenv("OPENAI_API_KEY", "not-used")

client = OpenAI(base_url=_base, api_key=_key)

def chat_complete(model_name: str, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def list_vllm_models() -> list[str]:
    """vLLM가 띄운 모델 이름들(served name)을 반환. 실패하면 빈 리스트."""
    try:
        out = client.models.list()
        return [m.id for m in out.data] if hasattr(out, "data") else []
    except Exception:
        return []

def chat_complete_on(base_url: str, model_name: str, prompt: str,
                     temperature: float = 0.2, max_tokens: int = 512) -> str:
    """alias별로 다른 vLLM 서버(base_url)로 보낼 때 사용"""
    from openai import OpenAI
    c = OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "not-used"))
    r = c.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content
