# app/services/model_registry.py
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional

Provider = Literal["vllm", "transformers"]

@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model_id: str           # HF repo id 또는 로컬 경로
    ctx_len: int = 4096     # 기본 컨텍스트 길이

# ⚠️ vLLM 쪽은 compose에서 --served-model-name 을 별칭과 맞추면 라우팅이 가장 깔끔합니다.
REGISTRY: Dict[str, ModelSpec] = {
    # 별칭(권장)
    "llama-3.2-1b":   ModelSpec("vllm", "meta-llama/Llama-3.2-1B-Instruct"),
    "llama-3.2-3b":   ModelSpec("vllm", "meta-llama/Llama-3.2-3B-Instruct"),
    "llama-3.1-8b":   ModelSpec("vllm", "meta-llama/Llama-3.1-8B-Instruct"),
    "ko-llama3-8b":   ModelSpec("vllm", "saltlux/Ko-Llama3-Luxia-8B"),
    "qwen2.5-14b":   ModelSpec("vllm", "Qwen/Qwen2.5-14B-Instruct"),
    "qwen2.5-7b":   ModelSpec("vllm", "Qwen/Qwen2.5-7B-Instruct"),
    "qwen3-14b":   ModelSpec("vllm", "Qwen/Qwen3-14B-FP8", ctx_len=16384),
    "gemma-3-12b":   ModelSpec("vllm", "google/gemma-3-12b-it", ctx_len=26000),

    # 실제 HF ID(그대로 받아도 로컬 로딩 가능)
    "meta-llama/Llama-3.2-1B-Instruct": ModelSpec("vllm", "meta-llama/Llama-3.2-1B-Instruct"),
    "meta-llama/Llama-3.2-3B-Instruct": ModelSpec("vllm", "meta-llama/Llama-3.2-3B-Instruct"),
    "meta-llama/Llama-3.1-8B-Instruct": ModelSpec("vllm", "meta-llama/Llama-3.1-8B-Instruct"),
    "saltlux/Ko-Llama3-Luxia-8B":       ModelSpec("vllm", "saltlux/Ko-Llama3-Luxia-8B"),
    "Qwen/Qwen2.5-14B-Instruct":        ModelSpec("vllm", "Qwen/Qwen2.5-14B-Instruct", ctx_len=32768),
    "Qwen/Qwen2.5-7B-Instruct":        ModelSpec("vllm", "Qwen/Qwen2.5-7B-Instruct", ctx_len=32768),
    "Qwen/Qwen3-14B-FP8":              ModelSpec("vllm", "Qwen/Qwen3-14B-FP8", ctx_len=16384),
    "google/gemma-3-12b-it":           ModelSpec("vllm", "google/gemma-3-12b-it", ctx_len=26000),
}

# DEFAULT_MODEL_ALIAS 환경변수로 오버라이드 가능 (llama_model.py와 동일한 env var 공유)
DEFAULT_ALIAS = os.getenv("DEFAULT_MODEL_ALIAS", "qwen2.5-14b")

def _alias_url_names() -> set:
    """OPENAI_ALIAS_URLS 에 등록된 별칭(별도 vLLM 서버로 보내는 모델)."""
    try:
        data = json.loads(os.getenv("OPENAI_ALIAS_URLS", "{}") or "{}")
        return set(data) if isinstance(data, dict) else set()
    except ValueError:
        return set()


def is_registered(model_name: Optional[str]) -> bool:
    """관리자가 등록한 모델인지(REGISTRY·기본 별칭·OPENAI_ALIAS_URLS). 미지정은 기본 모델이므로 True."""
    name = (model_name or "").strip()
    if not name:
        return True
    return name in REGISTRY or name == DEFAULT_ALIAS or name in _alias_url_names()


def ask_allowed_models() -> set:
    """일반 질의(/ask)에서 지정할 수 있는 모델. ASK_MODEL_ALLOWLIST(쉼표 구분), 기본은 기본 모델만."""
    raw = os.getenv("ASK_MODEL_ALLOWLIST", "")
    names = {n.strip() for n in raw.split(",") if n.strip()}
    return names or {DEFAULT_ALIAS}


def model_request_error(model_name: Optional[str], *, admin: bool = False) -> Optional[str]:
    """요청의 model_name 이 허용되지 않으면 오류 문구, 허용되면 None.
    일반 질의는 허용목록만, 관리자 경로는 등록된 모델 전체를 허용한다(임의 모델 로딩 방지)."""
    name = (model_name or "").strip()
    if not name:
        return None
    if admin:
        return None if is_registered(name) else "등록되지 않은 모델입니다."
    return None if name in ask_allowed_models() else "허용되지 않은 모델입니다."


def resolve(model_name: Optional[str]) -> ModelSpec:
    """요청값이 별칭이든 실제 ID든 받아서 스펙으로 통일."""
    name = model_name or DEFAULT_ALIAS
    spec = REGISTRY.get(name)
    if spec:
        return spec
    # 등록 안 된 문자열은 로컬 로딩(Transformers)로 시도 — 유연 모드
    return ModelSpec("transformers", name)
