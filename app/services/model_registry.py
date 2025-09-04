# app/services/model_registry.py
from __future__ import annotations
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
    "ko-llama3-8b":   ModelSpec("vllm", "saltlux/Ko-Llama3-Luxia-8B"),

    # 실제 HF ID(그대로 받아도 로컬 로딩 가능)
    "meta-llama/Llama-3.2-1B-Instruct": ModelSpec("vllm", "meta-llama/Llama-3.2-1B-Instruct"),
    "meta-llama/Llama-3.2-3B-Instruct": ModelSpec("vllm", "meta-llama/Llama-3.2-3B-Instruct"),
    "saltlux/Ko-Llama3-Luxia-8B":       ModelSpec("vllm", "saltlux/Ko-Llama3-Luxia-8B"),
}

# 데모 기본값(원하면 .env에서 DEFAULT_ALIAS 오버라이드 해도 됨)
DEFAULT_ALIAS = "llama-3.2-3b"

def resolve(model_name: Optional[str]) -> ModelSpec:
    """요청값이 별칭이든 실제 ID든 받아서 스펙으로 통일."""
    name = model_name or DEFAULT_ALIAS
    spec = REGISTRY.get(name)
    if spec:
        return spec
    # 등록 안 된 문자열은 로컬 로딩(Transformers)로 시도 — 유연 모드
    return ModelSpec("transformers", name)
