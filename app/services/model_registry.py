# app/services/model_registry.py
from dataclasses import dataclass
from typing import Dict, Literal, Optional

Provider = Literal["vllm", "transformers"]

@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model_id: str
    ctx_len: int = 8192

# 별칭(권장) + 실제 HF ID 둘 다 허용
REGISTRY: Dict[str, ModelSpec] = {
    # 별칭
    "llama-3.2-1b": ModelSpec("vllm", "meta-llama/Llama-3.2-1B-Instruct"),
    "ko-llama3-8b": ModelSpec("vllm", "saltlux/Ko-Llama3-Luxia-8B"),

    # 실제 ID(그대로 보내도 동작)
    "meta-llama/Llama-3.2-1B-Instruct": ModelSpec("vllm", "meta-llama/Llama-3.2-1B-Instruct"),
    "saltlux/Ko-Llama3-Luxia-8B": ModelSpec("vllm", "saltlux/Ko-Llama3-Luxia-8B"),
}

DEFAULT_ALIAS = "llama-3.2-1b"

def resolve(model_name: Optional[str]) -> ModelSpec:
    """요청값이 별칭이든 실제 ID든 받아서, 스펙으로 통일해서 반환.
       등록 안 된 문자열은 'transformers' 로컬 로딩으로 시도(유연 모드)."""
    name = model_name or DEFAULT_ALIAS
    spec = REGISTRY.get(name)
    if spec:
        return spec
    # 등록 안되어 있어도 그대로 로컬 로딩 시도 (관리 강화하려면 이 분기 제거)
    return ModelSpec("transformers", name)
