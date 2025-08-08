# app/services/llama_model.py
from __future__ import annotations

import os
import torch
from typing import Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config import HUGGINGFACE_TOKEN

# 한 번 로딩한 모델을 재사용
LOADED_MODELS: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

# 사용 가능한 모델 매핑
# - RAG Q&A엔 Instruct 버전이 더 적합
MODEL_REGISTRY: Dict[str, str] = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",  # 필요 시 유지
    # "ko-llama3-luxia-8b": "saltlux/Ko-Llama3-Luxia-8B",
    # "mistral": "mistralai/Mistral-7B-Instruct-v0.1",  # 예시로 추가
}

def _get_auth_kwargs() -> dict:
    # HF 토큰이 없어도 공개 모델이면 다운로드는 가능하지만,
    # 메타 모델은 보통 토큰 필요 → 있으면 넣어줌
    return {"token": HUGGINGFACE_TOKEN} if HUGGINGFACE_TOKEN else {}

def load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    지연 로딩 + 캐시. GPU 있으면 자동으로 CUDA, 없으면 CPU.
    CPU일 땐 float32로 안전하게.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    model_id = MODEL_REGISTRY[model_name]
    auth = _get_auth_kwargs()

    # dtype/device 선택: CPU 안전(f32), CUDA면 bf16/auto 가능
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16  # 최근 메타 계열은 bf16 선호
        device_map: Optional[str | dict] = "auto"
    else:
        torch_dtype = torch.float32
        device_map = {"": "cpu"}

    # 토크나이저/모델 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        **auth,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **auth,
    )

    LOADED_MODELS[model_name] = (model, tokenizer)
    return model, tokenizer


def generate_answer(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    """
    chat_template이 있으면 그걸 쓰고, 없으면 단순 프롬프트로 생성.
    pad/eos 설정을 명시해서 끊김 문제 방지.
    """
    # 메시지 포맷 (instruct 모델은 보통 chat_template 포함)
    inputs = None
    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    except Exception:
        # chat_template이 없다면 fallback: 일반 토크나이즈
        inputs = tokenizer(prompt, return_tensors="pt").input_ids

    # 디바이스 맞추기
    device = next(model.parameters()).device
    if isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = inputs.to(device)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
