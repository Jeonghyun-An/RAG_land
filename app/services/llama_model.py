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

def _need_hf_token(model_id: str) -> bool:
    # 메타 계열은 거의 토큰 필요
    return model_id.startswith("meta-llama/")

def _auth_kwargs() -> dict:
    return {"token": HUGGINGFACE_TOKEN} if HUGGINGFACE_TOKEN else {}

def load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    model_id = MODEL_REGISTRY[model_name]
    if _need_hf_token(model_id) and not HUGGINGFACE_TOKEN:
        raise RuntimeError(
            f"{model_id} 는 Hugging Face 토큰이 필요합니다. "
            "환경변수 HUGGINGFACE_TOKEN을 설정하세요."
        )

    auth = _auth_kwargs()

    # dtype/device
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device_map: Optional[str | dict] = "auto"
    else:
        torch_dtype = torch.float32
        device_map = {"": "cpu"}

    # 버전/네트워크 이슈 시 캐시 경로 고정하면 편함 (선택)
    cache_dir = os.environ.get("HF_HOME")  # 있으면 사용

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        cache_dir=cache_dir,
        **auth,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        cache_dir=cache_dir,
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
    # 메시지 -> 텐서
    try:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    except Exception:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,       # ← 명시적으로
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
