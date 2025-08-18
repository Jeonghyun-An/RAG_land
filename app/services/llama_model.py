# app/services/llama_model.py
from __future__ import annotations

import os, json, torch
from typing import Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.services.llm_client import chat_complete, list_vllm_models
from app.config import HUGGINGFACE_TOKEN

USE_VLLM = os.getenv("USE_VLLM", "1") == "1"
# 👇 하드코딩 금지: ENV에서 alias→HF ID 매핑(JSON)만 읽음
# 예) MODEL_ALIASES='{"llama-1b":"meta-llama/Llama-3.2-1B-Instruct"}'
try:
    MODEL_ALIASES: Dict[str, str] = json.loads(os.getenv("MODEL_ALIASES", "{}"))
except Exception:
    MODEL_ALIASES = {}

LOADED_MODELS: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

def _auth_kwargs() -> dict:
    return {"token": HUGGINGFACE_TOKEN} if HUGGINGFACE_TOKEN else {}

def _resolve_to_hf_id(name: str) -> Optional[str]:
    """
    name이 alias면 ENV 매핑으로 HF ID 반환.
    'org/repo' 같은 HF ID면 그대로 반환.
    둘 다 아니면 None.
    """
    if not name:
        return None
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    if "/" in name or os.path.isdir(name):
        return name
    return None  # 모르는 토큰(별칭도, HF ID도 아님)

def load_model(model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if model_id in LOADED_MODELS:
        return LOADED_MODELS[model_id]

    cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map: Optional[str | dict] = "auto" if torch.cuda.is_available() else {"": "cpu"}

    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True, cache_dir=cache_dir, **_auth_kwargs()
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch_dtype,
        device_map=device_map, low_cpu_mem_usage=True, cache_dir=cache_dir, **_auth_kwargs()
    )
    LOADED_MODELS[model_id] = (mdl, tok)
    return mdl, tok

def generate_answer(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    try:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    except Exception:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, top_k=top_k,
            eos_token_id=eos_id, pad_token_id=pad_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def generate_answer_unified(prompt: str, name_or_id: Optional[str]):
    """
    1) vLLM가 살아 있고, 요청값이 vLLM의 served name과 일치하면 vLLM로 바로 호출
    2) 아니면 요청값을 alias→HF ID 또는 '그대로 HF ID'로 해석
       - vLLM가 있더라도 served name이 다르면 폴백(Transformers)
    3) 마지막에 둘 다 못 하면 친절한 에러 메시지
    """
    name = (name_or_id or "").strip()

    # 1) vLLM served names에 있으면 그대로 vLLM 호출
    if USE_VLLM:
        served = set(list_vllm_models())  # ex: {"llama-1b", "meta-llama/Llama-3.2-1B-Instruct", ...}
        if name and name in served:
            return chat_complete(name, prompt)

    # 2) alias/HF ID 해석
    hf_id = _resolve_to_hf_id(name) or _resolve_to_hf_id(os.getenv("DEFAULT_MODEL_ALIAS", "llama-1b"))
    if hf_id:
        # vLLM로도 호출해보되, served name이 다르면 실패할 수 있으니 예외 무시하고 폴백
        if USE_VLLM and served:
            try:
                # served 이름이 HF ID와 같을 수도/다를 수도 있음. 같으면 운 좋게 바로 됨.
                if hf_id in served:
                    return chat_complete(hf_id, prompt)
            except Exception:
                pass
        # Transformers 폴백
        model, tok = load_model(hf_id)
        return generate_answer(prompt, model, tok)

    # 3) 전부 실패: 가능한 이름 제안
    suggestions = []
    if USE_VLLM and served:
        suggestions += list(sorted(served))
    if MODEL_ALIASES:
        suggestions += [f"{k} -> {v}" for k, v in MODEL_ALIASES.items()]
    hint = "; ".join(suggestions) or "환경변수 MODEL_ALIASES에 alias 매핑을 설정하세요."
    raise RuntimeError(f"모델 식별 실패: '{name_or_id}'. 사용 가능한 이름(일부): {hint}")
