# app/services/llama_model.py
"""
[개선] generate_answer_unified에 파라미터 전달 지원
- max_tokens, temperature, top_p를 외부에서 지정 가능
- 기본값은 환경변수 사용
"""
from __future__ import annotations

import os, json, torch
from typing import Dict, Tuple, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.services.llm_client import chat_complete, chat_complete_on, list_vllm_models
from app.services.model_registry import resolve as resolve_spec
from app.config import HUGGINGFACE_TOKEN

OPENAI_ALIAS_URLS: Dict[str, str] = json.loads(os.getenv("OPENAI_ALIAS_URLS", "{}"))
USE_VLLM = os.getenv("USE_VLLM", "1") == "1"
SERVED_NAME_MAP: Dict[str, str] = json.loads(os.getenv("SERVED_NAME_MAP", "{}"))
DEFAULT_MODEL_ALIAS = os.getenv("DEFAULT_MODEL_ALIAS", "qwen2.5-14b")

LOADED_MODELS: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

def _auth_kwargs() -> dict:
    return {"token": HUGGINGFACE_TOKEN} if HUGGINGFACE_TOKEN else {}

def _hf_or_path(name: str) -> Optional[str]:
    if not name:
        return None
    if "/" in name or os.path.isdir(name):
        return name
    return None

def _served_name_candidates(hf_id: str, alias: Optional[str]) -> List[str]:
    cands: List[str] = []
    if alias:
        cands.append(alias)
    if hf_id in SERVED_NAME_MAP:
        cands.append(SERVED_NAME_MAP[hf_id])
    cands.append(hf_id)
    try:
        base = hf_id.split("/")[-1]
        if base not in cands:
            cands.append(base)
    except Exception:
        pass
    seen = set()
    uniq = []
    for x in cands:
        if x and x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def load_model(model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if model_id in LOADED_MODELS:
        return LOADED_MODELS[model_id]

    cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map: Optional[str | dict] = "auto" if torch.cuda.is_available() else {"": "cpu"}

    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True, cache_dir=cache_dir, **_auth_kwargs()
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        **_auth_kwargs()
    )
    LOADED_MODELS[model_id] = (mdl, tok)
    return mdl, tok

def generate_answer(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 320,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 40,
    do_sample: bool = False
) -> str:
    try:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )
    except Exception:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
            repetition_penalty=1.12,
        )
    gen_ids = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def generate_answer_unified(
    prompt: str, 
    name_or_id: Optional[str],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None
):
    """
    [개선] 파라미터를 외부에서 지정 가능
    
    Args:
        prompt: 프롬프트 텍스트
        name_or_id: 모델 이름 또는 ID
        max_tokens: 최대 토큰 수 (None이면 환경변수 사용)
        temperature: 온도 (None이면 환경변수 사용)
        top_p: top_p (None이면 환경변수 사용)
    
    Returns:
        생성된 답변 텍스트
    """
    # 파라미터 기본값 설정
    if max_tokens is None:
        max_tokens = int(os.getenv("GEN_MAX_TOKENS", "320"))
    if temperature is None:
        temperature = float(os.getenv("GEN_TEMP", "0.0"))
    if top_p is None:
        top_p = float(os.getenv("GEN_TOP_P", "0.9"))
    
    # 0) 스펙 해석
    spec = resolve_spec(name_or_id)
    alias = (name_or_id or "").strip() or DEFAULT_MODEL_ALIAS
    hf_id = spec.model_id if _hf_or_path(spec.model_id) else _hf_or_path(alias) or spec.model_id

    # 1) vLLM 경로
    if USE_VLLM and spec.provider == "vllm":
        per_alias_base = OPENAI_ALIAS_URLS.get(alias)
        bases = [per_alias_base] if per_alias_base else [None]
        for base in bases:
            served = set(list_vllm_models(base))
            if served:
                for candidate in _served_name_candidates(hf_id, alias):
                    if candidate in served:
                        if base:
                            return chat_complete_on(
                                base, candidate, prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                stop=None
                            )
                        else:
                            return chat_complete(
                                candidate, prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                stop=None
                            )
        try:
            if per_alias_base:
                return chat_complete_on(
                    per_alias_base, alias, prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=None
                )
            return chat_complete(
                hf_id, prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=None
            )
        except Exception:
            pass

    # 2) Transformers 폴백
    model, tok = load_model(hf_id)
    return generate_answer(
        prompt, model, tok,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=40,
        do_sample=(temperature > 0.0)
    )