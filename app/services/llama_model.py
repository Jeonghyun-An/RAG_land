# app/services/llama_model.py
from __future__ import annotations

import os, json, torch
from typing import Dict, Tuple, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.services.llm_client import chat_complete, chat_complete_on, list_vllm_models
from app.services.model_registry import resolve as resolve_spec
from app.config import HUGGINGFACE_TOKEN

# alias -> vLLM base_url 매핑(JSON). 예) {"llama-3.2-1b":"http://vllm-a4000:8000/v1"}
OPENAI_ALIAS_URLS: Dict[str, str] = json.loads(os.getenv("OPENAI_ALIAS_URLS", "{}"))
USE_VLLM = os.getenv("USE_VLLM", "1") == "1"

# 선택: HF ID -> served name 매핑(서버에서 별칭으로 다르게 띄웠을 때)
# 예) {"meta-llama/Llama-3.2-1B-Instruct":"llama-3.2-1b"}
SERVED_NAME_MAP: Dict[str, str] = json.loads(os.getenv("SERVED_NAME_MAP", "{}"))

# 선택: 기본 모델 별칭(.env에서 바꿀 수 있음)
DEFAULT_MODEL_ALIAS = os.getenv("DEFAULT_MODEL_ALIAS", "llama-3.2-3b")

LOADED_MODELS: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

def _auth_kwargs() -> dict:
    return {"token": HUGGINGFACE_TOKEN} if HUGGINGFACE_TOKEN else {}

def _hf_or_path(name: str) -> Optional[str]:
    """별칭이 아닌 순수 HF ID나 로컬 경로면 그대로 반환."""
    if not name:
        return None
    if "/" in name or os.path.isdir(name):
        return name
    return None

def _served_name_candidates(hf_id: str, alias: Optional[str]) -> List[str]:
    """vLLM served name으로 시도할 후보들 생성."""
    cands: List[str] = []
    if alias:
        cands.append(alias)  # compose에서 --served-model-name을 별칭으로 맞춘 경우
    # 명시적 매핑이 있으면 최우선
    if hf_id in SERVED_NAME_MAP:
        cands.append(SERVED_NAME_MAP[hf_id])
    # HF ID 자체로 서빙한 경우
    cands.append(hf_id)
    # 레포 베이스명으로만 서빙한 경우 (ex. meta-llama/Llama-3.2-1B-Instruct -> Llama-3.2-1B-Instruct)
    try:
        base = hf_id.split("/")[-1]
        if base not in cands:
            cands.append(base)
    except Exception:
        pass
    # 중복 제거, 순서 유지
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
    # A4000 호환성/성능 무난: float16 (BF16도 가능하지만 통일성 위해 FP16)
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
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
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
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )
    gen_ids = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def generate_answer_unified(prompt: str, name_or_id: Optional[str]):
    """
    우선순위:
      1) model_registry.resolve 로 스펙 결정을 먼저 함(별칭/ID 모두 허용)
      2) provider == 'vllm' 이고 USE_VLLM 켜져 있으면 vLLM로 시도:
         - alias별 base_url 있으면 그 서버에서 served name 후보들 중 매칭되는 이름으로 호출
         - 없으면 기본 서버에서 동일 절차
      3) 실패 시 Transformers 로컬 로딩 폴백
    """
    # 0) 스펙 해석
    spec = resolve_spec(name_or_id)
    alias = (name_or_id or "").strip() or DEFAULT_MODEL_ALIAS
    hf_id = spec.model_id if _hf_or_path(spec.model_id) else _hf_or_path(alias) or spec.model_id

    # 1) vLLM 경로 (옵션)
    if USE_VLLM and spec.provider == "vllm":
        # alias 전용 서버가 있으면 우선
        per_alias_base = OPENAI_ALIAS_URLS.get(alias)
        bases = [per_alias_base] if per_alias_base else [None]  # None => 기본 base_url
        for base in bases:
            served = set(list_vllm_models(base))
            if served:
                for candidate in _served_name_candidates(hf_id, alias):
                    if candidate in served:
                        # base가 None이면 기본 서버로, 아니면 해당 서버로 전송
                        if base:
                            return chat_complete_on(base, candidate, prompt)
                        else:
                            return chat_complete(candidate, prompt)
        # served 목록 조회 실패했거나 후보가 없으면 마지막으로 "그냥 호출"도 한 번 시도
        try:
            if per_alias_base:
                return chat_complete_on(per_alias_base, alias, prompt)
            return chat_complete(hf_id, prompt)
        except Exception:
            pass  # 폴백으로 진행

    # 2) Transformers 폴백
    model, tok = load_model(hf_id)
    return generate_answer(prompt, model, tok)
