# app/services/llama_model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import HUGGINGFACE_TOKEN

# 모델 캐시 저장소 (한 번 로딩한 모델 재사용)
LOADED_MODELS = {}

# 모델 경로 매핑 (프론트에서 선택할 수 있게)
MODEL_REGISTRY = {
    "ko-llama3-luxia-8b": "saltlux/Ko-Llama3-Luxia-8B",
    # "mistral": "mistralai/Mistral-7B-Instruct-v0.1",  ← 예시로 추가 가능
    # "llama2": "meta-llama/Llama-2-7b-chat-hf",
}

def load_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
    
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    model_path = MODEL_REGISTRY[model_name]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=HUGGINGFACE_TOKEN
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=HUGGINGFACE_TOKEN
    )

    LOADED_MODELS[model_name] = (model, tokenizer)
    return model, tokenizer


def generate_answer(prompt: str, model, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
