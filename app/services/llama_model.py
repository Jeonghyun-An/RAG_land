# app/services/llama_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "saltlux/Ko-Llama3-Luxia-8B"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return model, tokenizer

def generate_answer(prompt: str, model, tokenizer) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
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
