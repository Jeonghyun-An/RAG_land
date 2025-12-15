#!/usr/bin/env python3
"""
LoRA 어댑터 병합 스크립트
finetune/merge_model.py

사용법:
    docker exec -it nuclear-finetune bash
    python finetune/merge_model.py
    
설명:
    LoRA 어댑터를 베이스 모델에 병합하여 단일 모델로 만듭니다.
    vLLM에서 LoRA를 직접 지원하지 않는 경우 사용합니다.
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==================== 설정 ====================
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LORA_PATH = os.getenv("OUTPUT_DIR", "/workspace/output/qwen2.5-7b-nuclear-lora")
MERGED_OUTPUT_DIR = os.getenv("MERGED_OUTPUT_DIR", "/workspace/output/qwen2.5-7b-nuclear-merged")

print("="*80)
print("🔧 LoRA 어댑터 병합")
print("="*80)
print(f"📦 Base Model: {MODEL_NAME}")
print(f"🎯 LoRA Adapter: {LORA_PATH}")
print(f"💾 Output Directory: {MERGED_OUTPUT_DIR}")
print("="*80)

# ==================== 검증 ====================
lora_config_path = Path(LORA_PATH) / "adapter_config.json"
if not lora_config_path.exists():
    print(f"❌ LoRA adapter not found at {LORA_PATH}")
    print("   Please run train_qlora.py first!")
    sys.exit(1)

# ==================== 모델 로드 ====================
print("\n📥 Loading models...")

try:
    # Tokenizer
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_PATH,
        trust_remote_code=True
    )
    
    # Base 모델
    print(f"   Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA 어댑터 적용
    print(f"   Applying LoRA adapter: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    print(" Models loaded successfully")

except Exception as e:
    print(f" Failed to load models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 병합 ====================
print("\n🔄 Merging LoRA weights into base model...")
print("   This may take several minutes...")

try:
    # LoRA 가중치를 베이스 모델에 병합
    merged_model = model.merge_and_unload()
    
    print(" Merge completed successfully")

except Exception as e:
    print(f" Failed to merge models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 저장 ====================
print(f"\n💾 Saving merged model to {MERGED_OUTPUT_DIR}...")

try:
    # 출력 디렉토리 생성
    output_dir = Path(MERGED_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 저장
    print("   Saving model weights...")
    merged_model.save_pretrained(
        MERGED_OUTPUT_DIR,
        safe_serialization=True,  # Safetensors 형식 사용
        max_shard_size="2GB"
    )
    
    # Tokenizer 저장
    print("   Saving tokenizer...")
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    
    # 메타데이터 저장
    metadata = {
        "base_model": MODEL_NAME,
        "lora_adapter": LORA_PATH,
        "merged_at": datetime.now().isoformat(),
        "merge_method": "merge_and_unload",
        "dtype": "bfloat16"
    }
    
    import json
    metadata_file = output_dir / "merge_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(" Model saved successfully")

except Exception as e:
    print(f" Failed to save model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 검증 ====================
print("\n Verifying saved model...")

try:
    # 저장된 모델이 로드 가능한지 확인
    test_model = AutoModelForCausalLM.from_pretrained(
        MERGED_OUTPUT_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    test_tokenizer = AutoTokenizer.from_pretrained(
        MERGED_OUTPUT_DIR,
        trust_remote_code=True
    )
    
    print(" Verification successful")
    
    # 모델 크기 정보
    total_params = sum(p.numel() for p in test_model.parameters())
    print(f"\n Merged Model Info:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 2 / 1e9:.2f} GB (bfloat16)")
    
    # 간단한 테스트
    print("\n Quick test:")
    test_input = "원자력 안전의 중요성은?"
    prompt = f"""<|im_start|>system
당신은 원자력 안전 전문가입니다.<|im_end|>
<|im_start|>user
{test_input}<|im_end|>
<|im_start|>assistant
"""
    
    inputs = test_tokenizer(prompt, return_tensors="pt").to(test_model.device)
    with torch.no_grad():
        outputs = test_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
    
    response = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    
    print(f"   Q: {test_input}")
    print(f"   A: {response[:200]}...")

except Exception as e:
    print(f"⚠️  Verification failed: {e}")
    import traceback
    traceback.print_exc()

# ==================== 완료 ====================
print("\n" + "="*80)
print("병합 완료!")
print("="*80)
print(f"\n출력 경로: {MERGED_OUTPUT_DIR}")
print("\n다음 단계:")
print("1. vLLM 설정에서 MODEL_NAME을 변경:")
print(f"   MODEL_NAME={MERGED_OUTPUT_DIR}")
print("\n2. docker-compose.yml 재시작:")
print("   docker-compose restart vllm")
print("\n3. 또는 새 컨테이너로 배포:")
print("   docker-compose up -d")
print("="*80)