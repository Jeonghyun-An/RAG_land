#!/usr/bin/env python3
"""
파인튜닝 모델 평가 스크립트
finetune/evaluate.py

사용법:
    docker exec -it nuclear-finetune bash
    python finetune/evaluate.py
"""

import os
import sys
import torch
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# ==================== 설정 ====================
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LORA_PATH = os.getenv("OUTPUT_DIR", "/workspace/output/qwen2.5-7b-nuclear-lora")
TEST_DATA = os.getenv("TEST_DATASET_PATH", "/workspace/data/test_qa.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print(" 파인튜닝 모델 평가")
print("="*80)
print(f" Base Model: {MODEL_NAME}")
print(f" LoRA Adapter: {LORA_PATH}")
print(f" Test Data: {TEST_DATA}")
print(f" Device: {DEVICE}")
print("="*80)

# ==================== 모델 로드 ====================
print("\n Loading model...")

try:
    # LoRA 어댑터가 있는지 확인
    lora_config_path = Path(LORA_PATH) / "adapter_config.json"
    if not lora_config_path.exists():
        print(f" LoRA adapter not found at {LORA_PATH}")
        print("   Please run train_qlora.py first!")
        sys.exit(1)
    
    # Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_PATH,
        trust_remote_code=True
    )
    
    # Base 모델 로드
    print(f"   Loading base model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA 어댑터 적용
    print(f"   Applying LoRA adapter: {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    
    print(" Model loaded successfully")
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Model Info:")
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Trainable ratio: {100 * trainable_params / total_params:.2f}%")

except Exception as e:
    print(f" Failed to load model: {e}")
    traceback.print_exc()
    sys.exit(1)

# ==================== 테스트 데이터 로드 ====================
print(f"\n Loading test data: {TEST_DATA}")

try:
    if not Path(TEST_DATA).exists():
        print(f" Test data not found: {TEST_DATA}")
        print("   Please run prepare_from_milvus.py first!")
        sys.exit(1)
    
    dataset = load_dataset('json', data_files=TEST_DATA)
    test_data = dataset['train']
    
    print(f" Loaded {len(test_data)} test examples")

except Exception as e:
    print(f" Failed to load test data: {e}")
    traceback.print_exc()
    sys.exit(1)

# ==================== 평가 함수 ====================
def generate_response(instruction: str, input_text: str = "") -> str:
    """모델 응답 생성"""
    
    # 프롬프트 구성
    if input_text:
        prompt = f"""<|im_start|>system
당신은 원자력 안전 전문가입니다. KINAC 규정과 IAEA 가이드라인에 기반하여 정확하고 상세한 답변을 제공하세요.<|im_end|>
<|im_start|>user
{instruction}

추가 정보: {input_text}<|im_end|>
<|im_start|>assistant
"""
    else:
        prompt = f"""<|im_start|>system
당신은 원자력 안전 전문가입니다. KINAC 규정과 IAEA 가이드라인에 기반하여 정확하고 상세한 답변을 제공하세요.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1].strip()
    else:
        response = generated_text
    
    return response

# ==================== 평가 실행 ====================
print("\n Evaluating model...")
print("="*80)

results = []
total_time = 0

for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
    instruction = example['instruction']
    input_text = example.get('input', '')
    expected_output = example['output']
    
    # 시간 측정
    start_time = datetime.now()
    
    try:
        generated_output = generate_response(instruction, input_text)
        
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds()
        total_time += inference_time
        
        result = {
            "index": i,
            "instruction": instruction,
            "input": input_text,
            "expected": expected_output,
            "generated": generated_output,
            "inference_time": inference_time,
            "success": True
        }
        
    except Exception as e:
        result = {
            "index": i,
            "instruction": instruction,
            "input": input_text,
            "expected": expected_output,
            "generated": f"[ERROR] {str(e)}",
            "inference_time": 0,
            "success": False
        }
    
    results.append(result)

# ==================== 결과 분석 ====================
print("\n" + "="*80)
print(" 평가 결과")
print("="*80)

successful_count = sum(1 for r in results if r['success'])
failed_count = len(results) - successful_count
avg_time = total_time / len(results) if results else 0

print(f"\n 통계:")
print(f"   총 테스트: {len(results)}")
print(f"   성공: {successful_count}")
print(f"   실패: {failed_count}")
print(f"   평균 추론 시간: {avg_time:.2f}초")
print(f"   총 소요 시간: {total_time:.2f}초")

# ==================== 결과 저장 ====================
output_dir = Path(LORA_PATH)
output_dir.mkdir(parents=True, exist_ok=True)

# 전체 결과 저장
results_file = output_dir / "evaluation_results.json"
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump({
        "metadata": {
            "model_name": MODEL_NAME,
            "lora_path": LORA_PATH,
            "test_data": TEST_DATA,
            "total_examples": len(results),
            "successful": successful_count,
            "failed": failed_count,
            "avg_inference_time": avg_time,
            "total_time": total_time,
            "evaluated_at": datetime.now().isoformat()
        },
        "results": results
    }, f, ensure_ascii=False, indent=2)

print(f"\n 결과 저장:")
print(f"   {results_file}")

# 샘플 결과 저장 (텍스트 파일)
samples_file = output_dir / "evaluation_samples.txt"
with open(samples_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("파인튜닝 모델 평가 샘플 결과\n")
    f.write("="*80 + "\n\n")
    
    for i, result in enumerate(results[:10]):  # 첫 10개만
        f.write(f"[Sample {i+1}]\n")
        f.write("-"*80 + "\n")
        f.write(f"질문: {result['instruction']}\n")
        if result['input']:
            f.write(f"입력: {result['input']}\n")
        f.write(f"\n기대 응답:\n{result['expected']}\n")
        f.write(f"\n생성 응답:\n{result['generated']}\n")
        f.write(f"\n추론 시간: {result['inference_time']:.2f}초\n")
        f.write("="*80 + "\n\n")

print(f"   {samples_file}")

# ==================== 샘플 출력 ====================
print("\n 샘플 결과 (처음 3개):\n")
print("="*80)

for i, result in enumerate(results[:3]):
    print(f"\n[Sample {i+1}]")
    print("-"*80)
    print(f"질문: {result['instruction']}")
    if result['input']:
        print(f"입력: {result['input']}")
    print(f"\n기대 응답:\n{result['expected'][:200]}...")
    print(f"\n생성 응답:\n{result['generated'][:200]}...")
    print(f"\n추론 시간: {result['inference_time']:.2f}초")
    print("="*80)

print("\n 평가 완료!")
print(f" 상세 결과: {output_dir}")