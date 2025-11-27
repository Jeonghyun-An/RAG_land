# finetune/evaluate.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import json
from tqdm import tqdm

# 설정
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LORA_PATH = os.getenv("OUTPUT_DIR", "/workspace/output/qwen2.5-7b-nuclear-lora")
TEST_DATA = os.getenv("TEST_DATASET_PATH", "/workspace/data/test_qa.jsonl")

print("="*60)
print("🧪 Model Evaluation")
print("="*60)
print(f"📦 Base Model: {MODEL_NAME}")
print(f"🎯 LoRA Adapter: {LORA_PATH}")
print(f"📊 Test Data: {TEST_DATA}")
print("="*60)

# 모델 로드
print("📥 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

print("✅ Model loaded")

# 테스트 데이터 로드
print(f"📂 Loading test data...")
dataset = load_dataset('json', data_files=TEST_DATA)
test_data = dataset['train']

print(f"✅ Loaded {len(test_data)} test examples")

# 평가
results = []
print("\n🔍 Evaluating...")

for example in tqdm(test_data):
    instruction = example['instruction']
    expected_output = example['output']
    
    # 프롬프트 생성
    prompt = f"""<|im_start|>system
당신은 원자력 안전 전문가입니다.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
    
    # 생성
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 결과 저장
    results.append({
        "instruction": instruction,
        "expected": expected_output,
        "generated": generated,
    })

# 결과 저장
output_file = os.path.join(LORA_PATH, "evaluation_results.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("="*60)
print(f"✅ Evaluation completed")
print(f"💾 Results saved to: {output_file}")
print("="*60)

# 샘플 출력
print("\n📝 Sample Results (first 3):\n")
for i, result in enumerate(results[:3]):
    print(f"--- Example {i+1} ---")
    print(f"Q: {result['instruction']}")
    print(f"Expected: {result['expected'][:100]}...")
    print(f"Generated: {result['generated'][:100]}...")
    print()