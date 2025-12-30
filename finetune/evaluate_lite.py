#!/usr/bin/env python3
"""
개선된 경량 평가 스크립트 - Tokenizer 문제 해결
finetune/evaluate_lite.py
"""

import os
import sys
import torch
import time
from datetime import datetime
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==================== 설정 ====================
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LORA_PATH = os.getenv("OUTPUT_DIR", "/workspace/output/qwen2.5-7b-nuclear-lora")
MAX_NEW_TOKENS = 200
GENERATION_TIMEOUT = 30

print("="*80)
print(" 개선된 경량 파인튜닝 모델 평가 (v2)")
print("="*80)
print(f" Base Model: {MODEL_NAME}")
print(f" LoRA Path: {LORA_PATH}")
print("="*80)

# ==================== 모델 로드 ====================
print("\n📥 Loading model...")
start_time = time.time()

try:
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    
    load_time = time.time() - start_time
    print(f" Model loaded ({load_time:.1f}s)")
    
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f" GPU Memory: {gpu_allocated:.2f} GB")

except Exception as e:
    print(f" Failed: {e}")
    sys.exit(1)

# ==================== 테스트 질문 ====================
test_questions = [
    {
        "question": "방사선작업종사자의 연간 선량한도는 얼마인가요?",
        "category": "법규_기본",
        "expected_keywords": ["50mSv", "연간", "선량한도", "밀리시버트"]
    },
    {
        "question": "원자력안전법에서 규정하는 방사선관리구역이란 무엇인가요?",
        "category": "법규_기본",
        "expected_keywords": ["방사선관리구역", "선량", "기준"]
    },
    {
        "question": "IAEA의 Defence in Depth(심층방호) 개념을 설명해주세요.",
        "category": "IAEA",
        "expected_keywords": ["defence", "depth", "심층방호", "level", "단계"]
    },
    {
        "question": "원자로 냉각재 상실사고 LOCA가 발생하면 어떻게 대응하나요?",
        "category": "기술",
        "expected_keywords": ["냉각재", "LOCA", "ECCS", "비상", "냉각"]
    },
    {
        "question": "격납건물의 주요 기능은 무엇인가요?",
        "category": "기술",
        "expected_keywords": ["격납건물", "방사성물질", "차단", "보호"]
    },
]

print(f"\n📝 Test questions: {len(test_questions)}")

# ==================== 평가 함수 ====================
def generate_response(question: str) -> tuple:
    """응답 생성 - 개선된 프롬프트 처리"""
    
    # Qwen2.5 Chat 템플릿 사용
    messages = [
        {"role": "system", "content": "당신은 원자력 안전 전문가입니다. KINAC 규정과 IAEA 가이드라인에 기반하여 정확하고 상세한 답변을 제공하세요."},
        {"role": "user", "content": question}
    ]
    
    # apply_chat_template 사용 (권장 방식)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    try:
        start = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        inference_time = time.time() - start
        
        # 입력 부분 제거하고 생성된 부분만 디코딩
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0].strip()
        
        return response, inference_time, True
    
    except Exception as e:
        return f"[ERROR: {str(e)}]", 0, False

# ==================== 평가 실행 ====================
print("\n Evaluating...\n" + "="*80)

results = []
total_time = 0
success_count = 0

for i, test_case in enumerate(test_questions, 1):
    question = test_case["question"]
    category = test_case["category"]
    expected_keywords = test_case.get("expected_keywords", [])
    
    print(f"\n[{i}/{len(test_questions)}] [{category}]")
    print(f"Q: {question}")
    
    response, inference_time, success = generate_response(question)
    
    if success:
        success_count += 1
        total_time += inference_time
        
        # 키워드 매칭
        response_lower = response.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        match_rate = matched / len(expected_keywords) if expected_keywords else 0
        
        # 응답 출력 (처음 200자)
        display_text = response[:200] + "..." if len(response) > 200 else response
        print(f"A: {display_text}")
        print(f"⏱  {inference_time:.2f}s | 키워드: {matched}/{len(expected_keywords)} ({match_rate*100:.0f}%)")
        
        result = {
            "index": i,
            "category": category,
            "question": question,
            "response": response,
            "inference_time": inference_time,
            "matched_keywords": matched,
            "total_keywords": len(expected_keywords),
            "match_rate": match_rate,
            "success": True
        }
    else:
        print(f" Failed: {response}")
        result = {
            "index": i,
            "category": category,
            "question": question,
            "response": response,
            "inference_time": 0,
            "success": False
        }
    
    results.append(result)

# ==================== 결과 분석 ====================
print("\n" + "="*80)
print(" 평가 결과")
print("="*80)

successful_results = [r for r in results if r["success"]]

if successful_results:
    avg_time = total_time / len(successful_results)
    avg_match_rate = sum(r["match_rate"] for r in successful_results) / len(successful_results)
    
    print(f"\n 성공: {success_count}/{len(test_questions)}")
    print(f"⏱  평균 추론 시간: {avg_time:.2f}s")
    print(f" 평균 키워드 매칭률: {avg_match_rate*100:.1f}%")
    
    # 카테고리별
    categories = {}
    for r in successful_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["match_rate"])
    
    print(f"\n 카테고리별 매칭률:")
    for cat, rates in categories.items():
        avg_rate = sum(rates) / len(rates)
        print(f"   [{cat}]: {avg_rate*100:.1f}%")
    
    # 판정
    print(f"\n 종합 평가:")
    if avg_match_rate >= 0.7:
        print("    우수 - 파인튜닝 성공!")
    elif avg_match_rate >= 0.5:
        print("     보통 - 추가 학습 권장")
    else:
        print("    미흡 - 재학습 필요")
else:
    print("\n 모든 테스트 실패")

# ==================== 결과 저장 ====================
output_dir = Path(LORA_PATH)
output_file = output_dir / "evaluation_lite_v2_results.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("개선된 경량 평가 결과 (v2)\n")
    f.write("="*80 + "\n\n")
    f.write(f"평가 시간: {datetime.now().isoformat()}\n")
    f.write(f"성공: {success_count}/{len(test_questions)}\n")
    if successful_results:
        f.write(f"평균 추론 시간: {avg_time:.2f}s\n")
        f.write(f"평균 키워드 매칭률: {avg_match_rate*100:.1f}%\n")
    f.write("\n" + "="*80 + "\n\n")
    
    for r in results:
        f.write(f"[{r['index']}] [{r['category']}]\n")
        f.write(f"Q: {r['question']}\n")
        if r['success']:
            f.write(f"A: {r['response']}\n")
            f.write(f"시간: {r['inference_time']:.2f}s | 매칭: {r['matched_keywords']}/{r['total_keywords']}\n")
        else:
            f.write(f" 실패: {r['response']}\n")
        f.write("\n" + "-"*80 + "\n\n")

print(f"\n 결과 저장: {output_file}")
print("\n" + "="*80)
print(" 평가 완료!")
print("="*80)