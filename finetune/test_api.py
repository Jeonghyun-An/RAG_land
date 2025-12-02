#!/usr/bin/env python3
"""
vLLM API 테스트 스크립트
finetune/test_api.py

사용법:
    python finetune/test_api.py --url http://localhost:28080
"""

import argparse
import requests
import json

def test_vllm_api(base_url: str):
    """vLLM API 테스트"""
    
    print("="*80)
    print("🧪 vLLM API 테스트")
    print("="*80)
    print(f"📌 API URL: {base_url}")
    print("="*80)
    
    # 1. Health Check
    print("\n1️⃣ Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ 서버 정상")
        else:
            print(f"   ❌ 서버 응답 이상: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ 연결 실패: {e}")
        return
    
    # 2. Models 확인
    print("\n2️⃣ 사용 가능한 모델 확인")
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        models = response.json()
        print(f"   모델: {json.dumps(models, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"   ❌ 모델 조회 실패: {e}")
    
    # 3. Chat Completion 테스트
    print("\n3️⃣ Chat Completion 테스트")
    
    test_messages = [
        {
            "role": "system",
            "content": "당신은 원자력 안전 전문가입니다. KINAC 규정과 IAEA 가이드라인에 기반하여 정확하고 상세한 답변을 제공하세요."
        },
        {
            "role": "user",
            "content": "방사선작업종사자의 연간 선량한도는 얼마인가요?"
        }
    ]
    
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",  # 또는 병합 모델 경로
        "messages": test_messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            
            print("   ✅ 응답 성공")
            print(f"\n   질문: {test_messages[1]['content']}")
            print(f"\n   답변:\n   {answer}")
            print(f"\n   사용 토큰: {result.get('usage', {})}")
        else:
            print(f"   ❌ 요청 실패: {response.status_code}")
            print(f"   {response.text}")
    
    except Exception as e:
        print(f"   ❌ 테스트 실패: {e}")
    
    # 4. 원자력 도메인 특화 테스트
    print("\n4️⃣ 원자력 도메인 특화 테스트")
    
    nuclear_questions = [
        "IAEA Safety Standards의 Defence in Depth 개념을 설명해주세요.",
        "원자로 냉각재 상실사고(LOCA) 발생 시 대응 절차는?",
        "격납건물의 주요 기능은 무엇인가요?"
    ]
    
    for i, question in enumerate(nuclear_questions, 1):
        print(f"\n   [{i}] {question}")
        
        payload["messages"] = [
            test_messages[0],  # system message
            {"role": "user", "content": question}
        ]
        
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content']
                print(f"   답변: {answer[:200]}...")
            else:
                print(f"   ❌ 실패: {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ 에러: {e}")
    
    print("\n" + "="*80)
    print("✅ 테스트 완료")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM API 테스트")
    parser.add_argument("--url", required=True, help="vLLM API URL (예: http://localhost:28080)")
    
    args = parser.parse_args()
    test_vllm_api(args.url)