#!/usr/bin/env python3
"""
베이스 모델 vs 파인튜닝 모델 성능 비교
finetune/compare_models.py

사용법:
    python finetune/compare_models.py --base-url http://192.168.12.72:18080 --finetuned-url http://192.168.12.72:28080
    
설명:
    운영 중인 두 vLLM 서버(베이스/파인튜닝)의 응답을 비교 평가합니다.
"""

import argparse
import requests
import json
from typing import List, Dict
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ==================== 테스트 질문 세트 ====================
NUCLEAR_TEST_QUESTIONS = [
    # 기본 규정
    {
        "category": "법규_기본",
        "question": "방사선작업종사자의 연간 선량한도는 얼마인가요?",
        "expected_keywords": ["50mSv", "연간", "선량한도"]
    },
    {
        "category": "법규_기본",
        "question": "원자력안전법에서 규정하는 방사선관리구역의 정의는?",
        "expected_keywords": ["방사선관리구역", "선량", "구역"]
    },
    
    # IAEA 가이드라인
    {
        "category": "IAEA",
        "question": "IAEA Safety Standards에서 Defence in Depth의 5가지 레벨은 무엇인가요?",
        "expected_keywords": ["defence", "depth", "level", "5"]
    },
    {
        "category": "IAEA",
        "question": "IAEA의 Safety Culture 정의는?",
        "expected_keywords": ["safety culture", "commitment", "protection"]
    },
    
    # 기술적 질문
    {
        "category": "기술",
        "question": "원자로 냉각재 상실사고(LOCA) 시 대응 절차는?",
        "expected_keywords": ["냉각재", "LOCA", "비상노심냉각계통", "ECCS"]
    },
    {
        "category": "기술",
        "question": "격납건물 설계압력 산정 시 고려사항은?",
        "expected_keywords": ["격납건물", "설계압력", "사고"]
    },
    
    # 절차/매뉴얼
    {
        "category": "절차",
        "question": "방사선 비상시 주민보호조치 절차는?",
        "expected_keywords": ["비상", "주민", "보호조치", "대피"]
    },
    {
        "category": "절차",
        "question": "방사성폐기물 처리 절차에 대해 설명해주세요.",
        "expected_keywords": ["폐기물", "처리", "저장", "처분"]
    },
    
    # 복합 질문
    {
        "category": "복합",
        "question": "중대사고 발생 시 격납건물 건전성 유지를 위한 설계 특징과 운전 절차를 설명해주세요.",
        "expected_keywords": ["중대사고", "격납건물", "건전성", "설계", "절차"]
    },
    {
        "category": "복합",
        "question": "원자력발전소의 심층방호 개념이 실제 안전계통 설계에 어떻게 반영되는지 설명해주세요.",
        "expected_keywords": ["심층방호", "defence in depth", "안전계통", "다중성", "독립성"]
    }
]

# ==================== API 호출 함수 ====================
def query_rag_api(base_url: str, query: str, doc_filter: List[str] = None) -> Dict:
    """RAG API 호출"""
    
    endpoint = f"{base_url}/java/chat"
    
    payload = {
        "query": query,
        "top_k": 5,
        "use_rerank": True
    }
    
    if doc_filter:
        payload["doc_filter"] = doc_filter
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        return {
            "error": str(e),
            "answer": "[ERROR] API 호출 실패",
            "chunks": []
        }

# ==================== 평가 함수 ====================
def evaluate_response(response: str, expected_keywords: List[str]) -> Dict:
    """응답 품질 평가"""
    
    response_lower = response.lower()
    
    # 키워드 매칭
    matched_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
    keyword_score = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0
    
    # 응답 길이 (너무 짧거나 길면 감점)
    length = len(response)
    if length < 50:
        length_score = 0.5
    elif length > 2000:
        length_score = 0.8
    else:
        length_score = 1.0
    
    # 종합 점수
    total_score = (keyword_score * 0.7 + length_score * 0.3)
    
    return {
        "keyword_score": keyword_score,
        "length_score": length_score,
        "total_score": total_score,
        "matched_keywords": matched_keywords,
        "response_length": length
    }

# ==================== 비교 실행 ====================
def compare_models(base_url: str, finetuned_url: str, output_dir: str = "./comparison_results"):
    """두 모델 성능 비교"""
    
    print("="*80)
    print(" 모델 성능 비교 테스트")
    print("="*80)
    print(f" Base Model URL: {base_url}")
    print(f" Finetuned Model URL: {finetuned_url}")
    print(f" Test Questions: {len(NUCLEAR_TEST_QUESTIONS)}")
    print("="*80)
    
    results = []
    
    for i, test_case in enumerate(tqdm(NUCLEAR_TEST_QUESTIONS, desc="Testing")):
        question = test_case["question"]
        category = test_case["category"]
        expected_keywords = test_case["expected_keywords"]
        
        # Base 모델 쿼리
        base_response = query_rag_api(base_url, question)
        base_answer = base_response.get("answer", "[ERROR]")
        
        # Finetuned 모델 쿼리
        finetuned_response = query_rag_api(finetuned_url, question)
        finetuned_answer = finetuned_response.get("answer", "[ERROR]")
        
        # 평가
        base_eval = evaluate_response(base_answer, expected_keywords)
        finetuned_eval = evaluate_response(finetuned_answer, expected_keywords)
        
        # 결과 저장
        result = {
            "index": i,
            "category": category,
            "question": question,
            "expected_keywords": expected_keywords,
            "base_model": {
                "answer": base_answer,
                "evaluation": base_eval,
                "chunks_count": len(base_response.get("chunks", []))
            },
            "finetuned_model": {
                "answer": finetuned_answer,
                "evaluation": finetuned_eval,
                "chunks_count": len(finetuned_response.get("chunks", []))
            },
            "winner": "finetuned" if finetuned_eval["total_score"] > base_eval["total_score"] else "base" if base_eval["total_score"] > finetuned_eval["total_score"] else "tie"
        }
        
        results.append(result)
    
    # ==================== 통계 분석 ====================
    total_tests = len(results)
    base_wins = sum(1 for r in results if r["winner"] == "base")
    finetuned_wins = sum(1 for r in results if r["winner"] == "finetuned")
    ties = sum(1 for r in results if r["winner"] == "tie")
    
    avg_base_score = sum(r["base_model"]["evaluation"]["total_score"] for r in results) / total_tests
    avg_finetuned_score = sum(r["finetuned_model"]["evaluation"]["total_score"] for r in results) / total_tests
    
    # 카테고리별 통계
    category_stats = {}
    for result in results:
        cat = result["category"]
        if cat not in category_stats:
            category_stats[cat] = {
                "base_scores": [],
                "finetuned_scores": [],
                "base_wins": 0,
                "finetuned_wins": 0
            }
        
        category_stats[cat]["base_scores"].append(result["base_model"]["evaluation"]["total_score"])
        category_stats[cat]["finetuned_scores"].append(result["finetuned_model"]["evaluation"]["total_score"])
        
        if result["winner"] == "base":
            category_stats[cat]["base_wins"] += 1
        elif result["winner"] == "finetuned":
            category_stats[cat]["finetuned_wins"] += 1
    
    # ==================== 결과 출력 ====================
    print("\n" + "="*80)
    print(" 비교 결과")
    print("="*80)
    print(f"\n전체 통계:")
    print(f"  총 테스트: {total_tests}")
    print(f"  Base 모델 승리: {base_wins} ({base_wins/total_tests*100:.1f}%)")
    print(f"  Finetuned 모델 승리: {finetuned_wins} ({finetuned_wins/total_tests*100:.1f}%)")
    print(f"  동점: {ties} ({ties/total_tests*100:.1f}%)")
    print(f"\n평균 점수:")
    print(f"  Base 모델: {avg_base_score:.3f}")
    print(f"  Finetuned 모델: {avg_finetuned_score:.3f}")
    print(f"  개선율: {(avg_finetuned_score - avg_base_score) / avg_base_score * 100:+.1f}%")
    
    print(f"\n카테고리별 통계:")
    for cat, stats in category_stats.items():
        avg_base = sum(stats["base_scores"]) / len(stats["base_scores"])
        avg_finetuned = sum(stats["finetuned_scores"]) / len(stats["finetuned_scores"])
        print(f"  [{cat}]")
        print(f"    Base: {avg_base:.3f} (승리 {stats['base_wins']}회)")
        print(f"    Finetuned: {avg_finetuned:.3f} (승리 {stats['finetuned_wins']}회)")
        print(f"    개선율: {(avg_finetuned - avg_base) / avg_base * 100:+.1f}%")
    
    # ==================== 결과 저장 ====================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON 저장
    results_file = output_path / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "base_url": base_url,
                "finetuned_url": finetuned_url,
                "total_tests": total_tests,
                "tested_at": datetime.now().isoformat()
            },
            "statistics": {
                "base_wins": base_wins,
                "finetuned_wins": finetuned_wins,
                "ties": ties,
                "avg_base_score": avg_base_score,
                "avg_finetuned_score": avg_finetuned_score,
                "improvement": (avg_finetuned_score - avg_base_score) / avg_base_score * 100
            },
            "category_stats": category_stats,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n 결과 저장: {results_file}")
    
    # 텍스트 리포트 저장
    report_file = output_path / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("모델 성능 비교 리포트\n")
        f.write("="*80 + "\n\n")
        
        # 샘플 비교 (승자/패자 각 3개)
        finetuned_better = [r for r in results if r["winner"] == "finetuned"][:3]
        base_better = [r for r in results if r["winner"] == "base"][:3]
        
        if finetuned_better:
            f.write(" Finetuned 모델이 우수한 케이스:\n\n")
            for r in finetuned_better:
                f.write(f"[{r['category']}] {r['question']}\n")
                f.write(f"Base 점수: {r['base_model']['evaluation']['total_score']:.3f}\n")
                f.write(f"Finetuned 점수: {r['finetuned_model']['evaluation']['total_score']:.3f}\n")
                f.write("-"*80 + "\n\n")
        
        if base_better:
            f.write(" Base 모델이 우수한 케이스:\n\n")
            for r in base_better:
                f.write(f"[{r['category']}] {r['question']}\n")
                f.write(f"Base 점수: {r['base_model']['evaluation']['total_score']:.3f}\n")
                f.write(f"Finetuned 점수: {r['finetuned_model']['evaluation']['total_score']:.3f}\n")
                f.write("-"*80 + "\n\n")
    
    print(f" 리포트 저장: {report_file}")
    print("="*80)

# ==================== 메인 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 성능 비교")
    parser.add_argument("--base-url", required=True, help="Base 모델 API URL")
    parser.add_argument("--finetuned-url", required=True, help="Finetuned 모델 API URL")
    parser.add_argument("--output-dir", default="./comparison_results", help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    compare_models(args.base_url, args.finetuned_url, args.output_dir)