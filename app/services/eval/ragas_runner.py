# app/services/eval/ragas_runner.py

import os
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Any, Optional
import asyncio

def run_ragas_eval(
    questions: List[str],
    contexts: List[List[str]],
    answers: List[str],
    ground_truths: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    RAGAS 평가를 실행합니다.
    
    Args:
        questions: 질문 리스트
        contexts: 각 질문에 대한 컨텍스트 리스트의 리스트
        answers: 생성된 답변 리스트
        ground_truths: 정답 리스트 (선택적)
        metrics: 사용할 메트릭 리스트 (선택적)
        
    Returns:
        평가 결과 딕셔너리
    """
    try:
        # RAGAS 동적 임포트 (버전 호환성 대응)
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            answer_similarity,
            answer_correctness,
            faithfulness,
        )
        
        # context_relevancy 대신 context_precision과 context_recall 사용
        try:
            from ragas.metrics import context_precision, context_recall
            available_context_metrics = [context_precision, context_recall]
        except ImportError:
            print("[RAGAS] Context metrics not available in this version")
            available_context_metrics = []
        
        # 기본 메트릭 설정
        default_metrics = [
            answer_relevancy,
            faithfulness,
        ] + available_context_metrics
        
        # ground_truths가 있는 경우 정확성 메트릭 추가
        if ground_truths:
            try:
                default_metrics.extend([answer_similarity, answer_correctness])
            except ImportError:
                print("[RAGAS] Similarity/correctness metrics not available")
        
        # 사용자 지정 메트릭이 있는 경우 필터링
        if metrics:
            metric_map = {
                'answer_relevancy': answer_relevancy,
                'faithfulness': faithfulness,
                'answer_similarity': answer_similarity if ground_truths else None,
                'answer_correctness': answer_correctness if ground_truths else None,
            }
            
            # context_precision, context_recall 추가
            if available_context_metrics:
                if len(available_context_metrics) >= 2:
                    metric_map['context_precision'] = available_context_metrics[0]
                    metric_map['context_recall'] = available_context_metrics[1]
            
            selected_metrics = []
            for metric_name in metrics:
                if metric_name in metric_map and metric_map[metric_name] is not None:
                    selected_metrics.append(metric_map[metric_name])
                else:
                    print(f"[RAGAS] Metric '{metric_name}' not available, skipping")
            
            eval_metrics = selected_metrics if selected_metrics else default_metrics
        else:
            eval_metrics = default_metrics
        
        # 데이터셋 준비
        data = {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # 평가 실행
        print(f"[RAGAS] Running evaluation with {len(eval_metrics)} metrics")
        result = evaluate(dataset, metrics=eval_metrics)
        
        # 결과 정리
        eval_results = {
            "total_samples": len(questions),
            "metrics_used": [metric.__name__ for metric in eval_metrics],
            "scores": dict(result),
            "average_scores": {},
        }
        
        # 평균 점수 계산
        for key, value in result.items():
            if isinstance(value, (int, float)):
                eval_results["average_scores"][key] = value
            elif hasattr(value, "mean"):
                eval_results["average_scores"][key] = float(value.mean())
        
        print("[RAGAS] Evaluation completed successfully")
        return eval_results
        
    except ImportError as e:
        print(f"[RAGAS] Import error: {e}")
        return {
            "error": "RAGAS library not properly installed or configured",
            "message": str(e),
            "total_samples": len(questions),
            "metrics_used": [],
            "scores": {},
            "average_scores": {},
        }
    except Exception as e:
        print(f"[RAGAS] Evaluation failed: {e}")
        return {
            "error": "RAGAS evaluation failed",
            "message": str(e),
            "total_samples": len(questions),
            "metrics_used": [],
            "scores": {},
            "average_scores": {},
        }

def create_ragas_dataset(
    questions: List[str],
    contexts: List[List[str]],
    answers: List[str],
    ground_truths: Optional[List[str]] = None
) -> Dataset:
    """
    RAGAS 평가용 데이터셋을 생성합니다.
    
    Args:
        questions: 질문 리스트
        contexts: 각 질문에 대한 컨텍스트 리스트의 리스트
        answers: 생성된 답변 리스트
        ground_truths: 정답 리스트 (선택적)
        
    Returns:
        Dataset 객체
    """
    data = {
        "question": questions,
        "contexts": contexts,
        "answer": answers,
    }
    
    if ground_truths:
        data["ground_truth"] = ground_truths
    
    return Dataset.from_dict(data)

async def run_ragas_eval_async(
    questions: List[str],
    contexts: List[List[str]],
    answers: List[str],
    ground_truths: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    RAGAS 평가를 비동기로 실행합니다.
    
    Args:
        questions: 질문 리스트
        contexts: 각 질문에 대한 컨텍스트 리스트의 리스트
        answers: 생성된 답변 리스트
        ground_truths: 정답 리스트 (선택적)
        metrics: 사용할 메트릭 리스트 (선택적)
        
    Returns:
        평가 결과 딕셔너리
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        run_ragas_eval,
        questions,
        contexts,
        answers,
        ground_truths,
        metrics
    )

def validate_ragas_input(
    questions: List[str],
    contexts: List[List[str]],
    answers: List[str],
    ground_truths: Optional[List[str]] = None
) -> tuple[bool, str]:
    """
    RAGAS 입력 데이터의 유효성을 검증합니다.
    
    Args:
        questions: 질문 리스트
        contexts: 각 질문에 대한 컨텍스트 리스트의 리스트
        answers: 생성된 답변 리스트
        ground_truths: 정답 리스트 (선택적)
        
    Returns:
        (유효성, 오류 메시지) 튜플
    """
    # 기본 길이 검증
    if len(questions) != len(contexts) or len(questions) != len(answers):
        return False, "Questions, contexts, and answers must have the same length"
    
    if ground_truths and len(questions) != len(ground_truths):
        return False, "Ground truths must have the same length as questions"
    
    # 빈 데이터 검증
    if not questions:
        return False, "Questions cannot be empty"
    
    # 컨텍스트 구조 검증
    for i, context_list in enumerate(contexts):
        if not isinstance(context_list, list):
            return False, f"Context at index {i} must be a list"
        if not context_list:
            return False, f"Context at index {i} cannot be empty"
    
    # 빈 문자열 검증
    for i, question in enumerate(questions):
        if not question.strip():
            return False, f"Question at index {i} cannot be empty"
        if not answers[i].strip():
            return False, f"Answer at index {i} cannot be empty"
    
    return True, "Valid"

# 하위 호환성을 위한 별칭
evaluate_with_ragas = run_ragas_eval