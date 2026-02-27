# app/api/admin_router.py
"""
RAG 파라미터 런타임 수정 API (파일럿 전용)
- 재시작 없이 파라미터 즉시 반영
- 재시작 시 환경변수 기본값으로 복원됨
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
import app.api.llama_router as llama_module

router = APIRouter(prefix="/admin", tags=["⚙️ RAG 파라미터 설정"])


class RagConfigUpdate(BaseModel):
    # ===== 단문형 (short) =====
    short_top_k: Optional[int] = Field(
        None, ge=1, le=20,
        description="단문형 청크 수 (현재: 3, 범위: 1~20)"
    )
    short_max_tokens: Optional[int] = Field(
        None, ge=100, le=1000,
        description="단문형 출력 토큰 (현재: 400, 범위: 100~1000)"
    )
    short_temperature: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="단문형 temperature (0=결정적, 1=창의적)"
    )
    short_context_budget: Optional[int] = Field(
        None, ge=256, le=4096,
        description="단문형 컨텍스트 예산 토큰"
    )

    # ===== 장문형 (long) =====
    long_top_k: Optional[int] = Field(
        None, ge=1, le=30,
        description="장문형 청크 수 (현재: 10, 범위: 1~30)"
    )
    long_max_tokens: Optional[int] = Field(
        None, ge=500, le=7000,
        description="장문형 출력 토큰 (현재: 3096, 범위: 500~7000)"
    )
    long_temperature: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="장문형 temperature (현재: 0.1)"
    )
    long_context_budget: Optional[int] = Field(
        None, ge=512, le=8192,
        description="장문형 컨텍스트 예산 토큰 (현재: 4096)"
    )

    # ===== 초장문형 (ultra_long) =====
    ultra_top_k: Optional[int] = Field(
        None, ge=50, le=150,
        description="초장문형 청크 수 - LONG_CONTEXT_TOP_K (현재: 150)"
    )
    ultra_max_tokens: Optional[int] = Field(
        None, ge=1000, le=5000,
        description="초장문형 출력 토큰 - LONG_MAX_TOKENS (현재: 5000)"
    )
    ultra_temperature: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="초장문형 temperature - LONG_TEMPERATURE (현재: 0.1)"
    )

    # ===== 검색 임계값 =====
    base_score_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="단문/장문 리랭킹 최소 스코어 - BASE_SCORE_THRESHOLD (현재: 0.25)"
    )
    ultra_score_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="초장문 리랭킹 최소 스코어 - LONG_SCORE_THRESHOLD (현재: 0.25)"
    )


@router.get("/rag-config", summary="현재 RAG 파라미터 조회")
def get_rag_config():
    cfg = llama_module.RESPONSE_MODE_CONFIG
    return {
        "short": {
            "top_k": cfg["short"]["top_k"],
            "max_tokens": cfg["short"]["max_tokens"],
            "temperature": cfg["short"]["temperature"],
            "context_budget": llama_module.CONTEXT_BUDGET_TOKENS,
        },
        "long": {
            "top_k": cfg["long"]["top_k"],
            "max_tokens": cfg["long"]["max_tokens"],
            "temperature": cfg["long"]["temperature"],
            "context_budget": llama_module.LONG_CONTEXT_BUDGET_TOKENS_NORMAL,
        },
        "ultra_long": {
            "top_k": llama_module.LONG_CONTEXT_TOP_K,
            "max_tokens": llama_module.LONG_MAX_TOKENS,
            "temperature": llama_module.LONG_TEMPERATURE,
        },
        "search": {
            "base_score_threshold": llama_module.BASE_SCORE_THRESHOLD,
            "ultra_score_threshold": llama_module.LONG_SCORE_THRESHOLD,
        },
        "note": "서버 재시작 시 환경변수 기본값으로 복원됩니다."
    }


@router.post("/rag-config", summary="RAG 파라미터 수정 (변경할 항목만 전달)")
def update_rag_config(req: RagConfigUpdate):
    cfg = llama_module.RESPONSE_MODE_CONFIG
    changed = {}

    # 단문형
    if req.short_top_k is not None:
        cfg["short"]["top_k"] = req.short_top_k
        changed["short_top_k"] = req.short_top_k
    if req.short_max_tokens is not None:
        cfg["short"]["max_tokens"] = req.short_max_tokens
        changed["short_max_tokens"] = req.short_max_tokens
    if req.short_temperature is not None:
        cfg["short"]["temperature"] = req.short_temperature
        changed["short_temperature"] = req.short_temperature
    if req.short_context_budget is not None:
        llama_module.CONTEXT_BUDGET_TOKENS = req.short_context_budget
        changed["short_context_budget"] = req.short_context_budget

    # 장문형
    if req.long_top_k is not None:
        cfg["long"]["top_k"] = req.long_top_k
        changed["long_top_k"] = req.long_top_k
    if req.long_max_tokens is not None:
        cfg["long"]["max_tokens"] = req.long_max_tokens
        changed["long_max_tokens"] = req.long_max_tokens
    if req.long_temperature is not None:
        cfg["long"]["temperature"] = req.long_temperature
        changed["long_temperature"] = req.long_temperature
    if req.long_context_budget is not None:
        llama_module.LONG_CONTEXT_BUDGET_TOKENS_NORMAL = req.long_context_budget
        changed["long_context_budget"] = req.long_context_budget

    # 초장문형
    if req.ultra_top_k is not None:
        llama_module.LONG_CONTEXT_TOP_K = req.ultra_top_k
        changed["ultra_top_k"] = req.ultra_top_k
    if req.ultra_max_tokens is not None:
        llama_module.LONG_MAX_TOKENS = req.ultra_max_tokens
        changed["ultra_max_tokens"] = req.ultra_max_tokens
    if req.ultra_temperature is not None:
        llama_module.LONG_TEMPERATURE = req.ultra_temperature
        changed["ultra_temperature"] = req.ultra_temperature

    # 검색 임계값
    if req.base_score_threshold is not None:
        llama_module.BASE_SCORE_THRESHOLD = req.base_score_threshold
        changed["base_score_threshold"] = req.base_score_threshold
    if req.ultra_score_threshold is not None:
        llama_module.LONG_SCORE_THRESHOLD = req.ultra_score_threshold
        changed["ultra_score_threshold"] = req.ultra_score_threshold

    return {
        "message": "적용 완료 ✓",
        "changed": changed,
        "current": {
            "short": {
                "top_k": cfg["short"]["top_k"],
                "max_tokens": cfg["short"]["max_tokens"],
                "temperature": cfg["short"]["temperature"],
                "context_budget": llama_module.CONTEXT_BUDGET_TOKENS,
            },
            "long": {
                "top_k": cfg["long"]["top_k"],
                "max_tokens": cfg["long"]["max_tokens"],
                "temperature": cfg["long"]["temperature"],
                "context_budget": llama_module.LONG_CONTEXT_BUDGET_TOKENS_NORMAL,
            },
            "ultra_long": {
                "top_k": llama_module.LONG_CONTEXT_TOP_K,
                "max_tokens": llama_module.LONG_MAX_TOKENS,
                "temperature": llama_module.LONG_TEMPERATURE,
            },
            "search": {
                "base_score_threshold": llama_module.BASE_SCORE_THRESHOLD,
                "ultra_score_threshold": llama_module.LONG_SCORE_THRESHOLD,
            },
        }
    }


@router.post("/rag-config/reset", summary="RAG 파라미터 기본값 복원")
def reset_rag_config():
    import os
    cfg = llama_module.RESPONSE_MODE_CONFIG

    cfg["short"].update({
        "top_k": 3,
        "max_tokens": 400,
        "top_p": 0.9,
        "temperature": 0.0,
        "context_style": "concise",
    })
    cfg["long"].update({
        "top_k": 10,
        "max_tokens": 3096,
        "top_p": 0.92,
        "temperature": 0.1,
        "context_style": "detailed",
    })
    llama_module.CONTEXT_BUDGET_TOKENS = int(os.getenv("RAG_CONTEXT_BUDGET_TOKENS", "1024"))
    llama_module.LONG_CONTEXT_TOP_K = int(os.getenv("RAG_LONG_CONTEXT_TOP_K", "150"))
    llama_module.LONG_MAX_TOKENS = int(os.getenv("RAG_LONG_MAX_TOKENS", "5000"))
    llama_module.LONG_TEMPERATURE = float(os.getenv("RAG_LONG_TEMPERATURE", "0.1"))
    llama_module.LONG_CONTEXT_BUDGET_TOKENS_NORMAL = int(os.getenv("RAG_LONG_CONTEXT_BUDGET_NORMAL", "4096"))
    llama_module.BASE_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.25"))
    llama_module.LONG_SCORE_THRESHOLD = float(os.getenv("RAG_LONG_SCORE_THRESHOLD", "0.25"))

    return {
        "message": "기본값으로 초기화 완료 ✓",
        "current": {
            "short": cfg["short"],
            "long": cfg["long"],
            "ultra_long": {
                "top_k": llama_module.LONG_CONTEXT_TOP_K,
                "max_tokens": llama_module.LONG_MAX_TOKENS,
                "temperature": llama_module.LONG_TEMPERATURE,
            },
            "search": {
                "base_score_threshold": llama_module.BASE_SCORE_THRESHOLD,
                "ultra_score_threshold": llama_module.LONG_SCORE_THRESHOLD,
            },
        }
    }