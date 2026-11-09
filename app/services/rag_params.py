# app/services/rag_params.py
"""
질의 1건에만 적용하는 RAG 파라미터 계산 (관리자 시험용).

관리자 화면에서 슬라이더 값을 바꿔 가며 질문해 볼 때, 서버 전체 설정(llama_router 의 전역 값)은
건드리지 않고 그 요청에만 값을 적용하기 위한 계산을 한다. 표준 라이브러리만 사용한다.

- modes   : llama_router.RESPONSE_MODE_CONFIG (short/long 의 top_k, max_tokens, temperature, top_p ...)
- scalars : 전역 값(컨텍스트 예산, 초장문 설정, 점수 임계값)의 현재 값
- overrides: admin_router.RagConfigUpdate 와 같은 이름의 값 (None 은 "지정 안 함")
"""
import copy
from typing import Any, Dict, Optional

# 오버라이드 키 → (모드, 필드)
MODE_FIELDS = {
    "short_top_k": ("short", "top_k"),
    "short_max_tokens": ("short", "max_tokens"),
    "short_temperature": ("short", "temperature"),
    "long_top_k": ("long", "top_k"),
    "long_max_tokens": ("long", "max_tokens"),
    "long_temperature": ("long", "temperature"),
}

# 전역 스칼라 값으로 관리되는 키 (이름은 오버라이드 키와 같다)
SCALAR_KEYS = (
    "short_context_budget",
    "long_context_budget",
    "ultra_top_k",
    "ultra_max_tokens",
    "ultra_temperature",
    "base_score_threshold",
    "ultra_score_threshold",
)

ALL_KEYS = frozenset(MODE_FIELDS) | frozenset(SCALAR_KEYS)


def effective_params(
    modes: Dict[str, Dict[str, Any]],
    scalars: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """현재 설정에 오버라이드를 얹은 이 요청 전용 값을 돌려준다. 입력(modes, scalars)은 바꾸지 않는다."""
    ov = {k: v for k, v in (overrides or {}).items() if v is not None}
    unknown = set(ov) - ALL_KEYS
    if unknown:
        raise ValueError(f"알 수 없는 파라미터: {sorted(unknown)}")

    eff_modes = copy.deepcopy(modes)
    for key, (mode, field) in MODE_FIELDS.items():
        if key in ov:
            eff_modes[mode][field] = ov[key]

    eff: Dict[str, Any] = {"modes": eff_modes}
    for key in SCALAR_KEYS:
        eff[key] = ov[key] if key in ov else scalars[key]
    return eff
