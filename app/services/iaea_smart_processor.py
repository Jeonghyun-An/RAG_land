# app/services/iaea_smart_processor.py
"""
IAEA 헤더 스마트 처리 (조건부)

전략:
1. 페이지 수 체크 (≤10페이지만 처리)
2. 1페이지만 빠른 체크
3. IAEA 패턴 발견되면 처리, 아니면 스킵

목적: 200페이지 문서에서는 작동 안 함 (효율)
"""
import re
from typing import List, Tuple, Optional


# IAEA 6개 언어 블록
IAEA_LANGUAGES = """الوكالة الدولية للطاقة الذرية
国际原子能机构
International Atomic Energy Agency
Agence internationale de l'énergie atomique
Международное агентство по атомной энергии
Organismo Internacional de Energía Atómica"""


def should_check_iaea(pages: List[Tuple[int, str]]) -> bool:
    """
    IAEA 체크 여부 판단
    
    조건:
    - 페이지 수 ≤ 10 (레터, 짧은 문서)
    - 200페이지 문서는 체크 안 함 (효율)
    
    Args:
        pages: [(page_no, text), ...]
    
    Returns:
        True if IAEA 체크해야 함
    """
    page_count = len(pages)
    
    # 10페이지 이하만 (레터, 짧은 메모)
    if page_count <= 10:
        return True
    
    # 200페이지 문서는 스킵
    return False


def quick_detect_iaea(first_page_text: str) -> bool:
    """
    1페이지만 빠르게 IAEA 여부 감지
    
    Args:
        first_page_text: 1페이지 텍스트
    
    Returns:
        True if IAEA 문서로 보임
    """
    
    # 스페인어 패턴 (가장 안정적)
    spanish_patterns = [
        r'Organismo\s+Internacional',
        r'Organismo\s+Inte[rm]nacional',
        r'Organismo\s+\w+\s+de\s+\w+\s+Atomica',
    ]
    
    for pattern in spanish_patterns:
        if re.search(pattern, first_page_text, re.IGNORECASE):
            return True
    
    # 추가 시그널 (IAEA 로고 등)
    iaea_signals = [
        'IAEA',
        'Atoms for Peace',
        'Vienna International Centre',
    ]
    
    signal_count = sum(1 for s in iaea_signals if s in first_page_text)
    
    # 2개 이상 시그널 → IAEA 가능성 높음
    return signal_count >= 2


def fix_iaea_header(text: str) -> Tuple[str, bool]:
    """
    IAEA 헤더 수정 (스페인어 기준)
    
    Args:
        text: 1페이지 텍스트
    
    Returns:
        (fixed_text, was_fixed)
    """
    
    # 스페인어 패턴 (3단계 폴백)
    patterns = [
        re.compile(r'Organismo\s+Internacional\s+de\s+Energ[ií]a\s+At[óo]mica', re.IGNORECASE),
        re.compile(r'Organismo\s+Inte[rm]nacional\s+de\s+Energ[ífj]a\s+At[óo]mica', re.IGNORECASE),
        re.compile(r'Organismo\s+\w+\s+de\s+\w+\s+Atomica', re.IGNORECASE),
    ]
    
    match = None
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            break
    
    if not match:
        return text, False
    
    # 스페인어 이후 부분 (본문)
    after = text[match.end():].strip()
    
    # 교체
    fixed = f"IAEA\n{IAEA_LANGUAGES}\n\n{after}"
    
    return fixed, True


def process_pages_smart(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    스마트 IAEA 처리 (조건부)
    
    전략:
    1. 페이지 수 체크 (≤10만)
    2. 1페이지만 빠른 감지
    3. IAEA면 수정, 아니면 원본 유지
    
    Args:
        pages: [(page_no, text), ...]
    
    Returns:
        처리된 페이지 리스트
    """
    
    if not pages:
        return pages
    
    # 1. 페이지 수 체크
    if not should_check_iaea(pages):
        print(f"[IAEA-SMART] Skipped: {len(pages)} pages (too many, only process ≤10 pages)")
        return pages
    
    # 2. 1페이지만 빠른 감지
    first_page_no, first_page_text = pages[0]
    
    if first_page_no != 1:
        print(f"[IAEA-SMART] Skipped: first page is not page 1")
        return pages
    
    if not quick_detect_iaea(first_page_text):
        print(f"[IAEA-SMART] No replacement needed")
        return pages
    
    # 3. IAEA 헤더 수정
    fixed_text, was_fixed = fix_iaea_header(first_page_text)
    
    if was_fixed:
        print(f"[IAEA-SMART] Fixed page 1 header")
        # 1페이지만 교체, 나머지 그대로
        return [(1, fixed_text)] + pages[1:]
    else:
        print(f"[IAEA-SMART] IAEA detected but pattern not found")
        return pages


# 편의 함수
def process_if_needed(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    필요시에만 IAEA 처리 (메인 진입점)
    
    Args:
        pages: [(page_no, text), ...]
    
    Returns:
        처리된 페이지 리스트
    """
    return process_pages_smart(pages)