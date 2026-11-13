# app/services/ocr_text_fix.py
"""
OCR 이 비슷한 글자를 섞어 읽은 것(O/0, 1/l/I) 처리 — 표준 라이브러리만 사용.

배경
  스캔 문서 OCR(Tesseract·EasyOCR)이 O 와 0, 1 과 l·I·L 을 섞어 읽어 같은 코드가 여러 형태로 색인된다.
  운영 데이터(2026-11-13, 단락 6,416개): KOF3 46회 / K0F3 37회, KOM2 22 / K0M2 7,
  KOM1 → KOMI 16·KOML 5, KOE1 → KOEl 13·KOEI 9, KOO1 → KOOl 10, 2006 → 2OO6·20O6, 90 → 9O 등.
  반면 UO2·U3O8·D2O 처럼 원래 알파벳 O 인 것도 있어 한쪽으로 몰아 바꿀 수 없고,
  KOI-(DUF4 전환시설)와 KOL-(PIEF)처럼 I 와 L 이 실제로 다른 코드도 있어 1·I·L 을 모두 같게 볼 수도 없다.
  코드마다 정답 표기를 확인할 수도 없다.

처리 방식 (사전 없이)
  1) fix_ocr_confusions : 오류가 확실한 패턴만 교정한다.
       - 대문자 사이에 낀 0 → O                                   (K0F3→KOF3, R0K→ROK)
       - 숫자로 시작하고 숫자와 O·l·I 로만 된 단어 → 숫자          (2OO6→2006, 9O→90, 0I26→0126)
         I·l 은 뒤에 숫자가 이어질 때만 바꾼다(끝에 붙은 131I 동위원소·10l 단위는 그대로).
     UO2·H2O·KOMI 처럼 문자로 시작하거나 문자에 붙은 것, 대문자 L(단위 10L 등)은 건드리지 않는다.
  2) 코드 비교(contains_code·align_codes)는 질문 쪽 글자 기준으로 OCR 이 섞어 읽는 글자까지 같게 본다.
       질문의 O·0 → O 0          질문의 숫자 1 → 1 I l L          질문의 문자 I → I l 1
       질문의 문자 L → L 만      (그래서 KOI 질문에 KOL- 단락이 섞이지 않는다)
     대소문자는 구분하지 않는다. 일반 단어 속에 묻힌 것(Komitees, Protokoll)은 OCR 표기로 보지 않는다.
  3) align_codes : 질문에 나온 코드와 위 규칙으로 같은 단어를 질문의 표기로 맞춘다(K004 → KOO4, KOMI → KOM1).
  4) variant_queries : OCR 이 다르게 읽은 단락을 찾도록 코드를 바꿔 쓴 보조 검색 질의(K0F3, KOMI, KOMl).
  원문(색인 데이터)은 바꾸지 않는다. 검색 후보의 사본에만 적용한다.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List

_UPPER_0_UPPER = re.compile(r"(?<=[A-Z])0(?=[A-Z])")
# 숫자로 시작, 숫자와 O·l·I 로만 이루어지고 그중 하나 이상이 있는 단어 (앞뒤가 영문·숫자가 아닐 것)
_NUM_WITH_LETTERS = re.compile(r"(?<![A-Za-z0-9])\d[0-9OolI]*[OolI][0-9OolI]*(?![A-Za-z0-9])")
_O_TO_0 = str.maketrans("Oo", "00")
_LI_BEFORE_DIGIT = re.compile(r"[lI](?=[0-9OolI])")

# 질문 코드의 글자 → 문서에서 같은 글자로 볼 글자 (대소문자 포함). 여기 없는 글자는 대소문자만 무시.
_OCR_CLASS = {
    "O": "[Oo0]", "0": "[Oo0]",
    "1": "[1IilL]",          # 숫자 1 을 I·l·L 로 읽음
    "I": "[Iil1]",           # 문자 I 를 l·1 로 읽음. L 은 넣지 않는다(KOI ≠ KOL)
}


def fix_ocr_confusions(text: str) -> str:
    """오류가 확실한 O/0·1/l/I 혼동만 교정한다. 여러 번 적용해도 결과가 같다."""
    if not text:
        return text
    text = _NUM_WITH_LETTERS.sub(lambda m: _LI_BEFORE_DIGIT.sub("1", m.group(0)).translate(_O_TO_0), text)
    return _UPPER_0_UPPER.sub("O", text)


def fold_o0(text: str) -> str:
    """비교용 키: 대문자로 바꾸고 0 을 O 로 본다. 화면·저장에는 쓰지 않는다."""
    return (text or "").upper().replace("0", "O")


def _code_body(code: str) -> str:
    parts = []
    for ch in code.upper():
        if ch in _OCR_CLASS:
            parts.append(_OCR_CLASS[ch])
        elif "A" <= ch <= "Z":
            parts.append(f"[{ch}{ch.lower()}]")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


@lru_cache(maxsize=512)
def _code_regex(code: str, bounded: bool) -> re.Pattern:
    """bounded=True : 앞뒤가 영문·숫자가 아닌 단어 전체 (align_codes)
       bounded=False: 앞은 영문·숫자가 아니고 뒤에 소문자가 붙지 않음 — 파일명 KOE12006001 은 찾고
                      Komitees·Protokoll 같은 일반 단어 속은 제외 (contains_code)"""
    body = _code_body(code)
    tail = r"(?![A-Za-z0-9])" if bounded else r"(?![a-z])"
    return re.compile(rf"(?<![A-Za-z0-9]){body}{tail}")


def contains_code(text: str, code: str) -> bool:
    """text 에 code 가 있는지 — OCR 이 섞어 읽는 글자까지 같게 본다.
    예전 비교(대소문자·O/0 무시한 부분 문자열)로 찾던 것은 그대로 찾는다."""
    if not text or not code:
        return False
    if fold_o0(code) in fold_o0(text):
        return True
    return _code_regex(code, False).search(text) is not None


@lru_cache(maxsize=256)
def _align_plan(codes: tuple) -> tuple:
    combined = re.compile("|".join(f"(?:{_code_regex(c, True).pattern})" for c in codes))
    return combined, {fold_o0(c): c for c in reversed(codes)}


def align_codes(text: str, codes: Iterable[str]) -> str:
    """text 안에서 codes 와 OCR 혼동 글자만 다른 단어를 codes 의 표기로 바꾼다(앞뒤가 붙은 단어는 그대로).
    한 번에 바꾼다: 비슷한 코드가 여럿이어도(KOM1·KOMI) 바꾼 결과를 다시 다른 코드로 바꾸지 않는다."""
    codes = tuple(dict.fromkeys(c for c in (codes or ()) if c))
    if not text or not codes:
        return text
    combined, exact = _align_plan(codes)

    def pick(m: re.Match) -> str:
        tok = m.group(0)
        if fold_o0(tok) in exact:                       # 표기가 그대로 같은 코드(대소문자·O/0 만 다름)를 먼저
            return exact[fold_o0(tok)]
        return next((c for c in codes if _code_regex(c, True).fullmatch(tok)), tok)

    return combined.sub(pick, text)


def zero_form(code: str) -> str:
    """첫 글자 뒤의 O 를 0 으로 바꾼 표기(KOF3→K0F3, KOO4→K004). OCR 이 0 으로 읽은 단락을 찾는 보조 검색용."""
    return code[:1] + code[1:].replace("O", "0") if code else code


# 보조 검색에 쓸 바꿔 쓰기 — 운영 데이터에서 많이 나온 순서 (O→0, 1→I, 1→l)
_VARIANTS = (
    zero_form,
    lambda c: c[:1] + c[1:].replace("1", "I"),
    lambda c: c[:1] + c[1:].replace("1", "l"),
)


def variant_queries(query: str, codes: Iterable[str]) -> List[str]:
    """query 의 코드를 OCR 이 읽었을 법한 표기로 바꾼 보조 검색 질의(최대 3개, 바뀐 것만)."""
    codes = [c for c in (codes or ()) if c]
    if not query or not codes:
        return []
    out: List[str] = []
    for make in _VARIANTS:
        q = query
        for code in codes:
            alt = make(code)
            q = re.sub(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])",
                       lambda _m, a=alt: a, q, flags=re.IGNORECASE)
        if q != query and q not in out:
            out.append(q)
    return out
