# app/services/ocr_text_fix.py
"""
OCR 이 알파벳 O 와 숫자 0 을 섞어 읽은 글자 처리 — 표준 라이브러리만 사용.

배경
  스캔 문서 OCR(Tesseract·EasyOCR)이 O 와 0 을 섞어 읽어 같은 코드가 두 형태로 색인된다.
  운영 데이터(2026-11-13, 단락 6,416개): KOF3 46회 / K0F3 37회, KOM2 22 / K0M2 7,
  2006 → 2OO6·20O6, 90 → 9O 등. 반면 UO2·U3O8·D2O 처럼 원래 알파벳 O 인 것도 있어
  한쪽으로 몰아 바꿀 수 없고, 코드마다 정답 표기를 확인할 수도 없다.

처리 방식 (사전 없이)
  1) fix_ocr_confusions : 오류가 확실한 패턴만 교정한다.
       - 대문자 사이에 낀 0 → O          (K0F3→KOF3, R0K→ROK)
       - 숫자로 시작하고 숫자와 O 로만 된 단어의 O → 0   (2OO6→2006, 9O→90)
     UO2·U3O8·H2O·O2 처럼 문자로 시작하거나 문자에 붙은 O 는 건드리지 않는다.
  2) fold_o0 : 비교할 때만 O 와 0 을 같은 글자로 본다(정답 표기를 몰라도 둘 다 찾힌다).
  3) align_codes : 질문에 나온 코드와 O/0 만 다른 단어를 질문의 표기로 맞춘다(K004 → KOO4).
  원문(색인 데이터)은 바꾸지 않는다. 검색 후보의 사본에만 적용한다.
"""
from __future__ import annotations

import re
from typing import Iterable

_UPPER_0_UPPER = re.compile(r"(?<=[A-Z])0(?=[A-Z])")
# 숫자로 시작, 숫자와 O 로만 이루어지고 O 가 하나 이상 있는 단어 (앞뒤가 영문·숫자가 아닐 것)
_NUM_WITH_O = re.compile(r"(?<![A-Za-z0-9])\d[0-9Oo]*[Oo][0-9Oo]*(?![A-Za-z0-9])")


def fix_ocr_confusions(text: str) -> str:
    """오류가 확실한 O/0 혼동만 교정한다. 여러 번 적용해도 결과가 같다."""
    if not text:
        return text
    text = _NUM_WITH_O.sub(lambda m: re.sub("[Oo]", "0", m.group(0)), text)
    return _UPPER_0_UPPER.sub("O", text)


def fold_o0(text: str) -> str:
    """비교용 키: 대문자로 바꾸고 0 을 O 로 본다. 화면·저장에는 쓰지 않는다."""
    return (text or "").upper().replace("0", "O")


def _code_pattern(code: str) -> re.Pattern:
    body = "".join("[O0]" if ch in "O0" else re.escape(ch) for ch in fold_o0(code))
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


def align_codes(text: str, codes: Iterable[str]) -> str:
    """text 안에서 codes 와 O/0(대소문자) 만 다른 단어를 codes 의 표기로 바꾼다."""
    if not text:
        return text
    for code in codes or ():
        if code:
            text = _code_pattern(code).sub(code, text)
    return text


def zero_form(code: str) -> str:
    """첫 글자 뒤의 O 를 0 으로 바꾼 표기(KOF3→K0F3, KOO4→K004). OCR 이 0 으로 읽은 단락을 찾는 보조 검색용."""
    return code[:1] + code[1:].replace("O", "0") if code else code
