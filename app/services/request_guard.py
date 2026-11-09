# app/services/request_guard.py
"""
질의 입·출력 보호 (보안대책 M09·M13·M14) — 표준 라이브러리만 사용.

입력
  - 제어문자·보이지 않는 문자 제거, 질의문 길이 제한(초과 시 거부)
  - 대화 이력은 최근 N건·건당 M자로 잘라서 받음(화면이 전체 이력을 보내므로 거부하지 않음)
  - 프롬프트 인젝션 패턴(지시 무시·시스템 프롬프트 노출 요청·탈옥) 탐지 → 차단·기록
출력
  - 개인정보 패턴(주민등록번호·전화번호·이메일) 마스킹
  - 답변에 섞인 지시문(시스템 프롬프트) 줄 제거, 많이 섞이면 답변 전체를 안내 문구로 대체
로그
  - 질의 내용은 남기지 않고 길이·해시만 기록(LOG_QUERY_TEXT=1 이면 앞부분 기록, 개발용)

환경변수 (괄호는 기본값)
  QUESTION_MAX_CHARS (2000)  HISTORY_MAX_ITEMS (10)  HISTORY_ITEM_MAX_CHARS (2000)
  INPUT_FILTER_ENFORCE (1)   "0" 이면 인젝션 패턴을 기록만 하고 통과
  OUTPUT_FILTER_ENFORCE (1)  "0" 이면 마스킹·지시문 제거를 하지 않음
  PII_MASK (rrn)             마스킹할 종류: rrn, mobile, landline, email (쉼표 구분) / all / none
  LOG_QUERY_TEXT (0)
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from typing import Any, Iterable, List, Optional, Tuple


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


QUESTION_MAX_CHARS = _int("QUESTION_MAX_CHARS", 2000)
HISTORY_MAX_ITEMS = _int("HISTORY_MAX_ITEMS", 10)
HISTORY_ITEM_MAX_CHARS = _int("HISTORY_ITEM_MAX_CHARS", 2000)


def input_filter_enforced() -> bool:
    return _flag("INPUT_FILTER_ENFORCE", "1")


def output_filter_enforced() -> bool:
    return _flag("OUTPUT_FILTER_ENFORCE", "1")


# ==================== 입력 정리 ====================
# 보이지 않는 문자(폭 없는 공백·방향 제어 등): 필터 우회에 쓰이므로 제거
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿­]")


def clean_text(text: Optional[str]) -> str:
    """제어문자(줄바꿈·탭 제외)와 보이지 않는 문자를 제거한다."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
    )


def validate_question(value: Any) -> str:
    """질의문 정리 + 길이 검사. 비었거나 길면 ValueError(→ 422)."""
    text = clean_text(value if isinstance(value, str) else str(value or "")).strip()
    if not text:
        raise ValueError("질문을 입력해 주세요.")
    if len(text) > QUESTION_MAX_CHARS:
        raise ValueError(f"질문은 {QUESTION_MAX_CHARS:,}자 이하로 입력해 주세요.")
    return text


def trim_history(value: Any) -> List[dict]:
    """대화 이력을 최근 HISTORY_MAX_ITEMS 건, 건당 HISTORY_ITEM_MAX_CHARS 자로 자른다."""
    if not value:
        return []
    if not isinstance(value, list):
        raise ValueError("history 는 목록이어야 합니다.")
    out: List[dict] = []
    for item in value[-HISTORY_MAX_ITEMS:] if HISTORY_MAX_ITEMS > 0 else []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", ""))[:20]
        content = clean_text(str(item.get("content", "") or ""))[:HISTORY_ITEM_MAX_CHARS]
        out.append({"role": role, "content": content})
    return out


# ==================== 프롬프트 인젝션 ====================
def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", clean_text(text)).lower()
    return re.sub(r"\s+", " ", text)


# 오탐을 줄이기 위해 "AI 에게 내리는 명령" 형태만 잡는다.
#  - 무시: 앞선 지시를 가리키는 말 + 지시/프롬프트 + 명령형(무시해/무시하고/잊어 …).
#    '원안위 명령을 무시하면 처벌은?'처럼 조건·과거형(무시하면/무시한/무시할)으로 묻는 업무 질문은 통과.
#  - 노출: 시스템 프롬프트·너의 지시 등 AI 자신의 설정 + 요청형(보여줘/알려줘/출력해/뭐야 …).
#    'KINAC 내부 지시사항', 'IAEA initial instructions' 같은 업무 용어는 통과.
_EN_PATTERNS = [
    ("en_ignore_instructions", re.compile(
        r"\b(ignore|disregard|forget)\b\s+(all\s+|any\s+|of\s+)*(the\s+|your\s+|my\s+)?"
        r"(previous|prior|above|earlier|preceding|foregoing|system)\s+(instructions?|prompts?|directions|directives)\b")),
    ("en_reveal_system_prompt", re.compile(
        r"\b(reveal|show|print|repeat|output|display|leak|tell|give|share|translate|summari[sz]e|what'?s|what\s+(is|are)|dump)\b"
        r"[^.\n]{0,30}?(\byour\s+(system\s+|hidden\s+|secret\s+|initial\s+|original\s+)?(prompts?|instructions?)\b|\bsystem\s+prompts?\b)")),
    ("en_repeat_above", re.compile(
        r"\b(repeat|print|output|copy)\b[^.\n]{0,20}?\b(everything|all\s+(the\s+)?text|all)\b[^.\n]{0,10}?\babove\b")),
    ("en_jailbreak", re.compile(
        r"\b(jailbreak|do\s+anything\s+now|dan\s+mode)\b|\byou\s+are\s+(now\s+)?dan\b")),
    ("en_no_restrictions", re.compile(
        r"\b(act\s+as|pretend|role-?play\s+as)\b[^.\n]{0,30}?\b(without|no|free\s+of)\b[^.\n]{0,15}?"
        r"\b(restrictions?|rules|filters?|guidelines|limits)\b")),
]

_KO_B = r"(?<![가-힣])"   # 앞말에 붙은 글자 제외(원안'위', 상'위' 등)
_KO_GAP = r"[^.?!\n]{0,8}?"
_KO_PATTERNS = [
    ("ko_ignore_instructions", re.compile(
        _KO_B + r"(이전|앞의|앞|위의|위|앞서|지금까지|위에서)\s*(의\s*|에\s*)?(받은\s*|말한\s*|주어진\s*|준\s*)?(모든\s*)?"
        r"(지시|지시사항|지시문|프롬프트|규칙|명령|instructions?|prompts?)(들)?"
        + _KO_GAP + r"(무시하고|무시해|무시하세요|무시하십시오|무시할\s*것|잊어|잊고|잊으세요|잊으십시오|따르지\s*말고|따르지\s*마)")),
    ("ko_reveal_system_prompt", re.compile(
        r"(시스템\s*프롬프트|system\s*prompt|프롬프트|" + _KO_B + r"(너의|네|당신의|니)\s*(지시|지시사항|지시문|설정|규칙)|숨겨진\s*(지시|프롬프트))"
        + _KO_GAP + r"(보여|출력해|출력하|알려|공개해|말해|복사해|반복해|번역해|요약해|뭐야|뭐니|뭔지|무엇인지|그대로)")),
    ("ko_jailbreak", re.compile(
        _KO_B + r"탈옥|개발자\s*모드로|dan\s*모드|제한\s*없이\s*(답해|대답해|말해|답하)")),
]

_SINGLE_SYLLABLES = re.compile(r"(?:(?<=\s)|^)(?:[가-힣]\s+){2,}[가-힣](?=\s|$|[.?!])")


def _collapse_spaced_syllables(text: str) -> str:
    """'이 전 지 시 를 무 시 해'처럼 한 글자씩 띄운 부분만 붙인다(일반 띄어쓰기는 그대로)."""
    return _SINGLE_SYLLABLES.sub(lambda m: re.sub(r"\s+", "", m.group(0)), text)


def detect_injection(text: Optional[str]) -> Optional[str]:
    """인젝션 패턴이면 패턴 이름을, 아니면 None 을 돌려준다."""
    if not text:
        return None
    norm = _normalize(text)
    for name, pat in _EN_PATTERNS:
        if pat.search(norm):
            return name
    for candidate in (norm, _collapse_spaced_syllables(norm)):
        for name, pat in _KO_PATTERNS:
            if pat.search(candidate):
                return name
    return None


INJECTION_REFUSAL = {
    "ko": "질문에 시스템 지시를 바꾸거나 내부 설정을 요구하는 표현이 있어 답변할 수 없습니다. 질문 내용을 바꿔 다시 입력해 주세요.",
    "en": "Your question contains wording that tries to change system instructions or reveal internal settings, so it cannot be answered. Please rephrase your question.",
}

LEAK_REPLACEMENT = {
    "ko": "답변을 만드는 과정에서 내부 지시문이 섞여 답변을 표시하지 않습니다. 질문을 바꿔 다시 시도해 주세요.",
    "en": "The generated answer contained internal instructions, so it is not shown. Please rephrase and try again.",
}


# ==================== 개인정보 마스킹 ====================
# 경계는 영문·숫자·구분기호만 본다. 한글 조사가 붙은 '010-1234-5678입니다'도 잡고,
# 문서 번호(RR-02-2021-0001)·날짜·수량처럼 다른 영숫자와 이어진 숫자는 제외한다.
_B = r"(?<![0-9A-Za-z./-])"
_E = r"(?![0-9A-Za-z-])"
_SEP = r"(?:[ \t]*[-.][ \t]*|[ \t])"          # 줄바꿈은 구분자로 보지 않음(표의 숫자 열 오탐 방지)
_RRN = re.compile(_B + r"\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[ \t]?-[ \t]?[1-8]\d{6}" + _E)
_MOBILE = re.compile(_B + r"(?:(01[016789])" + _SEP + r"\d{3,4}" + _SEP + r"\d{4}|(010)\d{8})" + _E)
_LANDLINE = re.compile(_B + r"(\(?(?:02|0[3-6][1-5]|070)\)?)" + _SEP + r"(\d{3,4})" + _SEP + r"\d{4}" + _E)
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]{1,64})@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})")
# 가운데가 연도(19xx·20xx)인 '02-2021-0001'은 문서 번호일 가능성이 커서, 앞에 연락처 단서가 있을 때만 가린다.
_YEAR = re.compile(r"(?:19|20)\d{2}")
_PHONE_CUE = re.compile(r"(전화|연락처|연락|휴대폰|핸드폰|팩스|담당|tel|fax|phone|mobile|☎|문의)", re.I)


def _mask_digits_after(s: str, keep: int) -> str:
    """앞 keep 글자는 두고 나머지 숫자만 * 로 바꾼다(구분자·괄호 모양은 유지)."""
    return s[:keep] + re.sub(r"\d", "*", s[keep:])


PII_TYPES = ("rrn", "mobile", "landline", "email")


def pii_mask_types() -> set:
    """마스킹할 종류. PII_MASK(쉼표 구분: rrn, mobile, landline, email / all / none), 기본은 rrn(주민등록번호)만.
    업무 연락처(담당자 전화·메일) 조회를 쓰는 환경을 고려해 전화·메일은 기관 요청 시 켠다."""
    raw = os.getenv("PII_MASK", "rrn").strip().lower()
    if raw in ("all", "*"):
        return set(PII_TYPES)
    if raw in ("none", "off", "0", ""):
        return set()
    return {t.strip() for t in raw.split(",") if t.strip() in PII_TYPES}


def mask_pii(text: Optional[str]) -> Tuple[str, int]:
    """(마스킹한 문자열, 마스킹 건수). 마스킹할 종류는 PII_MASK 를 따른다."""
    if not text:
        return text or "", 0
    types = pii_mask_types()
    if not types:
        return text, 0
    count = 0

    def _rrn(m):
        nonlocal count
        count += 1
        return re.sub(r"\d", "*", m.group(0))

    def _has_cue(m) -> bool:
        return bool(_PHONE_CUE.search(text[max(0, m.start() - 20):m.start()]))

    def _spaces_only(s: str) -> bool:
        # 공백으로만 띄운 숫자 묶음(표의 숫자 열 등)은 연락처 단서가 있을 때만 전화번호로 본다
        return not re.search(r"[-.]", s)

    def _mobile(m):
        nonlocal count
        if _spaces_only(m.group(0)) and not _has_cue(m):
            return m.group(0)
        count += 1
        return _mask_digits_after(m.group(0), 3)

    def _landline(m):
        nonlocal count
        if (_YEAR.fullmatch(m.group(2)) or _spaces_only(m.group(0)[len(m.group(1)):])) and not _has_cue(m):
            return m.group(0)
        count += 1
        return _mask_digits_after(m.group(0), len(m.group(1)))

    def _email(m):
        nonlocal count
        count += 1
        return f"{m.group(1)[:1]}***@{m.group(2)}"

    if "rrn" in types:
        text = _RRN.sub(_rrn, text)
    if "mobile" in types:
        text = _MOBILE.sub(_mobile, text)
    if "landline" in types:
        text = _LANDLINE.sub(_landline, text)
    if "email" in types:
        text = _EMAIL.sub(_email, text)
    return text, count


# ==================== 지시문(시스템 프롬프트) 노출 ====================
_QUOTED = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”|'[^'\n]*'")
_MARKUP = re.compile(r"[\s#*>`|•·\-–—_=~:;,.!?()\[\]{}\"'“”/\\]+")
LEAK_MIN_CHARS = 20          # 이보다 짧은 지시문 줄은 비교하지 않음(제목·짧은 형식어)
LEAK_BLOCK_LINES = 3         # 이만큼 이상 섞이면 답변 전체를 대체


def _key(line: str) -> str:
    return _MARKUP.sub("", unicodedata.normalize("NFKC", line).lower())


def instruction_lines(prompt: str, exclude: Iterable[Optional[str]] = ()) -> List[str]:
    """프롬프트에서 문서(context)·질문 줄을 뺀 지시문 줄의 비교 키 목록.
    - 문서·질문은 줄 단위로 뺀다(질문 단어가 지시문 안에도 있을 때 지시문이 깨지지 않도록).
    - 답변에 그대로 써야 하는 문구는 따옴표 안에 있으므로 빼고 비교한다.
    - 마크다운 제목 줄(# …)은 답변 형식으로 쓰라고 준 것이라 비교하지 않는다."""
    skip = set()
    for part in exclude:
        for line in (part or "").splitlines():
            k = _key(line)
            if k:
                skip.add(k)
    keys = []
    for line in (prompt or "").splitlines():
        if line.lstrip().startswith("#") or _key(line) in skip:
            continue
        k = _key(_QUOTED.sub(" ", line))
        if len(k) >= LEAK_MIN_CHARS:
            keys.append(k)
    return keys


def strip_instruction_leak(answer: str, prompt: str, exclude: Iterable[Optional[str]] = ()) -> Tuple[str, int, bool]:
    """(정리한 답변, 제거한 줄 수, 전체 대체 여부). 대체 여부가 True 면 호출 측에서 안내 문구로 바꾼다.
    지시문 한 줄이 답변 줄 안에 그대로 들어 있을 때만 그 줄을 뺀다(답변 줄이 지시문의 일부인 경우는 정상으로 봄)."""
    if not answer or not prompt:
        return answer, 0, False
    keys = instruction_lines(prompt, exclude)
    if not keys:
        return answer, 0, False
    kept, removed = [], 0
    for line in answer.splitlines():
        k = _key(line)
        if len(k) >= LEAK_MIN_CHARS and any(key in k for key in keys):
            removed += 1
            continue
        kept.append(line)
    if not removed:
        return answer, 0, False
    cleaned = "\n".join(kept).strip()
    if removed >= LEAK_BLOCK_LINES or not cleaned:
        return answer, removed, True
    return cleaned, removed, False


# ==================== 로그용 요약 ====================
def describe_query(text: Optional[str]) -> str:
    """질의 내용 대신 길이·해시만 남긴다(M09). LOG_QUERY_TEXT=1 이면 앞 120자."""
    text = text or ""
    if _flag("LOG_QUERY_TEXT", "0"):
        return text[:120]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"len={len(text)} sha={digest}"
