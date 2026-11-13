"""OCR O/0 혼동 처리(ocr_text_fix) 테스트 — 예시는 운영 데이터(2026-11-13)에서 실제로 나온 표기."""
import pytest

from app.services.ocr_text_fix import (
    align_codes, contains_code, fix_ocr_confusions, fold_o0, variant_queries, zero_form,
)


@pytest.mark.parametrize("raw, fixed", [
    ("K0F3 시설", "KOF3 시설"),               # 대문자 사이의 0
    ("K0F2, K0C2, K0M2", "KOF2, KOC2, KOM2"),
    ("10-R0K-009", "10-ROK-009"),
    ("K0O6/2006/002", "KOO6/2006/002"),
    ("2OO6년", "2006년"),                      # 숫자 속 O
    ("20O6, 2O06, 2oO6", "2006, 2006, 2006"),
    ("9O%", "90%"),
    ("1O-ROK-0O2", "10-ROK-002"),
    ("KOV1/2006/0O2", "KOV1/2006/002"),
    ("MT2-0I26, XBT1/8I0", "MT2-0126, XBT1/810"),   # 숫자 속 I·l → 1 (봉인 번호)
    ("20l1년, 26I0", "2011년, 2610"),
])
def test_fixes_confident_patterns(raw, fixed):
    assert fix_ocr_confusions(raw) == fixed


@pytest.mark.parametrize("text", [
    "UO2 pellet", "U3O8 powder", "D2O", "H2O", "O2 gas", "CO2",   # 원래 알파벳 O
    "KOF3", "2006", "Room 101", "S7OM", "S70M", "0A3", "OA3",     # 정상 또는 규칙으로 판단 불가 → 그대로
    "META: {\"page\": 10, \"pages\": [10]}", "제10조", "Ko0a",
    "KOMI", "KOEl", "IAEA", "III", "10L", "Il",                 # 문자로 시작하거나 대문자 L → 그대로
    "131I", "125I", "10l",                                       # 숫자 뒤 끝의 I·l(동위원소·단위) → 그대로
])
def test_leaves_other_text_unchanged(text):
    assert fix_ocr_confusions(text) == text


def test_fix_is_idempotent():
    s = "K0F3 2OO6 9O UO2 1O-R0K"
    assert fix_ocr_confusions(fix_ocr_confusions(s)) == fix_ocr_confusions(s)


def test_fold_treats_o_and_zero_as_same():
    assert fold_o0("K004") == fold_o0("KOO4") == fold_o0("k0o4")
    assert fold_o0("S70M") == fold_o0("S7OM")
    assert fold_o0("KOF3") != fold_o0("KOF8")


def test_align_codes_uses_question_spelling():
    text = "K004 inspection, KO04 report, k0o4 seal; KOO41 and XK004 untouched"
    out = align_codes(text, ["KOO4"])
    assert out.startswith("KOO4 inspection, KOO4 report, KOO4 seal;")
    assert "KOO41" in out and "XK004" in out           # 앞뒤가 붙은 다른 단어는 그대로


def test_zero_form():
    assert zero_form("KOF3") == "K0F3"
    assert zero_form("KOO4") == "K004"


# ---- 1 / l / I 혼동 — 운영 데이터(2026-11-13): KOM1 → KOMI 16·KOML 5, KOE1 → KOEl 13·KOEI 9 ...
# KOI-(DUF4 전환시설)와 KOL-(PIEF)은 서로 다른 실제 시설 코드라 I 와 L 을 같게 보면 안 된다.
@pytest.mark.parametrize("text", ["KOMI", "KOMl", "KOML", "K0MI", "kom1", "KOM1"])
def test_digit_one_in_code_matches_ocr_variants(text):
    assert contains_code(f"ICR at {text} facility", "KOM1")


def test_letter_i_and_letter_l_codes_stay_distinct():
    assert contains_code("Facility Code: Kol-", "KOI")          # I 를 l 로 읽은 것
    assert not contains_code("Facility Code: KOL-", "KOI")
    assert not contains_code("Facility Code: KOI-", "KOL")
    assert not contains_code("seal at KOF8", "KOF3")


def test_contains_code_keeps_substring_match():
    assert contains_code("KOE12006001_AL.pdf", "KOE1")           # 파일명처럼 붙어 있어도 찾음(기존과 같음)
    assert contains_code("seal at K004", "KOO4")
    assert contains_code("XKOF3Y", "KOF3")                       # 예전 부분 문자열 비교로 찾던 것은 그대로 찾음


@pytest.mark.parametrize("text, code", [
    ("Zangger-Komitees", "KOM1"), ("Modell des Zusatzprotokolls", "KOI"), ("Rilakovic, H.", "KOV1"),
])
def test_ocr_variant_match_skips_ordinary_words(text, code):
    assert not contains_code(text, code)                         # 일반 단어 속에 묻힌 것은 OCR 표기로 보지 않음


def test_align_codes_handles_one_variants():
    out = align_codes("KOMI, KOMl, KOML and K0MI report", ["KOM1"])
    assert out == "KOM1, KOM1, KOM1 and KOM1 report"
    assert align_codes("KOI- and KOL- sites", ["KOI"]) == "KOI- and KOL- sites"
    assert align_codes("KOI- and KOL- sites", ["KOL"]) == "KOI- and KOL- sites"


def test_align_codes_single_pass_keeps_each_code():
    # 질문에 비슷한 코드가 둘 있어도 서로 덮어쓰지 않는다(정확히 같은 표기를 먼저 고름)
    assert align_codes("KOM1 and KOMI and kom1", ["KOM1", "KOMI"]) == "KOM1 and KOMI and KOM1"
    assert align_codes("kof3의 봉인", ["KOF3"]) == "KOF3의 봉인"


def test_variant_queries():
    assert variant_queries("KOM1 ICR report", ["KOM1"]) == [
        "K0M1 ICR report", "KOMI ICR report", "KOMl ICR report"]
    assert variant_queries("KOF3 seal", ["KOF3"]) == ["K0F3 seal"]
    assert variant_queries("KOI- plant", ["KOI"]) == ["K0I- plant"]   # 문자 I 는 바꿔 검색하지 않음
    assert variant_queries("no code here", []) == []
