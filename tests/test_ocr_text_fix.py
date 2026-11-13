"""OCR O/0 혼동 처리(ocr_text_fix) 테스트 — 예시는 운영 데이터(2026-11-13)에서 실제로 나온 표기."""
import pytest

from app.services.ocr_text_fix import align_codes, fix_ocr_confusions, fold_o0, zero_form


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
])
def test_fixes_confident_patterns(raw, fixed):
    assert fix_ocr_confusions(raw) == fixed


@pytest.mark.parametrize("text", [
    "UO2 pellet", "U3O8 powder", "D2O", "H2O", "O2 gas", "CO2",   # 원래 알파벳 O
    "KOF3", "2006", "Room 101", "S7OM", "S70M", "0A3", "OA3",     # 정상 또는 규칙으로 판단 불가 → 그대로
    "META: {\"page\": 10, \"pages\": [10]}", "제10조", "Ko0a",
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
