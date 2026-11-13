"""EasyOCR 실행 공통부 테스트 — 운영(2026-11-13)에서 GPU 메모리 부족으로 업로드 문서가 글자 없이 색인된 문제.

실제 EasyOCR·GPU 없이 가짜 Reader 로 동작만 확인한다.
"""
import sys
import types

import pytest

from app.services import easyocr_runtime as rt

IMG = object()          # 가짜 Reader 는 이미지를 들여다보지 않는다


@pytest.fixture
def fake_easyocr(monkeypatch):
    state = {"created": [], "gpu": "ok"}

    class Reader:
        def __init__(self, langs, gpu=True):
            state["created"].append((tuple(langs), gpu))
            self.gpu = gpu

        def readtext(self, img, **kw):
            if self.gpu and state["gpu"] == "oom":
                raise RuntimeError("CUDA out of memory. Tried to allocate 1.11 GiB")
            if self.gpu and state["gpu"] == "boom":
                raise ValueError("bad image")
            return [([[0, 0], [1, 0], [1, 1], [0, 1]], "GPU" if self.gpu else "CPU", 0.9)]

    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(Reader=Reader))
    monkeypatch.setattr(rt, "_READERS", {})
    monkeypatch.setattr(rt, "_LOCKS", {})
    monkeypatch.setattr(rt, "_cpu_until", 0.0)
    return state


def test_gpu_oom_reads_the_page_again_on_cpu(fake_easyocr):
    fake_easyocr["gpu"] = "oom"
    res = rt.readtext(IMG, ["ko", "en"], gpu=True, detail=1)
    assert res[0][1] == "CPU"
    assert fake_easyocr["created"] == [(("ko", "en"), True), (("ko", "en"), False)]


def test_after_oom_gpu_is_not_retried_for_a_while(fake_easyocr):
    fake_easyocr["gpu"] = "oom"
    rt.readtext(IMG, ["ko", "en"], gpu=True)
    fake_easyocr["gpu"] = "ok"
    assert rt.readtext(IMG, ["ko", "en"], gpu=True)[0][1] == "CPU"      # 잠시 CPU 로 계속


def test_other_errors_are_not_hidden(fake_easyocr):
    fake_easyocr["gpu"] = "boom"
    with pytest.raises(ValueError):
        rt.readtext(IMG, ["ko", "en"], gpu=True)


def test_reader_is_created_once_and_reused(fake_easyocr):
    for _ in range(3):
        assert rt.readtext(IMG, ["ko", "en"], gpu=True)[0][1] == "GPU"
    assert fake_easyocr["created"] == [(("ko", "en"), True)]


def test_cpu_setting_never_touches_gpu(fake_easyocr):
    assert rt.readtext(IMG, ["ko", "en"], gpu=False)[0][1] == "CPU"
    assert fake_easyocr["created"] == [(("ko", "en"), False)]


@pytest.mark.parametrize("exc, oom", [
    (RuntimeError("CUDA out of memory. Tried to allocate 1.11 GiB"), True),
    (type("OutOfMemoryError", (RuntimeError,), {})("x"), True),
    (RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"), True),
    (ValueError("bad image"), False),
])
def test_is_cuda_oom(exc, oom):
    assert rt.is_cuda_oom(exc) is oom


# ---- file_parser.parse_pdf (Java 연계 업로드 경로) ------------------------------------------
def _blank_pdf(path):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    return str(path)


def test_parse_pdf_falls_back_to_tesseract_and_logs_when_easyocr_fails(tmp_path, monkeypatch, capsys):
    pytest.importorskip("pytesseract")
    from app.services import file_parser as fp

    def easy_fail(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setenv("OCR_MODE", "auto")
    monkeypatch.setenv("OCR_ENGINE", "easyocr")
    monkeypatch.setattr(fp, "_ocr_with_easyocr", easy_fail)
    monkeypatch.setattr(fp, "_ocr_with_tesseract", lambda imgs, by_page, lang: [(1, f"TEXT via {lang}")])
    out = fp.parse_pdf(_blank_pdf(tmp_path / "scan.pdf"), by_page=True)
    assert out == [(1, "TEXT via kor+eng")]
    assert "CUDA out of memory" in capsys.readouterr().out          # 예전에는 오류를 조용히 삼켰다


def test_parse_pdf_logs_when_it_leaves_placeholders(tmp_path, monkeypatch, capsys):
    pytest.importorskip("pytesseract")
    from app.services import file_parser as fp

    def fail(*a, **k):
        raise RuntimeError("engine down")

    monkeypatch.setenv("OCR_MODE", "auto")
    monkeypatch.setenv("OCR_ENGINE", "easyocr")
    monkeypatch.setattr(fp, "_ocr_with_easyocr", fail)
    monkeypatch.setattr(fp, "_ocr_with_tesseract", fail)
    out = fp.parse_pdf(_blank_pdf(tmp_path / "scan.pdf"), by_page=True)
    assert out == [(1, "[page 1: image or low-text content]")]
    log = capsys.readouterr().out
    assert "engine down" in log and "placeholder" in log


def test_pdf_fusion_tesseract_uses_tesseract_language_setting(monkeypatch):
    pytest.importorskip("pytesseract")
    fitz = pytest.importorskip("fitz")
    from app.services import pdf_fusion as pf

    seen = {}
    monkeypatch.setenv("OCR_ENGINE", "tesseract")
    monkeypatch.setenv("OCR_LANG", "ko,en")            # EasyOCR 용 값
    monkeypatch.setenv("OCR_LANGS", "kor+eng")         # Tesseract 용 값
    def fake_tesseract(img, lang):
        seen["lang"] = lang
        return "", []

    monkeypatch.setattr(pf, "_ocr_page_with_tesseract", fake_tesseract)
    doc = fitz.open()
    doc.new_page()
    pf._ocr_page_image(doc[0])
    assert seen["lang"] == "kor+eng"


def test_bad_retry_setting_does_not_break_fallback(fake_easyocr, monkeypatch):
    monkeypatch.setenv("OCR_GPU_RETRY_SEC", "")
    fake_easyocr["gpu"] = "oom"
    assert rt.readtext(IMG, ["ko", "en"], gpu=True)[0][1] == "CPU"


def test_parse_pdf_keeps_text_layer_when_easyocr_fails(tmp_path, monkeypatch):
    # 글자층이 있는데 OCR 이 필요하다고 판단된 경우: EasyOCR 가 실패하면 예전처럼 글자층을 쓴다(Tesseract 결과로 바꾸지 않음)
    pytest.importorskip("pytesseract")
    from app.services import file_parser as fp

    def easy_fail(*a, **k):
        raise RuntimeError("CUDA out of memory")

    calls = []
    monkeypatch.setenv("OCR_MODE", "auto")
    monkeypatch.setenv("OCR_ENGINE", "easyocr")
    monkeypatch.setattr(fp, "_extract_text_pdfminer", lambda path, by_page: [(1, "short")])
    monkeypatch.setattr(fp, "_ocr_with_easyocr", easy_fail)
    monkeypatch.setattr(fp, "_ocr_with_tesseract", lambda *a: calls.append(1) or [(1, "TESS")])
    assert fp.parse_pdf(_blank_pdf(tmp_path / "s.pdf"), by_page=True) == [(1, "short")]
    assert calls == []


def test_parse_pdf_runs_tesseract_once_when_it_fails_in_empty_path(tmp_path, monkeypatch):
    pytest.importorskip("pytesseract")
    from app.services import file_parser as fp

    calls = []

    def tess_fail(*a):
        calls.append(1)
        raise RuntimeError("tesseract down")

    monkeypatch.setenv("OCR_MODE", "auto")
    monkeypatch.setenv("OCR_ENGINE", "easyocr")
    monkeypatch.setattr(fp, "_ocr_with_easyocr", lambda *a: [])          # EasyOCR 결과 없음 → Tesseract 대체
    monkeypatch.setattr(fp, "_ocr_with_tesseract", tess_fail)
    out = fp.parse_pdf(_blank_pdf(tmp_path / "s.pdf"), by_page=True)
    assert out == [(1, "[page 1: image or low-text content]")]
    assert calls == [1]
