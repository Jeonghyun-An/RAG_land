# app/services/ocr_service.py
from __future__ import annotations
import os, subprocess
from pathlib import Path
from typing import Tuple, Dict
from io import BytesIO

OCR_MODE = os.getenv("OCR_MODE", "auto")  # off | auto | force
OCR_LANGS = os.getenv("OCR_LANGS", "kor+eng")
OCR_MIN_CHARS_PER_PAGE = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", "50"))
OCR_MAX_PAGES_FOR_OCR = int(os.getenv("OCR_MAX_PAGES_FOR_OCR", "500"))

def _pdf_text_stats(pdf_path: str) -> Dict[str, int]:
    """가볍게 텍스트 레이어 유무만 체크 (PyPDF2). 실패해도 조용히 0으로."""
    pages, chars = 0, 0
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            r = PyPDF2.PdfReader(f)
            pages = len(r.pages)
            for i in range(pages):
                try:
                    t = r.pages[i].extract_text() or ""
                except Exception:
                    t = ""
                chars += len(t.strip())
    except Exception:
        pass
    return {"pages": pages, "chars": chars}

def _run_ocrmypdf(src_pdf: str, out_pdf: str) -> None:
    cmd = [
        "ocrmypdf",
        "--optimize", "1",
        "--deskew",
        "--clean",
        "-l", OCR_LANGS,
    ]
    if OCR_MODE == "force":
        cmd.append("--force-ocr")
    cmd += [src_pdf, out_pdf]
    subprocess.run(cmd, check=True)

def ocr_if_needed(pdf_path: str) -> Tuple[str, Dict]:
    """
    필요하면 OCR 수행 후 (검색가능 PDF) 경로 반환.
    (pdf_path, stats) 형태로 리턴. 실패시 원본 그대로.
    """
    src = Path(pdf_path)
    assert src.suffix.lower() == ".pdf", "OCR는 PDF만 허용"

    stats = _pdf_text_stats(str(src))
    if OCR_MODE == "off":
        return str(src), {"mode": "off", **stats}

    if stats["pages"] and stats["pages"] > OCR_MAX_PAGES_FOR_OCR:
        return str(src), {"mode": "skipped(too_many_pages)", **stats}

    need = OCR_MODE == "force" or (stats["chars"] < OCR_MIN_CHARS_PER_PAGE * max(1, stats["pages"]))
    if not need:
        return str(src), {"mode": "no_ocr", **stats}

    out = src.with_suffix(".ocr.pdf")
    try:
        _run_ocrmypdf(str(src), str(out))
        if out.exists() and out.stat().st_size > 0:
            return str(out), {"mode": "ocr_done", **stats}
        return str(src), {"mode": "ocr_failed_empty", **stats}
    except Exception as e:
        return str(src), {"mode": f"ocr_error:{e}", **stats}

def try_ocr_pdf_bytes(pdf_bytes: bytes, enabled: bool) -> str|None:
    """
    PDF 바이트를 이미지로 변환해 OCR. 엔진 미설치/비활성 시 None 반환.
    """
    if not enabled:
        return None
    try:
        # 예시 의사코드: pdf2image + easyocr (또는 사내 OCR API)
        from pdf2image import convert_from_bytes
        import easyocr
        images = convert_from_bytes(pdf_bytes, dpi=200)
        reader = easyocr.Reader(['ko','en'], gpu=True)
        txts = []
        for img in images:
            res = reader.readtext(np.array(img), detail=0, paragraph=True)
            txts.append("\n".join(res))
        return "\n".join(txts).strip() or None
    except Exception:
        return None