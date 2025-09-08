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

def _norm_easyocr_langs(lang: str) -> list[str]:
    raw = [t.strip() for t in (lang or "ko,en").replace("+", ",").split(",") if t.strip()]
    alias = {
        "kor": "ko", "kr": "ko", "korean": "ko", "ko": "ko",
        "eng": "en", "english": "en", "en": "en",
        "jpn": "ja", "jp": "ja", "japanese": "ja", "ja": "ja",
        # 필요시 더 추가
    }
    out = []
    seen = set()
    for t in raw:
        v = alias.get(t.lower(), t.lower())
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out

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

# app/services/ocr_service.py

def try_ocr_pdf_bytes(pdf_bytes: bytes, enabled: bool) -> str | None:
    """
    PDF 바이트를 PyMuPDF로 렌더링 → 선택 엔진(easyocr|tesseract)로 OCR.
    외부 poppler 의존성 없음. 실패/비활성 시 None.
    ENV:
      OCR_ENGINE: easyocr|tesseract (default easyocr)
      OCR_LANG:   easyocr: "ko,en" / tesseract: "kor+eng"
      OCR_EASYOCR_GPU: "1"이면 GPU 사용
      OCR_DPI:    렌더 DPI (기본 300)
    """
    if not enabled:
        return None
    try:
        import fitz  # PyMuPDF
        import numpy as np
        dpi = int(os.getenv("OCR_DPI", "300"))
        zoom = max(1.0, dpi / 72.0)
        mat = fitz.Matrix(zoom, zoom)

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        engine = os.getenv("OCR_ENGINE", "easyocr").strip().lower()

        texts: list[str] = []

        if engine == "tesseract":
            import pytesseract
            from PIL import Image
            tcmd = os.getenv("OCR_TESSERACT_CMD", "").strip()
            if tcmd:
                pytesseract.pytesseract.tesseract_cmd = tcmd
            lang = os.getenv("OCR_LANGS", "kor+eng")
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                t = pytesseract.image_to_string(img, lang=lang).strip()
                if t:
                    texts.append(t)
        else:
            import easyocr
            langs = [s.strip() for s in os.getenv("OCR_LANG", "ko,en").replace("+", ",").split(",")]
            gpu = os.getenv("OCR_EASYOCR_GPU", "0").strip() == "1"
            reader = easyocr.Reader(langs, gpu=gpu)
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                res = reader.readtext(img, detail=0, paragraph=True)  # list[str]
                if res:
                    texts.append("\n".join([r for r in res if r]))

        out = "\n\n".join([t for t in texts if t and t.strip()])
        return out.strip() or None
    except Exception as e:
        print(f"[OCR] try_ocr_pdf_bytes error: {e}")
        return None