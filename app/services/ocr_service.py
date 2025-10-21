# app/services/ocr_service.py
from __future__ import annotations
import math
import re
import os, subprocess
from pathlib import Path
from typing import Tuple, Dict
from io import BytesIO
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image
import easyocr

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
        dpi = int(os.getenv("OCR_DPI", "300"))
        zoom = max(1.0, dpi / 72.0)
        mat = fitz.Matrix(zoom, zoom)

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        engine = os.getenv("OCR_ENGINE", "easyocr").strip().lower()

        texts: list[str] = []

        if engine == "tesseract":
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
    
from collections import Counter

def _bbox_angle_deg(box):
    # EasyOCR box: 4 points [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    (x1,y1),(x2,y2),(x3,y3),(x4,y4) = box
    dx, dy = x2-x1, y2-y1
    return abs(math.degrees(math.atan2(dy, dx)))

def _bbox_area(box):
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    return (max(xs)-min(xs)) * (max(ys)-min(ys))

def _normalize_watermark_text(s):
    t = re.sub(r"\s+", "", s or "")
    t = re.sub(r"[^\w가-힣]", "", t)
    return t.lower()

def filter_watermarks(easyocr_results_by_page, page_w, page_h):
    """
    easyocr_results_by_page: list[ list[(box, text, conf)] ]
    반환: 동일 구조, 워터마크 후보 제거됨
    """
    # 1) 후보 수집: 각도 20~70°, 큰 박스(페이지 면적 대비 10%+)
    cand = []
    for pno, items in enumerate(easyocr_results_by_page, start=1):
        for box, text, conf in items:
            ang = _bbox_angle_deg(box)
            if 20 <= ang <= 70:
                area = _bbox_area(box)
                if area >= 0.10 * (page_w * page_h):
                    cand.append((_normalize_watermark_text(text), pno))

    # 2) 전역 반복 텍스트만 워터마크로 확정(최소 3페이지 이상)
    counts = Counter([t for t,_ in cand])
    wm_texts = {t for t,c in counts.items() if c >= 3 and len(t) >= 4}

    # 3) 제거
    filtered = []
    for items in easyocr_results_by_page:
        kept = []
        for box, text, conf in items:
            tnorm = _normalize_watermark_text(text)
            ang = _bbox_angle_deg(box)
            area = _bbox_area(box)
            if (tnorm in wm_texts) and (20 <= ang <= 70) and (area >= 0.08 * (page_w * page_h)):
                continue  # drop watermark
            kept.append((box, text, conf))
        filtered.append(kept)
    return filtered
