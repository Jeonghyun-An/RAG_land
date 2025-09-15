# app/services/pdf_fusion.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import os
from io import BytesIO

import fitz
import numpy as np

# easyocr 언어코드 보정 함수는 ocr_service에 이미 있으니 재사용
try:
    from app.services.ocr_service import _norm_easyocr_langs
except Exception:
    def _norm_easyocr_langs(lang: str) -> list[str]:
        raw = (lang or "ko,en").replace("+", ",")
        out = []
        for s in (x.strip().lower() for x in raw.split(",") if x.strip()):
            if s in ("ko","kor","korean"): out.append("ko")
            elif s in ("en","eng","english"): out.append("en")
            else: out.append(s)
        return out or ["ko","en"]

# ---------- 내부: pdfminer 로 per-page 텍스트 & 블록 ----------
def _pdfminer_pages_and_blocks_from_path(path: str) -> Tuple[List[Tuple[int,str]], Dict[int, List[Dict]]]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LAParams
    laparams = LAParams()

    pages: List[Tuple[int,str]] = []
    layout_map: Dict[int, List[Dict]] = {}

    for i, layout in enumerate(extract_pages(path, laparams=laparams), start=1):
        texts = []
        blocks: List[Dict] = []
        for elem in layout:
            if isinstance(elem, LTTextContainer):
                t = (elem.get_text() or "").strip()
                if t:
                    texts.append(t)
                    x0, y0, x1, y1 = elem.bbox
                    blocks.append({"text": t, "bbox": [float(x0), float(y0), float(x1), float(y1)]})
        pages.append((i, "\n".join(texts).strip()))
        layout_map[i] = blocks
    return pages, layout_map

def _pdfminer_pages_and_blocks_from_bytes(pdf_bytes: bytes) -> Tuple[List[Tuple[int,str]], Dict[int, List[Dict]]]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LAParams
    laparams = LAParams()

    pages: List[Tuple[int,str]] = []
    layout_map: Dict[int, List[Dict]] = {}

    bio = BytesIO(pdf_bytes)
    for i, layout in enumerate(extract_pages(bio, laparams=laparams), start=1):
        texts = []
        blocks: List[Dict] = []
        for elem in layout:
            if isinstance(elem, LTTextContainer):
                t = (elem.get_text() or "").strip()
                if t:
                    texts.append(t)
                    x0, y0, x1, y1 = elem.bbox
                    blocks.append({"text": t, "bbox": [float(x0), float(y0), float(x1), float(y1)]})
        pages.append((i, "\n".join(texts).strip()))
        layout_map[i] = blocks
    return pages, layout_map

# ---------- 내부: 특정 페이지 이미지를 OCR로 처리 ----------
def _ocr_page_with_easyocr(img_nd: "np.ndarray", lang: str, gpu: bool) -> Tuple[str, List[Dict]]:
    import easyocr, numpy as np
    langs = _norm_easyocr_langs(lang)
    reader = easyocr.Reader(langs, gpu=gpu)
    # detail=1 로 박스,텍스트,conf 받기 → 라인 텍스트와 박스 만들기
    res = reader.readtext(img_nd, detail=1, paragraph=False)
    texts = []
    blocks: List[Dict] = []
    for item in res:
        if not item or len(item) < 2:
            continue
        # item: [poly_pts, text, conf]
        pts, txt = item[0], (item[1] or "").strip()
        if not txt:
            continue
        try:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, y0, x1, y1 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
            blocks.append({"text": txt, "bbox": [x0, y0, x1, y1]})
        except Exception:
            pass
        texts.append(txt)
    # 대략 읽는 순서대로 join
    return ("\n".join(texts).strip(), blocks)

def _ocr_page_with_tesseract(img_nd: "np.ndarray", lang: str) -> Tuple[str, List[Dict]]:
    import pytesseract
    from PIL import Image
    import numpy as np
    tcmd = os.getenv("OCR_TESSERACT_CMD", "").strip()
    if tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd
    pil = Image.fromarray(img_nd)
    # 전체 텍스트
    text = (pytesseract.image_to_string(pil, lang=(lang or "kor+eng")) or "").strip()
    # 단어 단위 bbox
    try:
        data = pytesseract.image_to_data(pil, lang=(lang or "kor+eng"), output_type=pytesseract.Output.DICT)
        blocks: List[Dict] = []
        n = len(data.get("text", []))
        for i in range(n):
            wtxt = (data["text"][i] or "").strip()
            if not wtxt:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            blocks.append({"text": wtxt, "bbox": [float(x), float(y), float(x + w), float(y + h)]})
    except Exception:
        blocks = []
    return text, blocks

# ---------- 내부: 한 페이지에 OCR 적용(이미지 렌더 포함) ----------
def _ocr_page_image(fitz_page: "fitz.Page") -> Tuple[str, List[Dict]]:
    import fitz, numpy as np
    dpi = int(os.getenv("OCR_DPI", "300"))
    zoom = max(1.0, dpi / 72.0)
    mat  = fitz.Matrix(zoom, zoom)
    pix  = fitz_page.get_pixmap(matrix=mat, alpha=False)
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    engine = os.getenv("OCR_ENGINE", "easyocr").strip().lower()
    if engine == "tesseract":
        lang = os.getenv("OCR_LANG", "kor+eng")
        return _ocr_page_with_tesseract(img, lang)
    else:
        lang = os.getenv("OCR_LANG", "ko,en")
        gpu  = os.getenv("OCR_EASYOCR_GPU", "0").strip() == "1"
        return _ocr_page_with_easyocr(img, lang, gpu)

# ---------- 공개: 경로/바이트 입력을 OCR-융합으로 뽑기 ----------
def extract_pdf_fused(path: str) -> Tuple[List[Tuple[int,str]], Dict[int, List[Dict]]]:
    """
    반환:
      pages: [(page_no, text)]
      layout_map: {page_no: [ {"text":..., "bbox":[x0,y0,x1,y1]}, ... ] }
    규칙:
      - OCR_MODE=off → pdfminer 결과 그대로
      - OCR_MODE=force → 모든 페이지를 OCR로 대체(텍스트/박스)
      - OCR_MODE=auto → 페이지 텍스트 길이가 OCR_MIN_CHARS_PER_PAGE 미만이면 해당 페이지만 OCR로 교체
    """
    import fitz
    pages, layout_map = _pdfminer_pages_and_blocks_from_path(path)

    ocr_mode = os.getenv("OCR_MODE", "auto").strip().lower()  # off|auto|force
    min_chars = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", "50"))

    if ocr_mode == "off":
        return pages, layout_map

    doc = fitz.open(path)
    try:
        for (idx, _), pg in zip(pages, doc, strict=False):
            need_ocr = (ocr_mode == "force") or (len((pages[idx-1][1] or "").strip()) < min_chars)
            if need_ocr:
                txt, blocks = _ocr_page_image(pg)
                # 교체
                pages[idx-1] = (idx, txt or "")
                layout_map[idx] = blocks or []
    finally:
        doc.close()
    return pages, layout_map

def extract_pdf_fused_from_bytes(pdf_bytes: bytes) -> Tuple[List[Tuple[int,str]], Dict[int, List[Dict]]]:
    import fitz
    pages, layout_map = _pdfminer_pages_and_blocks_from_bytes(pdf_bytes)

    ocr_mode = os.getenv("OCR_MODE", "auto").strip().lower()
    min_chars = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", "50"))

    if ocr_mode == "off":
        return pages, layout_map

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for i, pg in enumerate(doc, start=1):
            cur_txt = (pages[i-1][1] or "") if i-1 < len(pages) else ""
            need_ocr = (ocr_mode == "force") or (len(cur_txt.strip()) < min_chars)
            if need_ocr:
                txt, blocks = _ocr_page_image(pg)
                if i-1 < len(pages):
                    pages[i-1] = (i, txt or "")
                else:
                    pages.append((i, txt or ""))
                layout_map[i] = blocks or []
    finally:
        doc.close()
    return pages, layout_map
