from __future__ import annotations
from typing import Dict, List, Tuple, Union, Optional
import os
from io import BytesIO

"""
PDF 텍스트 추출
- OCR 모드: OCR_MODE in {"auto","always","never"} (default=auto)
- OCR 엔진: OCR_ENGINE in {"paddle","tesseract","easyocr"} (default=paddle)
- 언어:
    * paddle : "korean" 또는 "en", "ch" 등
    * tesseract : "kor+eng" 같은 조합
    * easyocr : "ko,en" 같은 콤마구분
- 렌더링 DPI: OCR_DPI (default=300)
- 추가 옵션:
    * OCR_TESSERACT_CMD: pytesseract 실행 파일 경로(Windows 등)
    * OCR_EASYOCR_GPU: "1"이면 GPU 사용 시도, 기본 "0"
    * OCR_MIN_CHARS: auto 모드에서 OCR 전환 기준(기본 40)
    이 파일은 “문서를 읽어서 페이지 텍스트와 레이아웃 블록(BBox) 을 만들어 indexer/청커에 넘겨주는” 역할
    임베딩 텍스트를 어떻게 구성할지는(섹션 합치기 등) 라우터/청커 단계에서 결정
"""

# ---------------------- Text extract (no OCR) ---------------------- #
def _extract_text_pdfminer(path: str, by_page: bool) -> Union[str, List[Tuple[int, str]]]:
    import pdfminer.high_level
    from pdfminer.layout import LAParams
    laparams = LAParams()
    if not by_page:
        return (pdfminer.high_level.extract_text(path, laparams=laparams) or "").strip()
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(pdfminer.high_level.extract_pages(path, laparams=laparams), start=1):
        text = "".join([elem.get_text() for elem in page if hasattr(elem, "get_text")]).strip()
        if text:
            pages.append((i, text))
    return pages

def _extract_text_pypdf2(path: str, by_page: bool) -> Union[str, List[Tuple[int, str]]]:
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    if not by_page:
        txt = "\n\n".join([(p.extract_text() or "") for p in reader.pages]).strip()
        return txt
    pages: List[Tuple[int, str]] = []
    for i, p in enumerate(reader.pages, start=1):
        t = (p.extract_text() or "").strip()
        if t:
            pages.append((i, t))
    return pages

# ---------------------- Common rendering (PyMuPDF) ---------------------- #
def _render_pdf_pages_fitz(path: str, dpi: int):
    """Yields (page_index_1based, np.ndarray[h,w,3] RGB)"""
    import fitz  # PyMuPDF
    import numpy as np

    zoom = max(1.0, dpi / 72.0)  # 72dpi 기준
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(path)

    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        yield i, img

# ---------------------- OCR engines (via fitz images) ---------------------- #
def _ocr_with_paddle(images_iter, by_page: bool, lang: str) -> Union[str, List[Tuple[int, str]]]:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang=("korean" if lang in ("kor", "korean", "ko") else lang), show_log=False)
    pages: List[Tuple[int, str]] = []
    for pno, img in images_iter:
        result = ocr.ocr(img, cls=True)
        # result[0] = [ [box, (text, conf)], ... ]
        text = " ".join([line[1][0] for line in (result[0] or []) if line[1] and line[1][0]]).strip()
        if text:
            pages.append((pno, text))
    if by_page:
        return pages
    return "\n\n".join([t for _, t in pages]).strip()

def _ocr_with_tesseract(images_iter, by_page: bool, lang: str) -> Union[str, List[Tuple[int, str]]]:
    import pytesseract
    from PIL import Image
    # Windows 등 경로 지정이 필요할 때:
    tcmd = os.getenv("OCR_TESSERACT_CMD")
    if tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd

    pages: List[Tuple[int, str]] = []
    for pno, img in images_iter:
        pil = Image.fromarray(img)
        text = pytesseract.image_to_string(pil, lang=(lang or "kor+eng")).strip()
        if text:
            pages.append((pno, text))
    if by_page:
        return pages
    return "\n\n".join([t for _, t in pages]).strip()

def _ocr_with_easyocr(images_iter, by_page: bool, lang: str) -> Union[str, List[Tuple[int, str]]]:
    import easyocr
    # easyocr 언어코드: ['ko','en'] 형태
    langs = [l.strip() for l in (lang or "ko,en").replace("+", ",").split(",")]
    gpu = os.getenv("OCR_EASYOCR_GPU", "0").strip() == "1"
    reader = easyocr.Reader(langs, gpu=gpu)

    pages: List[Tuple[int, str]] = []
    for pno, img in images_iter:
        res = reader.readtext(img)  # [ [box, text, conf], ... ]
        text = " ".join([x[1] for x in res if len(x) >= 2 and x[1]]).strip()
        if text:
            pages.append((pno, text))
    if by_page:
        return pages
    return "\n\n".join([t for _, t in pages]).strip()

# ---------------------- Heuristic for OCR fallback ---------------------- #
def _should_ocr(txt_or_pages: Union[str, List[Tuple[int, str]]]) -> bool:
    """
    텍스트 밀도 낮으면 OCR로 전환 (auto 모드에서만 사용)
    - 문자열: 길이 < OCR_MIN_CHARS (기본 40)
    - by_page 결과: 모든 페이지 텍스트 길이 합 < OCR_MIN_CHARS
    """
    try:
        th = int(os.getenv("OCR_MIN_CHARS", "40"))
    except Exception:
        th = 40
    if isinstance(txt_or_pages, str):
        return len(txt_or_pages.strip()) < th
    total = sum(len(t or "") for _, t in (txt_or_pages or []))
    return total < th

# ---------------------- Public API ---------------------- #
# 추가: Word/Excel 파서
def parse_docx_sections(path: str) -> List[Tuple[int, str]]:
    """
    DOCX를 '섹션 단위'로 반환: (section_no, text)
    - 제목(Heading 1~3) 경계로 묶고, 표는 행단위 문자열로 합쳐 섹션 뒤에 덧붙임
    """
    from docx import Document
    doc = Document(path)
    sections: List[Tuple[int, str]] = []
    cur = []
    sec_no = 0

    def flush():
        nonlocal cur, sec_no
        if cur:
            sec_no += 1
            sections.append((sec_no, "\n".join(cur).strip()))
            cur = []

    # 본문/제목
    for p in doc.paragraphs:
        style = (p.style.name or "").lower() if p.style else ""
        line = p.text.strip()
        if not line:
            continue
        if "heading" in style:  # 새 섹션
            flush()
            cur.append(line)
        else:
            cur.append(line)

    # 표
    for t in doc.tables:
        lines = []
        for r in t.rows:
            cells = [c.text.strip().replace("\n", " ") for c in r.cells]
            lines.append(" | ".join(cells))
        if lines:
            cur.append("[표]\n" + "\n".join(lines))

    flush()
    return sections  # [(1,"..."), (2,"...")]

def parse_xlsx_tables(path: str) -> List[Tuple[int, str]]:
    """
    XLSX를 시트 단위로 반환: (sheet_index_1based, sheet_text)
    - 첫 행을 헤더로 보고 "열:값" 형태로 행을 직렬화
    """
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    out: List[Tuple[int,str]] = []
    for si, ws in enumerate(wb.worksheets, start=1):
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        headers = [str(h or "").strip() for h in rows[0]]
        lines = [f"[시트] {ws.title}"]
        for r in rows[1:]:
            pairs = []
            for h, v in zip(headers, r):
                if h:
                    pairs.append(f"{h}: {'' if v is None else str(v)}")
            if pairs:
                lines.append(", ".join(pairs))
        txt = "\n".join(lines).strip()
        if txt:
            out.append((si, txt))
    return out

def parse_pdf(path: str, by_page: bool = False) -> Union[str, List[Tuple[int, str]]]:
    ocr_mode   = os.getenv("OCR_MODE", "auto").lower()          # auto | always | never
    ocr_engine = os.getenv("OCR_ENGINE", "paddle").lower()      # paddle | tesseract | easyocr
    ocr_dpi    = int(os.getenv("OCR_DPI", "300"))

    # 언어 기본값
    if ocr_engine == "tesseract":
        ocr_lang = os.getenv("OCR_LANG", "kor+eng")
    elif ocr_engine == "easyocr":
        ocr_lang = os.getenv("OCR_LANG", "ko,en")
    else:
        ocr_lang = os.getenv("OCR_LANG", "korean")

    # 1) 텍스트 파싱 (auto/never)
    text_result: Optional[Union[str, List[Tuple[int, str]]]] = None
    if ocr_mode in ("auto", "never"):
        try:
            text_result = _extract_text_pdfminer(path, by_page)
        except Exception:
            try:
                text_result = _extract_text_pypdf2(path, by_page)
            except Exception:
                text_result = "" if not by_page else []
        if ocr_mode == "never":
            if text_result is None:
                raise RuntimeError("PDF 텍스트 추출 실패(never 모드): 내부 파서 실패")
            return text_result
        if text_result and not _should_ocr(text_result):
            return text_result

    # 2) OCR (PyMuPDF로 렌더 → 선택한 엔진)
    try:
        images_iter = _render_pdf_pages_fitz(path, ocr_dpi)
        if ocr_engine == "tesseract":
            return _ocr_with_tesseract(images_iter, by_page, ocr_lang)
        elif ocr_engine == "easyocr":
            return _ocr_with_easyocr(images_iter, by_page, ocr_lang)
        else:
            return _ocr_with_paddle(images_iter, by_page, ocr_lang)
    except Exception as e:
        # OCR 실패 시: 텍스트 결과라도 반환
        if text_result and ((isinstance(text_result, str) and text_result.strip()) or (isinstance(text_result, list) and len(text_result) > 0)):
            return text_result
        raise RuntimeError(f"OCR 실패 및 텍스트 추출 실패: {e}")


def parse_pdf_blocks(path: str) -> list[tuple[int, list[dict]]]:
    """
    PyMuPDF 기반: 각 페이지의 텍스트 블록을 [ {text, bbox:[x0,y0,x1,y1]} ] 형태로 반환.
    반환: [(page_no, blocks), ...]
    """
    import fitz  # PyMuPDF
    out = []
    doc = fitz.open(path)
    for i, page in enumerate(doc, start=1):
        blocks = []
        # get_text("blocks"): (x0, y0, x1, y1, text, block_no, ...)
        for b in page.get_text("blocks"):
            if not b or len(b) < 5:
                continue
            x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], (b[4] or "").strip()
            if txt:
                blocks.append({
                    "text": txt,
                    "bbox": [float(x0), float(y0), float(x1), float(y1)]
                })
        out.append((i, blocks))
    return out

    
def parse_any(path: str) -> List[Tuple[int, str]]:
    """
    파일 확장자에 따라:
      - PDF: parse_pdf(by_page=True)
      - DOCX: parse_docx_sections()
      - XLSX/CSV: parse_xlsx_tables()/CSV 파싱
      - 기타: (옵션) PDF 변환 → parse_pdf
    반드시 [(index:int, text:str)] 형태로 반환
    """
    ext = (os.path.splitext(path)[1] or "").lower()
    direct_docx = os.getenv("RAG_PARSE_DIRECT_DOCX", "1") == "1"
    direct_xlsx = os.getenv("RAG_PARSE_DIRECT_XLSX", "1") == "1"
    allow_convert = os.getenv("RAG_CONVERT_NONPDF_TO_PDF", "1") == "1"

    if ext == ".pdf":
        return parse_pdf(path, by_page=True)

    if ext in (".docx",) and direct_docx:
        return parse_docx_sections(path)

    if ext in (".xlsx", ".xlsm", ".csv") and direct_xlsx:
        if ext == ".csv":
            # 간단 CSV → 시트1로 취급
            import csv
            lines = []
            with open(path, "r", encoding="utf-8", errors="ignore",newline="") as f:
                rdr = csv.reader(f)
                for row in rdr:
                    lines.append(" | ".join([c.strip() for c in row]))
            return [(1, "\n".join(lines))]
        return parse_xlsx_tables(path)

    # 최후: PDF로 변환 (이미 라우터에 convert_to_pdf가 있으면 그걸 사용)
    if allow_convert:
        from app.services.pdf_converter import convert_to_pdf
        pdf_path = convert_to_pdf(path)
        return parse_pdf(pdf_path, by_page=True)

    # 변환 불가 시 실패 처리
    raise RuntimeError(f"Unsupported file type without conversion: {ext}")

def sniff_ext_from_name(name: str) -> str:
    return (os.path.splitext(name)[1] or "").lower()

def parse_pdf_pages_from_bytes(pdf_bytes: bytes) -> List[str]:
    """
    pdfminer.six 기반으로 bytes에서 페이지별 텍스트 추출.
    기존 parse_pdf(file_path, by_page=True)와 호환되는 리스트를 반환.
    """
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer
    bio = BytesIO(pdf_bytes)
    pages: List[str] = []
    for layout in extract_pages(bio):
        lines = []
        for elem in layout:
            if isinstance(elem, LTTextContainer):
                lines.append(elem.get_text())
        pages.append(("".join(lines)).strip())
    return pages

# --- PDF bytes → (page_no, blocks[{text,bbox}]) ---
def parse_pdf_blocks_from_bytes(pdf_bytes: bytes) -> List[Tuple[int, List[Dict]]]:
    """
    PDF bytes -> [(page_no, [ { "text": str, "bbox": [x0,y0,x1,y1] } ])]
    - 좌표계: pdfminer 기본(user space), 원점 좌하단
    """
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    out: List[Tuple[int, List[Dict]]] = []
    bio = BytesIO(pdf_bytes)
    for i, layout in enumerate(extract_pages(bio), start=1):
        blocks: List[Dict] = []
        for elem in layout:
            if isinstance(elem, LTTextContainer):
                txt = (elem.get_text() or "").strip()
                if not txt:
                    continue
                x0, y0, x1, y1 = elem.bbox
                blocks.append({"text": txt, "bbox": [float(x0), float(y0), float(x1), float(y1)]})
        out.append((i, blocks))
    return out

# --- DOCX bytes → 라인 항목 ---
def parse_docx_from_bytes(docx_bytes: bytes) -> List[Dict]:
    try:
        from docx import Document
        doc = Document(BytesIO(docx_bytes))
        out: List[Dict] = []
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if t:
                out.append({"text": t})
        return out
    except Exception:
        # 폴백: 평문
        return [{"text": docx_bytes.decode("utf-8", "ignore")}]

# --- 평문 bytes ---
def parse_plaintext_bytes(content: bytes) -> List[Dict]:
    t = (content.decode("utf-8", "ignore") or "").strip()
    return [{"text": t}] if t else []

# --- 통합 진입점: 파일명 힌트 + bytes ---
def parse_any_bytes(name_hint: str, content: bytes) -> Dict[str, object]:
    """
    반환 예:
      {
        "kind": "pdf|docx|plain",
        "ext": ".pdf",
        "pages": [...],                  # PDF면 문자열 리스트
        "blocks": [(page_no, blocks)],   # PDF면 레이아웃 블록
        "items": [{"text":..}, ...],     # DOCX/PLAIN 등 라인 단위
      }
    """
    ext = sniff_ext_from_name(name_hint)

    if ext == ".pdf":
        pages = parse_pdf_pages_from_bytes(content)
        blocks = parse_pdf_blocks_from_bytes(content)
        return {"kind": "pdf", "ext": ext, "pages": pages, "blocks": blocks}

    if ext == ".docx":
        items = parse_docx_from_bytes(content)
        return {"kind": "docx", "ext": ext, "items": items}

    if ext in (".hwpx", ".hwp"):
        # 변환기를 이용해 PDF bytes 만들기 → PDF 파이프 재사용
        from app.services.pdf_converter import convert_stream_to_pdf_bytes
        pdf_bytes = convert_stream_to_pdf_bytes(content, ext)
        if pdf_bytes:
            pages = parse_pdf_pages_from_bytes(pdf_bytes)
            blocks = parse_pdf_blocks_from_bytes(pdf_bytes)
            return {"kind": "pdf", "ext": ".pdf", "pages": pages, "blocks": blocks}
        # 폴백: 평문
        return {"kind": "plain", "ext": ext, "items": parse_plaintext_bytes(content)}

    # 기타 확장자는 평문 취급
    return {"kind": "plain", "ext": ext, "items": parse_plaintext_bytes(content)}