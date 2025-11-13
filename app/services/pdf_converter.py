# app/services/pdf_converter.py
from __future__ import annotations
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    try:
        Image.ANTIALIAS = Image.LANCZOS
    except Exception:
        from PIL import Image as _I
        Image.ANTIALIAS = getattr(_I, "BICUBIC", None)
import os, io, time, requests, shutil, subprocess
from typing import Optional
from pathlib import Path

GOTENBERG_URL = os.getenv("GOTENBERG_URL", "http://gotenberg:3000")
GOTENBERG_TIMEOUT = int(os.getenv("GOTENBERG_TIMEOUT", "120"))
GOTENBERG_MAX_RETRIES = int(os.getenv("GOTENBERG_MAX_RETRIES", "3"))
GOTENBERG_BACKOFF_BASE = float(os.getenv("GOTENBERG_BACKOFF_BASE", "0.6"))
PDF_PAPER = os.getenv("PDF_PAPER", "auto")
PDF_MARGIN_MM = int(os.getenv("PDF_MARGIN_MM", "10"))
CONVERTER_ENDPOINT = os.getenv("DOC_CONVERTER_URL", "").strip()

OFFICE_EXT = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".odp", ".ods", ".rtf"}
HTML_EXT   = {".html", ".htm"}
IMG_EXT    = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
TXT_EXT    = {".txt", ".csv", ".md"}

class ConvertStreamError(Exception):
    pass

class ConvertError(RuntimeError): 
    pass

def _ensure_parent(p: Path): 
    p.parent.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def _gotenberg_ok() -> bool:
    try:
        r = requests.get(f"{GOTENBERG_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def _post_retry(url: str, files, data: Optional[dict] = None) -> bytes:
    last = None
    for i in range(GOTENBERG_MAX_RETRIES):
        try:
            r = requests.post(url, files=files, data=data or {}, timeout=GOTENBERG_TIMEOUT)
            if r.status_code == 200:
                return r.content
            last = ConvertError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last = e
        time.sleep(GOTENBERG_BACKOFF_BASE * (2 ** i))
    raise ConvertError(f"Gotenberg 요청 실패: {last!s}")

def _chromium_opts(no_margins: bool = False) -> dict:
    data = {}
    if PDF_PAPER and PDF_PAPER.lower() != "auto":
        # A4(210x297mm) -> 8.27 x 11.69 inch
        if PDF_PAPER.lower() == "a4":
            data["paperWidth"] = "8.27"
            data["paperHeight"] = "11.69"
        elif PDF_PAPER.lower() == "letter":
            data["paperWidth"] = "8.5"
            data["paperHeight"] = "11"
    if not no_margins:
        margin_in = max(0.0, float(PDF_MARGIN_MM)) / 25.4
        data.update({
            "marginTop": str(margin_in),
            "marginBottom": str(margin_in),
            "marginLeft": str(margin_in),
            "marginRight": str(margin_in),
        })
    return data

# ---------- reportlab 기반 TXT → PDF (bytes 버전) ----------
def _text_to_pdf_bytes(text: str) -> bytes:
    """
    텍스트를 reportlab로 PDF bytes로 변환
    [핵심 추가] TXT 파일 지원을 위한 bytes 기반 변환
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError as e:
        raise ConvertError(f"reportlab이 필요합니다: {e}")

    # 메모리 버퍼
    buffer = io.BytesIO()
    
    # 폰트 등록 시도 (없어도 기본 폰트로 동작)
    try:
        # 한글 폰트가 있으면 등록
        # pdfmetrics.registerFont(TTFont("NotoSans", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"))
        pass
    except Exception:
        pass

    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin_x = 20 * mm
    margin_y = 20 * mm
    max_width = width - 2 * margin_x
    y = height - margin_y

    # 폰트 설정
    try:
        c.setFont("Helvetica", 10)
    except Exception:
        pass

    # 워드랩
    import textwrap
    lines = []
    for para in (text or "").splitlines():
        wrap = textwrap.wrap(para, width=95) or [""]
        lines.extend(wrap)

    line_height = 12  # pt
    for line in lines:
        if y <= margin_y:
            c.showPage()
            try:
                c.setFont("Helvetica", 10)
            except Exception:
                pass
            y = height - margin_y
        c.drawString(margin_x, y, line)
        y -= line_height

    c.showPage()
    c.save()
    
    # bytes 반환
    buffer.seek(0)
    return buffer.read()

# ---------- Local file path 기반 변환 (기존 로직) ----------
def _libreoffice_to_pdf(src: Path, out: Path):
    if not _gotenberg_ok():
        raise ConvertError("Gotenberg 서비스 없음")
    url = f"{GOTENBERG_URL}/forms/libreoffice/convert"
    with open(src, "rb") as f:
        files = {"files": (src.name, f, "application/octet-stream")}
        pdf_bytes = _post_retry(url, files)
    with open(out, "wb") as fw:
        fw.write(pdf_bytes)

def _html_to_pdf(src: Path, out: Path):
    if not _gotenberg_ok():
        raise ConvertError("Gotenberg 서비스 없음")
    url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
    with open(src, "rb") as f:
        files = [("files", ("index.html", f, "text/html; charset=utf-8"))]
        pdf_bytes = _post_retry(url, files, data=_chromium_opts())
    with open(out, "wb") as fw:
        fw.write(pdf_bytes)

def _image_to_pdf(src: Path, out: Path):
    if not _gotenberg_ok():
        raise ConvertError("Gotenberg 서비스 없음")
    url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
    html = (b'<!doctype html><meta charset="utf-8">'
            b'<style>html,body{margin:0;padding:0}img{width:100%;height:auto}</style>'
            b'<img src="file.bin">')
    with open(src, "rb") as f:
        files = [
            ("files", ("index.html", io.BytesIO(html), "text/html; charset=utf-8")),
            ("files", ("file.bin", f, "application/octet-stream")),
        ]
        pdf_bytes = _post_retry(url, files, data=_chromium_opts(no_margins=True))
    with open(out, "wb") as fw:
        fw.write(pdf_bytes)

def _text_to_pdf(src: Path, out: Path):
    """로컬 파일 기반 TXT → PDF 변환"""
    with open(src, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    pdf_bytes = _text_to_pdf_bytes(text)
    
    with open(out, "wb") as fw:
        fw.write(pdf_bytes)

# ---------- public: 로컬 파일 경로 기반 변환 ----------
def convert_to_pdf(src_path: str) -> str:
    """입력 파일을 PDF로 변환해서 로컬 경로 반환. 이미 PDF면 그대로 반환."""
    src = Path(src_path)
    ext = src.suffix.lower()
    if ext == ".pdf":
        return str(src)

    out = src.with_suffix(".pdf")
    _ensure_parent(out)

    if ext in OFFICE_EXT:
        _libreoffice_to_pdf(src, out)
    elif ext in HTML_EXT:
        _html_to_pdf(src, out)
    elif ext in IMG_EXT:
        _image_to_pdf(src, out)
    elif ext in TXT_EXT:
        _text_to_pdf(src, out)
    else:
        raise ConvertError(f"지원하지 않는 파일 유형: {ext}")

    if not out.exists() or out.stat().st_size == 0:
        raise ConvertError("변환된 PDF가 비어있습니다.")
    return str(out)

# ---------- public: bytes 기반 변환 (java_router용) ----------
def convert_bytes_to_pdf_bytes(content: bytes, src_ext: str) -> bytes | None:
    """
    bytes를 PDF bytes로 변환
    [핵심 수정] TXT 파일 지원 추가
    
    Returns:
        PDF bytes if success, None if unsupported
    """
    ext = (src_ext or "").lower()

    # 이미 PDF면 그대로
    if ext == ".pdf":
        return content

    # 1) TXT 류 → reportlab 변환 [핵심 추가]
    if ext in TXT_EXT:
        try:
            text = content.decode("utf-8", errors="ignore")
            return _text_to_pdf_bytes(text)
        except Exception as e:
            print(f"[CONVERT] TXT→PDF 변환 실패: {e}")
            return None

    # 2) Office 류 → LibreOffice 변환
    if ext in OFFICE_EXT:
        if not _gotenberg_ok():
            return None
        url = f"{GOTENBERG_URL}/forms/libreoffice/convert"
        files = {"files": (f"upload{ext}", io.BytesIO(content), "application/octet-stream")}
        try:
            return _post_retry(url, files)
        except Exception:
            return None

    # 3) HTML → Chromium 변환
    if ext in HTML_EXT:
        if not _gotenberg_ok():
            return None
        url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
        files = [("files", ("index.html", io.BytesIO(content), "text/html; charset=utf-8"))]
        try:
            return _post_retry(url, files, data=_chromium_opts())
        except Exception:
            return None

    # 4) 단일 이미지 → HTML로 감싸 Chromium 변환
    if ext in IMG_EXT:
        if not _gotenberg_ok():
            return None
        url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
        html = (b'<!doctype html><meta charset="utf-8">'
                b'<style>html,body{margin:0;padding:0}img{width:100%;height:auto}</style>'
                b'<img src="file.bin">')
        files = [
            ("files", ("index.html", io.BytesIO(html), "text/html; charset=utf-8")),
            ("files", ("file.bin", io.BytesIO(content), "application/octet-stream")),
        ]
        try:
            return _post_retry(url, files, data=_chromium_opts(no_margins=True))
        except Exception:
            return None

    # 지원하지 않는 확장자
    return None

def convert_stream_to_pdf_bytes(content: bytes, src_ext: str) -> Optional[bytes]:
    """
    외부 변환기(예: ONLYOFFICE, 사내 컨버터)로 bytes를 보내 PDF bytes로 받는다.
    - env DOC_CONVERTER_URL 필요 (POST multipart/form-data)
    - 실패/미설정 시 None 반환(상위에서 폴백)
    """
    if not CONVERTER_ENDPOINT:
        return None
    try:
        files = {"file": (f"upload{src_ext}", content)}
        data = {"target": "pdf"}
        r = requests.post(CONVERTER_ENDPOINT, files=files, data=data, timeout=120)
        r.raise_for_status()
        # 변환기가 application/pdf 바이너리를 바로 반환한다고 가정
        return r.content
    except Exception as e:
        raise ConvertStreamError(f"stream->pdf 변환 실패: {e}")