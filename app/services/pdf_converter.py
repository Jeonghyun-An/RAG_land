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

class ConvertError(RuntimeError): ...
def _ensure_parent(p: Path): p.parent.mkdir(parents=True, exist_ok=True)

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

# ---------- public ----------
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

import io

# Gotenberg로 bytes를 직접 보내서 PDF bytes를 얻는다.
# - 성공: PDF bytes 반환
# - 미지원 확장자/실패: None
def convert_bytes_to_pdf_bytes(content: bytes, src_ext: str) -> bytes | None:
    ext = (src_ext or "").lower()

    # 이미 PDF면 그대로
    if ext == ".pdf":
        return content

    # 1) Office 류 → LibreOffice 변환
    if ext in {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".odp", ".ods", ".rtf"}:
        if not _gotenberg_ok():
            return None
        url = f"{GOTENBERG_URL}/forms/libreoffice/convert"
        files = {"files": (f"upload{ext}", io.BytesIO(content), "application/octet-stream")}
        try:
            return _post_retry(url, files)
        except Exception:
            return None

    # 2) HTML → Chromium 변환
    if ext in {".html", ".htm"}:
        if not _gotenberg_ok():
            return None
        url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
        files = [("files", ("index.html", io.BytesIO(content), "text/html; charset=utf-8"))]
        try:
            return _post_retry(url, files, data=_chromium_opts())
        except Exception:
            return None

    # 3) 단일 이미지 → 간단한 HTML로 감싸 Chromium 변환
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}:
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

    # 4) 미지원 (예: .hwpx, .hwp 등) → None
    return None

# ---------- converters ----------
def _libreoffice_to_pdf(src: Path, out: Path):
    if not _gotenberg_ok():
        raise ConvertError("Gotenberg healthcheck 실패")
    url = f"{GOTENBERG_URL}/forms/libreoffice/convert"
    with open(src, "rb") as f:
        files = {"files": (src.name, f, "application/octet-stream")}
        pdf = _post_retry(url, files)
    out.write_bytes(pdf)

def _html_to_pdf(src: Path, out: Path):
    if not _gotenberg_ok():
        raise ConvertError("Gotenberg healthcheck 실패")
    url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
    with open(src, "rb") as f:
        files = {"files": ("index.html", f, "text/html; charset=utf-8")}
        data = _chromium_opts()
        pdf = _post_retry(url, files, data=data)
    out.write_bytes(pdf)

def _text_to_pdf(src: Path, out: Path):
    # TXT/CSV/MD를 간단히 HTML로 감싸서 Chromium 경로 사용(폰트/여백 안정화)
    content = src.read_text(encoding="utf-8", errors="ignore")
    style = f"""
    <style>
      @page {{ size: {PDF_PAPER if PDF_PAPER!='auto' else 'A4'}; margin: {PDF_MARGIN_MM}mm; }}
      body {{ font-family: system-ui, 'Segoe UI', 'Apple SD Gothic Neo', 'Malgun Gothic', Arial, sans-serif;
              white-space: pre-wrap; word-break: break-word; }}
      pre {{ white-space: pre-wrap; }}
      table {{ border-collapse: collapse; width: 100%; }}
      td, th {{ border: 1px solid #ddd; padding: 4px; }}
    </style>"""
    html = f"<!doctype html><meta charset='utf-8'>{style}<pre>{content}</pre>"
    if not _gotenberg_ok():
        raise ConvertError("Gotenberg healthcheck 실패")
    url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
    files = {"files": ("index.html", io.BytesIO(html.encode("utf-8")), "text/html; charset=utf-8")}
    data = _chromium_opts()
    pdf = _post_retry(url, files, data=data)
    out.write_bytes(pdf)

def _image_to_pdf(src: Path, out: Path):
    """이미지 → PDF 변환 (Pillow 10.0+ 호환)"""
    try:
        from PIL import Image, ImageSequence
        im = Image.open(src)
        
        # Pillow 10.0+ 호환: LANCZOS 사용 (ANTIALIAS 제거됨)
        frames = []
        for frame in ImageSequence.Iterator(im):
            rgb_frame = frame.convert("RGB")
            # 리샘플링이 필요한 경우 LANCZOS 명시
            frames.append(rgb_frame)
        
        if not frames:
            frames = [im.convert("RGB")]
        
        if len(frames) == 1:
            frames[0].save(out, "PDF", resolution=100.0)
        else:
            frames[0].save(out, "PDF", save_all=True, append_images=frames[1:], resolution=100.0)
        return
    except Exception as e:
        print(f"[PDF_CONVERTER] Pillow 변환 실패, Chromium 폴백: {e}")
        pass
    _image_to_pdf_via_chromium(src, out)
    
def _image_to_pdf_via_chromium(src: Path, out: Path):
    if not _gotenberg_ok():
        raise ConvertError("Gotenberg healthcheck 실패")
    url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
    html = (b'<!doctype html><meta charset="utf-8">'
            b'<style>html,body{margin:0;padding:0}img{width:100%;height:auto}</style>'
            b'<img src="file.bin">')
    files = [
        ("files", ("index.html", io.BytesIO(html), "text/html; charset=utf-8")),
        ("files", ("file.bin", open(src, "rb"), "application/octet-stream")),
    ]
    data = _chromium_opts(no_margins=True)
    pdf = _post_retry(url, files, data=data)
    out.write_bytes(pdf)
