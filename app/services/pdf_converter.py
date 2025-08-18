from __future__ import annotations
import os, io, tempfile, requests
from typing import Optional
from pathlib import Path

GOTENBERG_URL = os.getenv("GOTENBERG_URL", "http://gotenberg:3000")

OFFICE_EXT = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
              ".odt", ".odp", ".ods", ".rtf"}
HTML_EXT   = {".html", ".htm"}
IMG_EXT    = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
TXT_EXT    = {".txt", ".csv", ".md"}

class ConvertError(RuntimeError): ...
def _ensure_parent(p: Path): p.parent.mkdir(parents=True, exist_ok=True)

def convert_to_pdf(src_path: str) -> str:
    """입력 파일을 PDF로 변환해서 로컬 경로 반환. 이미 PDF면 그대로 반환."""
    src = Path(src_path)
    ext = src.suffix.lower()
    if ext == ".pdf":
        return str(src)

    out = src.with_suffix(".pdf")  # 같은 위치에 저장
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

def _libreoffice_to_pdf(src: Path, out: Path):
    # Gotenberg LibreOffice 변환
    url = f"{GOTENBERG_URL}/forms/libreoffice/convert"
    with open(src, "rb") as f:
        files = {"files": (src.name, f, "application/octet-stream")}
        r = requests.post(url, files=files, timeout=120)
    if r.status_code != 200:
        raise ConvertError(f"LibreOffice 변환 실패: {r.status_code} {r.text[:200]}")
    out.write_bytes(r.content)

def _html_to_pdf(src: Path, out: Path):
    # Chromium로 HTML→PDF
    url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
    with open(src, "rb") as f:
        files = {"files": ("index.html", f, "text/html")}
        r = requests.post(url, files=files, timeout=120)
    if r.status_code != 200:
        raise ConvertError(f"HTML 변환 실패: {r.status_code} {r.text[:200]}")
    out.write_bytes(r.content)

def _text_to_pdf(src: Path, out: Path):
    # TXT/CSV/MD는 간단히 HTML 래핑 후 Chromium 사용
    content = src.read_text(encoding="utf-8", errors="ignore")
    html = f"""<!doctype html><meta charset="utf-8">
    <style>body{{font-family:system-ui,Segoe UI,Apple SD Gothic Neo,Malgun Gothic,Arial; white-space:pre-wrap}}</style>
    <pre>{content}</pre>"""
    url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
    files = {"files": ("index.html", io.BytesIO(html.encode("utf-8")), "text/html")}
    r = requests.post(url, files=files, timeout=120)
    if r.status_code != 200:
        raise ConvertError(f"TEXT 변환 실패: {r.status_code} {r.text[:200]}")
    out.write_bytes(r.content)

def _image_to_pdf(src: Path, out: Path):
    # Pillow 기반 (단일 이미지). 멀티페이지 TIFF는 Gotenberg HTML 경로 권장.
    try:
        from PIL import Image
    except ImportError:
        # Pillow 미설치 시, Chromium로 <img> 렌더링
        _image_to_pdf_via_chromium(src, out); return
    im = Image.open(src)
    rgb = im.convert("RGB")
    rgb.save(out, "PDF")

def _image_to_pdf_via_chromium(src: Path, out: Path):
    # 이미지 1장을 HTML로 감싸서 Chromium 변환
    url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
    html = b'<!doctype html><img src="file.png" style="max-width:100%;">'
    files = [
        ("files", ("index.html", io.BytesIO(html), "text/html")),
        ("files", ("file.png", open(src, "rb"), "application/octet-stream")),
    ]
    r = requests.post(url, files=files, timeout=120)
    if r.status_code != 200:
        raise ConvertError(f"IMAGE 변환 실패: {r.status_code} {r.text[:200]}")
    out.write_bytes(r.content)
