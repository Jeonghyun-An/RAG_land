# app/services/pdf_converter.py - COMPLETE VERSION
from __future__ import annotations
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    try:
        Image.ANTIALIAS = Image.LANCZOS
    except Exception:
        from PIL import Image as _I
        Image.ANTIALIAS = getattr(_I, "BICUBIC", None)

import os, io, time, requests, shutil, subprocess
from typing import Optional, List
from pathlib import Path
import hashlib
import base64
import zipfile
import xml.etree.ElementTree as ET
import json

GOTENBERG_URL = os.getenv("GOTENBERG_URL", "http://gotenberg:3000")
GOTENBERG_TIMEOUT = int(os.getenv("GOTENBERG_TIMEOUT", "120"))
GOTENBERG_MAX_RETRIES = int(os.getenv("GOTENBERG_MAX_RETRIES", "3"))
GOTENBERG_BACKOFF_BASE = float(os.getenv("GOTENBERG_BACKOFF_BASE", "0.6"))
PDF_PAPER = os.getenv("PDF_PAPER", "auto")
PDF_MARGIN_MM = int(os.getenv("PDF_MARGIN_MM", "10"))
CONVERTER_ENDPOINT = os.getenv("DOC_CONVERTER_URL", "").strip()

OFFICE_EXT = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".odp", ".ods", ".rtf"}
HWP_EXT    = {".hwp", ".hwpx"}
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
    raise last or ConvertError("Unknown error")

def _chromium_opts(no_margins: bool = False, prefer_css_page_size: bool = True) -> dict:
    d = {"preferCssPageSize": "true" if prefer_css_page_size else "false"}
    if PDF_PAPER.lower() != "auto":
        d["paperWidth"] = "8.27"
        d["paperHeight"] = "11.7"
    if not no_margins:
        mm_str = f"{PDF_MARGIN_MM}mm"
        d["marginTop"] = d["marginBottom"] = d["marginLeft"] = d["marginRight"] = mm_str
    return d

# ========== [추가] 한글 폰트 등록 ==========
def _register_korean_font():
    """reportlab에 한글 폰트 등록"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    if "NanumGothic" in pdfmetrics.getRegisteredFontNames():
        return "NanumGothic"
    
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/truetype/nanum-gothic/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("NanumGothic", font_path))
                print(f"[FONT] Registered Korean font: {font_path}")
                return "NanumGothic"
            except Exception as e:
                print(f"[FONT] Failed to register {font_path}: {e}")
                continue
    
    print("[FONT] No Korean font found")
    return "Helvetica"

# ========== TXT → PDF 변환 (한글 지원) ==========
def _text_to_pdf_bytes(text: str, skip_ocr: bool = True) -> bytes:
    """텍스트를 PDF bytes로 변환 (한글 폰트 지원)"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm

    korean_font = _register_korean_font()
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    if skip_ocr:
        c.setTitle("HWP_CONVERTED")
        c.setSubject("NO_OCR_NEEDED")
        c.setKeywords("text_layer_complete")
    

    margin_x = 20 * mm
    margin_y = 20 * mm
    y = height - margin_y

    font_size = 10
    c.setFont(korean_font, font_size)

    import textwrap
    lines = []
    for para in (text or "").splitlines():
        wrap_width = 45 if any(ord(ch) > 127 for ch in para) else 95
        wrapped = textwrap.wrap(para, width=wrap_width, break_long_words=False, break_on_hyphens=False)
        if wrapped:
            lines.extend(wrapped)
        else:
            lines.append("")

    line_height = font_size * 1.5
    
    for line in lines:
        if y <= margin_y:
            c.showPage()
            c.setFont(korean_font, font_size)
            y = height - margin_y
        
        try:
            c.drawString(margin_x, y, line)
        except Exception as e:
            print(f"[FONT] ⚠️ Failed to draw line: {e}")
            safe_line = line.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            c.drawString(margin_x, y, safe_line)
        
        y -= line_height

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

# ========== HWPX XML 파싱 (인코딩 자동 감지) ==========
def _extract_text_from_hwpx(hwpx_path: Path) -> str:
    """HWPX 파일에서 텍스트 추출 (인코딩 자동 감지)"""
    try:
        texts = []
        
        with zipfile.ZipFile(hwpx_path, 'r') as zf:
            section_files = [f for f in zf.namelist() if f.startswith('Contents/section') and f.endswith('.xml')]
            section_files.sort()
            
            print(f"[HWPX] Found {len(section_files)} section files")
            
            for section_file in section_files:
                try:
                    with zf.open(section_file) as f:
                        raw_content = f.read()
                    
                    # 인코딩 자동 감지
                    encoding = 'utf-8'
                    
                    if raw_content.startswith(b'<?xml'):
                        declaration = raw_content.split(b'?>')[0] + b'?>'
                        if b'encoding=' in declaration:
                            enc_match = declaration.split(b'encoding=')[1]
                            if enc_match.startswith(b'"'):
                                encoding = enc_match.split(b'"')[1].decode('ascii').lower()
                            elif enc_match.startswith(b"'"):
                                encoding = enc_match.split(b"'")[1].decode('ascii').lower()
                    
                    if raw_content.startswith(b'\xff\xfe'):
                        encoding = 'utf-16-le'
                    elif raw_content.startswith(b'\xfe\xff'):
                        encoding = 'utf-16-be'
                    elif raw_content.startswith(b'\xef\xbb\xbf'):
                        encoding = 'utf-8-sig'
                    
                    print(f"[HWPX] Parsing {section_file} with encoding: {encoding}")
                    
                    try:
                        content_str = raw_content.decode(encoding)
                    except (UnicodeDecodeError, LookupError):
                        for fallback_enc in ['utf-16', 'utf-16-le', 'utf-8', 'cp949', 'euc-kr']:
                            try:
                                content_str = raw_content.decode(fallback_enc)
                                print(f"[HWPX] Fallback encoding worked: {fallback_enc}")
                                break
                            except:
                                continue
                        else:
                            print(f"[HWPX] All encodings failed for {section_file}")
                            continue
                    
                    root = ET.fromstring(content_str.encode('utf-8'))
                    
                    def extract_text_recursive(element) -> List[str]:
                        result = []
                        if element.text and element.text.strip():
                            result.append(element.text.strip())
                        for child in element:
                            result.extend(extract_text_recursive(child))
                            if child.tail and child.tail.strip():
                                result.append(child.tail.strip())
                        return result
                    
                    section_texts = extract_text_recursive(root)
                    texts.extend(section_texts)
                    
                    print(f"[HWPX] Extracted {len(section_texts)} text elements from {section_file}")
                    
                except Exception as e:
                    print(f"[HWPX] ⚠️ Failed to parse {section_file}: {e}")
                    continue
        
        full_text = "\n".join(texts)
        
        if not full_text.strip():
            raise ValueError("HWPX에서 텍스트를 추출할 수 없습니다")
        
        print(f"[HWPX] Total extracted: {len(full_text)} characters")
        return full_text
        
    except zipfile.BadZipFile:
        raise ConvertError("유효하지 않은 HWPX 파일입니다")
    except Exception as e:
        raise ConvertError(f"HWPX 텍스트 추출 실패: {e}")

# ========== HWP → PDF 변환 (pyhwp) ==========
def _hwp_to_pdf_via_text(src: Path, out: Path):
    """HWP → TXT → PDF 변환 (pyhwp 사용)"""
    import subprocess
    import tempfile
    
    print(f"[CONVERT] Converting HWP to PDF via pyhwp: {src}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_txt:
        txt_path = tmp_txt.name
    
    try:
        result = subprocess.run(
            ["hwp5txt", "--output", txt_path, str(src)],
            capture_output=True,
            timeout=60,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[CONVERT] hwp5txt failed: {result.stderr}")
            raise ConvertError(f"hwp5txt 실패: {result.stderr[:200]}")
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text.strip():
            raise ConvertError("HWP에서 추출된 텍스트가 비어있습니다")
        
        print(f"[CONVERT] Extracted {len(text)} characters from HWP")
        
        pdf_bytes = _text_to_pdf_bytes(text)
        
        with open(out, 'wb') as f:
            f.write(pdf_bytes)
        
        os.unlink(txt_path)
        
        print(f"[CONVERT] HWP→PDF via pyhwp 성공: {out}")
        
    except subprocess.TimeoutExpired:
        if os.path.exists(txt_path):
            os.unlink(txt_path)
        raise ConvertError("HWP 변환 타임아웃")
    except Exception as e:
        if os.path.exists(txt_path):
            os.unlink(txt_path)
        raise ConvertError(f"HWP 변환 실패: {e}")

# ========== HWP/HWPX → PDF 변환 (통합) ==========
def _hwp_to_pdf(src: Path, out: Path):
    """HWP/HWPX → PDF 변환"""
    ext = src.suffix.lower()
    
    # HWPX 우선 처리
    if ext == ".hwpx":
        print(f"[CONVERT] Converting HWPX to PDF via XML parsing: {src}")
        try:
            text = _extract_text_from_hwpx(src)
            print(f"[CONVERT] Extracted {len(text)} characters from HWPX")
            print(f"[CONVERT] Sample text: {text[:200]}")
            
            pdf_bytes = _text_to_pdf_bytes(text)
            with open(out, 'wb') as f:
                f.write(pdf_bytes)
            
            print(f"[CONVERT] HWPX→PDF via XML parsing 성공: {out}")
            return
        except Exception as e:
            print(f"[CONVERT] HWPX XML parsing 실패: {e}")
    
    # HWP 처리
    if ext == ".hwp":
        if shutil.which("hwp5txt"):
            try:
                _hwp_to_pdf_via_text(src, out)
                return
            except Exception as e:
                print(f"[CONVERT] pyhwp 실패: {e}")
    
    # 외부 컨버터
    if CONVERTER_ENDPOINT:
        try:
            with open(src, "rb") as f:
                content = f.read()
            pdf_bytes = convert_stream_to_pdf_bytes(content, src.suffix.lower())
            if pdf_bytes:
                with open(out, "wb") as fw:
                    fw.write(pdf_bytes)
                print(f"[CONVERT] HWP/HWPX→PDF via DOC_CONVERTER_URL: {out}")
                return
        except Exception as e:
            print(f"[CONVERT] DOC_CONVERTER_URL 실패: {e}")
    
    if ext == ".hwpx":
        raise ConvertError("HWPX 변환 실패: XML 파싱이 실패했습니다")
    else:
        raise ConvertError("HWP 변환 실패: pyhwp를 설치하거나 DOC_CONVERTER_URL을 설정해주세요")

# ========== Local file path 기반 변환 ==========
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

# ========== public: 로컬 파일 경로 기반 변환 ==========
def convert_to_pdf(src_path: str) -> str:
    """입력 파일을 PDF로 변환해서 로컬 경로 반환"""
    src = Path(src_path)
    ext = src.suffix.lower()
    if ext == ".pdf":
        return str(src)

    out = src.with_suffix(".pdf")
    _ensure_parent(out)

    if ext in HWP_EXT:
        _hwp_to_pdf(src, out)
    elif ext in OFFICE_EXT:
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
        raise ConvertError("변환된 PDF가 비어있습니다")
    
    return str(out)

# ========== public: bytes 기반 변환 ==========
def convert_bytes_to_pdf_bytes(content: bytes, src_ext: str) -> bytes | None:
    """bytes를 PDF bytes로 변환"""
    ext = (src_ext or "").lower()

    if ext == ".pdf":
        return content

    if ext in TXT_EXT:
        try:
            text = content.decode("utf-8", errors="ignore")
            return _text_to_pdf_bytes(text)
        except Exception as e:
            print(f"[CONVERT] TXT→PDF 변환 실패: {e}")
            return None

    if ext in HWP_EXT:
        if CONVERTER_ENDPOINT:
            try:
                return convert_stream_to_pdf_bytes(content, ext)
            except Exception as e:
                print(f"[CONVERT] HWP→PDF 변환 실패: {e}")
        return None

    if ext in OFFICE_EXT:
        if not _gotenberg_ok():
            return None
        url = f"{GOTENBERG_URL}/forms/libreoffice/convert"
        files = {"files": (f"upload{ext}", io.BytesIO(content), "application/octet-stream")}
        try:
            return _post_retry(url, files)
        except Exception:
            return None

    if ext in HTML_EXT:
        if not _gotenberg_ok():
            return None
        url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
        files = [("files", ("index.html", io.BytesIO(content), "text/html; charset=utf-8"))]
        try:
            return _post_retry(url, files, data=_chromium_opts())
        except Exception:
            return None

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

    return None

def convert_stream_to_pdf_bytes(content: bytes, src_ext: str) -> Optional[bytes]:
    """외부 변환기로 bytes를 PDF bytes로 변환"""
    if not CONVERTER_ENDPOINT:
        raise ConvertStreamError("DOC_CONVERTER_URL이 설정되지 않았습니다")
    
    file_hash = hashlib.md5(content).hexdigest()
    ext = src_ext.lstrip('.').lower()
    
    format_map = {
        'hwp': 'hwp', 'hwpx': 'hwpx',
        'doc': 'doc', 'docx': 'docx',
        'xls': 'xls', 'xlsx': 'xlsx',
        'ppt': 'ppt', 'pptx': 'pptx',
        'odt': 'odt', 'ods': 'ods', 'odp': 'odp', 'rtf': 'rtf'
    }
    
    if ext not in format_map:
        raise ConvertStreamError(f"지원하지 않는 형식: {ext}")
    
    base64_content = base64.b64encode(content).decode('utf-8')
    
    payload = {
        "async": False,
        "filetype": format_map[ext],
        "key": file_hash,
        "outputtype": "pdf",
        "title": f"document.{ext}",
        "url": f"data:application/octet-stream;base64,{base64_content}"
    }
    
    try:
        response = requests.post(
            f"{CONVERTER_ENDPOINT}/ConvertService.ashx",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("error") == 0:
                pdf_url = result.get("fileUrl")
                if pdf_url:
                    pdf_response = requests.get(pdf_url, timeout=60)
                    if pdf_response.status_code == 200:
                        pdf_bytes = pdf_response.content
                        if len(pdf_bytes) > 100:
                            return pdf_bytes
                        else:
                            raise ConvertStreamError(f"변환된 PDF가 너무 작습니다: {len(pdf_bytes)} bytes")
                    else:
                        raise ConvertStreamError(f"PDF 다운로드 실패: HTTP {pdf_response.status_code}")
                else:
                    raise ConvertStreamError("ONLYOFFICE 응답에 fileUrl이 없음")
            else:
                error_msg = result.get("error", "알 수 없는 오류")
                raise ConvertStreamError(f"ONLYOFFICE 변환 실패: {error_msg}")
        else:
            raise ConvertStreamError(f"ONLYOFFICE 서버 오류: HTTP {response.status_code}")
    
    except requests.exceptions.Timeout:
        raise ConvertStreamError("ONLYOFFICE 서버 타임아웃")
    except requests.exceptions.RequestException as e:
        raise ConvertStreamError(f"ONLYOFFICE 연결 실패: {e}")
    except Exception as e:
        raise ConvertStreamError(f"ONLYOFFICE 변환 오류: {e}")