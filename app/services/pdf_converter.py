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
import hashlib
import base64
import zipfile
import xml.etree.ElementTree as ET

GOTENBERG_URL = os.getenv("GOTENBERG_URL", "http://gotenberg:3000")
GOTENBERG_TIMEOUT = int(os.getenv("GOTENBERG_TIMEOUT", "120"))
GOTENBERG_MAX_RETRIES = int(os.getenv("GOTENBERG_MAX_RETRIES", "3"))
GOTENBERG_BACKOFF_BASE = float(os.getenv("GOTENBERG_BACKOFF_BASE", "0.6"))
PDF_PAPER = os.getenv("PDF_PAPER", "auto")
PDF_MARGIN_MM = int(os.getenv("PDF_MARGIN_MM", "10"))
CONVERTER_ENDPOINT = os.getenv("DOC_CONVERTER_URL", "").strip()

OFFICE_EXT = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".odp", ".ods", ".rtf"}
HWP_EXT    = {".hwp", ".hwpx"}  # 한글 파일은 별도 처리
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

def _register_korean_font():
    """
    reportlab에 한글 폰트 등록
    나눔고딕 폰트를 시스템에서 찾아 등록
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # 이미 등록되어 있으면 스킵
    if "NanumGothic" in pdfmetrics.getRegisteredFontNames():
        return "NanumGothic"
    
    # 나눔고딕 폰트 경로 후보들
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/truetype/nanum-gothic/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("NanumGothic", font_path))
                print(f"[FONT] ✅ Registered Korean font: {font_path}")
                return "NanumGothic"
            except Exception as e:
                print(f"[FONT] ⚠️ Failed to register {font_path}: {e}")
                continue
    
    # 폰트를 찾지 못한 경우 경고
    print("[FONT] ⚠️ No Korean font found, text may display as squares")
    return "Helvetica"  # 기본 폰트로 폴백

# ---------- TXT → PDF 변환 (reportlab) ----------
def _text_to_pdf_bytes(text: str) -> bytes:
    """
    텍스트를 PDF bytes로 변환 (reportlab 사용)
    ✅ 한글 폰트 지원 추가
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm

    # ✅ 한글 폰트 등록
    korean_font = _register_korean_font()

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin_x = 20 * mm
    margin_y = 20 * mm
    max_width = width - 2 * margin_x
    y = height - margin_y

    # ✅ 한글 폰트 설정
    font_size = 10
    c.setFont(korean_font, font_size)

    # 워드랩 (한글 고려)
    import textwrap
    lines = []
    for para in (text or "").splitlines():
        # ✅ 한글은 폭이 넓으므로 줄당 글자 수 조정
        # 영문: ~95자, 한글: ~45자
        wrap_width = 45 if any(ord(ch) > 127 for ch in para) else 95
        wrapped = textwrap.wrap(para, width=wrap_width, break_long_words=False, break_on_hyphens=False)
        if wrapped:
            lines.extend(wrapped)
        else:
            lines.append("")  # 빈 줄 유지

    line_height = font_size * 1.5  # 줄 간격
    
    for line in lines:
        if y <= margin_y:
            c.showPage()
            c.setFont(korean_font, font_size)
            y = height - margin_y
        
        # ✅ 안전한 텍스트 그리기
        try:
            c.drawString(margin_x, y, line)
        except Exception as e:
            # 특수문자 처리 실패 시 대체
            print(f"[FONT] ⚠️ Failed to draw line: {e}")
            safe_line = line.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            c.drawString(margin_x, y, safe_line)
        
        y -= line_height

    c.showPage()
    c.save()
    
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

# ==========  HWPX XML 파싱 함수 ==========
def _extract_text_from_hwpx(hwpx_path: Path) -> str:
    """
    HWPX 파일에서 텍스트 추출
    HWPX는 ZIP 압축된 XML 파일들로 구성
    """
    try:
        texts = []
        
        # HWPX는 ZIP 파일
        with zipfile.ZipFile(hwpx_path, 'r') as zf:
            # Contents/section*.xml 파일들에 본문 내용
            section_files = [f for f in zf.namelist() if f.startswith('Contents/section') and f.endswith('.xml')]
            section_files.sort()  # 순서대로 정렬
            
            for section_file in section_files:
                try:
                    with zf.open(section_file) as f:
                        content = f.read()
                        
                    # XML 파싱
                    root = ET.fromstring(content)
                    
                    # 텍스트 노드 찾기 (재귀적으로)
                    for text_elem in root.iter():
                        # 텍스트 내용이 있는 요소
                        if text_elem.text and text_elem.text.strip():
                            texts.append(text_elem.text.strip())
                        if text_elem.tail and text_elem.tail.strip():
                            texts.append(text_elem.tail.strip())
                            
                except Exception as e:
                    print(f"[HWPX] Warning: Failed to parse {section_file}: {e}")
                    continue
        
        full_text = "\n".join(texts)
        
        if not full_text.strip():
            raise ValueError("HWPX에서 텍스트를 추출할 수 없습니다")
        
        return full_text
        
    except zipfile.BadZipFile:
        raise ConvertError("유효하지 않은 HWPX 파일입니다 (ZIP 형식 오류)")
    except Exception as e:
        raise ConvertError(f"HWPX 텍스트 추출 실패: {e}")


def _hwp_to_pdf_via_text(src: Path, out: Path):
    """
    HWP → TXT → PDF 변환 (pyhwp 사용)
    무료 오픈소스 방식
    """
    import subprocess
    import tempfile
    
    print(f"[CONVERT] Converting HWP to PDF via pyhwp: {src}")
    
    # 1단계: HWP → TXT 변환 (pyhwp)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_txt:
        txt_path = tmp_txt.name
    
    try:
        # hwp5txt 명령어로 텍스트 추출
        result = subprocess.run(
            ["hwp5txt", "--output", txt_path, str(src)],
            capture_output=True,
            timeout=60,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[CONVERT] hwp5txt failed: {result.stderr}")
            raise ConvertError(f"hwp5txt 실패: {result.stderr[:200]}")
        
        # 2단계: TXT → PDF 변환 (reportlab)
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text.strip():
            raise ConvertError("HWP에서 추출된 텍스트가 비어있습니다")
        
        print(f"[CONVERT] Extracted {len(text)} characters from HWP")
        
        # reportlab로 PDF 생성
        pdf_bytes = _text_to_pdf_bytes(text)
        
        with open(out, 'wb') as f:
            f.write(pdf_bytes)
        
        # 임시 파일 정리
        os.unlink(txt_path)
        
        print(f"[CONVERT] ✅ HWP→PDF via pyhwp 성공: {out}")
        
    except subprocess.TimeoutExpired:
        if os.path.exists(txt_path):
            os.unlink(txt_path)
        raise ConvertError("HWP 변환 타임아웃")
    except Exception as e:
        if os.path.exists(txt_path):
            os.unlink(txt_path)
        raise ConvertError(f"HWP 변환 실패: {e}")

# ========== _hwp_to_pdf 함수 수정 (HWPX 지원 추가) ==========
def _hwp_to_pdf(src: Path, out: Path):
    """
    HWP/HWPX → PDF 변환
    우선순위:
    1. HWPX: XML 파싱 → reportlab
    2. HWP: pyhwp → reportlab
    3. 외부 컨버터 (DOC_CONVERTER_URL)
    """
    ext = src.suffix.lower()
    
    # ✅ HWPX 우선 처리 (pyhwp는 지원 안 함)
    if ext == ".hwpx":
        print(f"[CONVERT] Converting HWPX to PDF via XML parsing: {src}")
        try:
            # HWPX → 텍스트 추출
            text = _extract_text_from_hwpx(src)
            print(f"[CONVERT] Extracted {len(text)} characters from HWPX")
            
            # 텍스트 → PDF
            pdf_bytes = _text_to_pdf_bytes(text)
            with open(out, 'wb') as f:
                f.write(pdf_bytes)
            
            print(f"[CONVERT] ✅ HWPX→PDF via XML parsing 성공: {out}")
            return
            
        except Exception as e:
            print(f"[CONVERT] HWPX XML parsing 실패: {e}")
            # HWPX XML 파싱 실패 시 외부 컨버터로 폴백
    
    # ✅ HWP 처리 (pyhwp 사용)
    if ext == ".hwp":
        if shutil.which("hwp5txt"):
            try:
                _hwp_to_pdf_via_text(src, out)
                return
            except Exception as e:
                print(f"[CONVERT] pyhwp 실패: {e}")
    
    # ✅ 외부 컨버터 (HWP/HWPX 모두 지원)
    if CONVERTER_ENDPOINT:
        try:
            with open(src, "rb") as f:
                content = f.read()
            pdf_bytes = convert_stream_to_pdf_bytes(content, src.suffix.lower())
            if pdf_bytes:
                with open(out, "wb") as fw:
                    fw.write(pdf_bytes)
                print(f"[CONVERT] ✅ HWP/HWPX→PDF via DOC_CONVERTER_URL: {out}")
                return
        except Exception as e:
            print(f"[CONVERT] DOC_CONVERTER_URL 실패: {e}")
    
    # ✅ 모든 방법 실패
    if ext == ".hwpx":
        raise ConvertError(
            "HWPX 변환 실패: XML 파싱이 실패했습니다.\n"
            "대안: DOC_CONVERTER_URL을 설정하거나 HWP 5.0 형식으로 저장해주세요."
        )
    else:
        raise ConvertError(
            "HWP 변환 실패: pyhwp를 설치하거나 DOC_CONVERTER_URL을 설정해주세요\n"
            "설치 방법: pip install pyhwp"
        )
# ---------- public: 로컬 파일 경로 기반 변환 ----------
def convert_to_pdf(src_path: str) -> str:
    """입력 파일을 PDF로 변환해서 로컬 경로 반환. 이미 PDF면 그대로 반환."""
    src = Path(src_path)
    ext = src.suffix.lower()
    if ext == ".pdf":
        return str(src)

    out = src.with_suffix(".pdf")
    _ensure_parent(out)

    # ========== [핵심 수정] HWP 별도 처리 ==========
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
        raise ConvertError("변환된 PDF가 비어있습니다.")
    
    return str(out)

# ---------- public: bytes 기반 변환 (java_router용) ----------
def convert_bytes_to_pdf_bytes(content: bytes, src_ext: str) -> bytes | None:
    """
    bytes를 PDF bytes로 변환
    [핵심 수정] TXT, HWP 파일 지원 추가
    
    Returns:
        PDF bytes if success, None if unsupported
    """
    ext = (src_ext or "").lower()

    # 이미 PDF면 그대로
    if ext == ".pdf":
        return content

    # 1) TXT 류 → reportlab 변환
    if ext in TXT_EXT:
        try:
            text = content.decode("utf-8", errors="ignore")
            return _text_to_pdf_bytes(text)
        except Exception as e:
            print(f"[CONVERT] TXT→PDF 변환 실패: {e}")
            return None

    # 2) HWP 류 → 외부 컨버터 필수 (Gotenberg 지원 안 함)
    if ext in HWP_EXT:
        if CONVERTER_ENDPOINT:
            try:
                return convert_stream_to_pdf_bytes(content, ext)
            except Exception as e:
                print(f"[CONVERT] HWP→PDF 변환 실패: {e}")
        return None  # HWP는 bytes 변환 실패 시 None 반환 (로컬 폴백으로)

    # 3) Office 류 → LibreOffice 변환
    if ext in OFFICE_EXT:
        if not _gotenberg_ok():
            return None
        url = f"{GOTENBERG_URL}/forms/libreoffice/convert"
        files = {"files": (f"upload{ext}", io.BytesIO(content), "application/octet-stream")}
        try:
            return _post_retry(url, files)
        except Exception:
            return None

    # 4) HTML → Chromium 변환
    if ext in HTML_EXT:
        if not _gotenberg_ok():
            return None
        url = f"{GOTENBERG_URL}/forms/chromium/convert/html"
        files = [("files", ("index.html", io.BytesIO(content), "text/html; charset=utf-8"))]
        try:
            return _post_retry(url, files, data=_chromium_opts())
        except Exception:
            return None

    # 5) 단일 이미지 → HTML로 감싸 Chromium 변환
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
    외부 변환기로 bytes를 PDF bytes로 변환
    ONLYOFFICE Document Server 지원
    """
    if not CONVERTER_ENDPOINT:
        raise ConvertStreamError("DOC_CONVERTER_URL이 설정되지 않았습니다.")
    
    # 파일 해시 생성 (캐시 키로 사용 가능)
    file_hash = hashlib.md5(content).hexdigest()
    
    # src_ext에서 . 제거
    ext = src_ext.lstrip('.').lower()
    
    # ONLYOFFICE 지원 형식 매핑
    format_map = {
        'hwp': 'hwp',
        'hwpx': 'hwpx',
        'doc': 'doc',
        'docx': 'docx',
        'xls': 'xls',
        'xlsx': 'xlsx',
        'ppt': 'ppt',
        'pptx': 'pptx',
        'odt': 'odt',
        'ods': 'ods',
        'odp': 'odp',
        'rtf': 'rtf'
    }
    
    if ext not in format_map:
        raise ConvertStreamError(f"지원하지 않는 형식: {ext}")
    
    # Base64 인코딩
    base64_content = base64.b64encode(content).decode('utf-8')
    
    # ONLYOFFICE API 요청
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
                # PDF 다운로드
                pdf_url = result.get("fileUrl")
                if pdf_url:
                    pdf_response = requests.get(pdf_url, timeout=60)
                    if pdf_response.status_code == 200:
                        pdf_bytes = pdf_response.content
                        print(f"[CONVERT] ✅ ONLYOFFICE 변환 성공: {len(pdf_bytes)} bytes")
                        
                        # 최소 크기 검증
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
            raise ConvertStreamError(
                f"ONLYOFFICE 서버 오류: HTTP {response.status_code} - {response.text[:200]}"
            )
    
    except requests.exceptions.Timeout:
        raise ConvertStreamError("ONLYOFFICE 서버 타임아웃")
    except requests.exceptions.RequestException as e:
        raise ConvertStreamError(f"ONLYOFFICE 연결 실패: {e}")
    except Exception as e:
        raise ConvertStreamError(f"ONLYOFFICE 변환 오류: {e}")