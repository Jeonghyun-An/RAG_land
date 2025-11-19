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
    reportlab이 없으면 ImportError 발생 -> 상위에서 처리.
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

# ========== [추가] HWP → PDF 변환 함수 ==========
def _hwp_to_pdf(src: Path, out: Path):
    """
    HWP → PDF 변환
    우선순위:
    1. pyhwp (텍스트 추출) → reportlab (PDF 생성)
    2. DOC_CONVERTER_URL (외부 서비스)
    3. unoconv/soffice (작동 안 함)
    """
    # 1순위: pyhwp (무료 오픈소스)
    if shutil.which("hwp5txt"):
        try:
            _hwp_to_pdf_via_text(src, out)
            return
        except Exception as e:
            print(f"[CONVERT] pyhwp 실패: {e}")
    
    # 2순위: 외부 컨버터
    if CONVERTER_ENDPOINT:
        try:
            with open(src, "rb") as f:
                content = f.read()
            pdf_bytes = convert_stream_to_pdf_bytes(content, src.suffix.lower())
            if pdf_bytes:
                with open(out, "wb") as fw:
                    fw.write(pdf_bytes)
                print(f"[CONVERT] ✅ HWP→PDF via DOC_CONVERTER_URL: {out}")
                return
        except Exception as e:
            print(f"[CONVERT] DOC_CONVERTER_URL 실패: {e}")
    
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
    
    try:
        import json
        import base64
        import hashlib
        
        filename = f"document{src_ext}"
        
        # ========== ONLYOFFICE 전용 요청 형식 ==========
        print(f"[CONVERT] Attempting conversion via ONLYOFFICE: {filename}")
        
        # 파일을 base64로 인코딩
        file_base64 = base64.b64encode(content).decode('utf-8')
        
        # ONLYOFFICE 요청 페이로드
        payload = {
            "async": False,
            "filetype": src_ext.lstrip('.'),  # .hwp → hwp
            "key": hashlib.md5(content).hexdigest(),  # 고유 키
            "outputtype": "pdf",
            "title": filename,
            "url": f"data:application/octet-stream;base64,{file_base64}"
        }
        
        print(f"[CONVERT] Sending to ONLYOFFICE: filetype={payload['filetype']}, key={payload['key'][:8]}...")
        
        response = requests.post(
            CONVERTER_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=GOTENBERG_TIMEOUT
        )
        
        print(f"[CONVERT] ONLYOFFICE response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[CONVERT] ONLYOFFICE result: {result}")
            
            # ONLYOFFICE 응답 구조: {"endConvert": true, "fileUrl": "...", "percent": 100}
            if result.get("endConvert"):
                pdf_url = result.get("fileUrl")
                if pdf_url:
                    print(f"[CONVERT] Downloading PDF from: {pdf_url}")
                    
                    # PDF URL에서 실제 파일 다운로드
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