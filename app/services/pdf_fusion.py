# app/services/pdf_fusion.py
"""
PDF 융합 파서 개선 버전
- pdfminer 텍스트 추출 + OCR 보강
- bbox 정보 정확도 향상
- CID 패턴 명시적 감지 추가
- 워터마크 필터링 강화
- 표 영역 bbox 최적화
"""
from __future__ import annotations
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    try:
        Image.ANTIALIAS = Image.LANCZOS
    except Exception:
        from PIL import Image as _I
        Image.ANTIALIAS = getattr(_I, "BICUBIC", None)
import fitz
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import re
from io import BytesIO

# [신규] EasyOCR Reader 싱글톤 캐싱
_EASYOCR_READER_CACHE = {}

def _get_easyocr_reader(langs: list[str], gpu: bool):
    """
    [성능 개선] EasyOCR Reader를 재사용
    - 언어와 GPU 설정별로 캐싱
    - 페이지마다 새로 생성하지 않음
    """
    cache_key = (tuple(sorted(langs)), gpu)
    
    if cache_key not in _EASYOCR_READER_CACHE:
        import easyocr
        print(f"[OCR] Creating EasyOCR Reader: langs={langs}, gpu={gpu}")
        _EASYOCR_READER_CACHE[cache_key] = easyocr.Reader(langs, gpu=gpu)
    
    return _EASYOCR_READER_CACHE[cache_key]

# easyocr 언어코드 보정
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


# 🔥 NEW: OCR 필요 여부 종합 판단
def _needs_ocr_for_page(text: str, min_chars: int) -> bool:
    """
    페이지 텍스트가 OCR이 필요한지 판단
    
    Args:
        text: 페이지 텍스트
        min_chars: 최소 문자 수 임계값
    
    Returns:
        True if OCR needed, False otherwise
    """
    if not text or len(text.strip()) < min_chars:
        return True
    
    # 🔥 CID 패턴 감지 (명시적)
    cid_threshold = float(os.getenv("OCR_CID_THRESHOLD", "0.2"))
    cid_pattern = re.compile(r'\(cid:\d+\)')
    cid_matches = cid_pattern.findall(text)
    cid_char_count = sum(len(m) for m in cid_matches)
    
    if len(text) > 0:
        cid_ratio = cid_char_count / len(text)
        if cid_ratio > cid_threshold:
            print(f"[OCR] CID pattern detected ({cid_ratio:.1%} > {cid_threshold:.0%}), triggering OCR")
            return True
    
    # 🔥 읽을 수 있는 문자 비율 체크
    min_readable_ratio = float(os.getenv("OCR_MIN_READABLE_RATIO", "0.3"))
    readable_chars = len(re.findall(r'[a-zA-Z0-9가-힣]', text))
    total_chars = len(text)
    
    if total_chars > 0:
        readable_ratio = readable_chars / total_chars
        if readable_ratio < min_readable_ratio:
            print(f"[OCR] Low readable ratio ({readable_ratio:.1%} < {min_readable_ratio:.0%}), triggering OCR")
            return True
    
    return False


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
                    blocks.append({
                        "text": t, 
                        "bbox": {
                            "x0": float(x0), 
                            "y0": float(y0), 
                            "x1": float(x1), 
                            "y1": float(y1)
                        }
                    })
        full = "\n\n".join(texts).strip()
        pages.append((i, full))
        if blocks:
            layout_map[i] = blocks
    return pages, layout_map

def _pdfminer_pages_and_blocks_from_bytes(pdf_bytes: bytes) -> Tuple[List[Tuple[int,str]], Dict[int, List[Dict]]]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LAParams
    laparams = LAParams()

    pages: List[Tuple[int,str]] = []
    layout_map: Dict[int, List[Dict]] = {}

    with BytesIO(pdf_bytes) as bio:
        for i, layout in enumerate(extract_pages(bio, laparams=laparams), start=1):
            texts = []
            blocks: List[Dict] = []
            for elem in layout:
                if isinstance(elem, LTTextContainer):
                    t = (elem.get_text() or "").strip()
                    if t:
                        texts.append(t)
                        x0, y0, x1, y1 = elem.bbox
                        blocks.append({
                            "text": t,
                            "bbox": {
                                "x0": float(x0), 
                                "y0": float(y0), 
                                "x1": float(x1), 
                                "y1": float(y1)
                            }
                        })
            full = "\n\n".join(texts).strip()
            pages.append((i, full))
            if blocks:
                layout_map[i] = blocks
    return pages, layout_map


def _filter_watermark_blocks(blocks: List[Dict], page_width: float, 
                            page_height: float) -> List[Dict]:
    """
    [신규] 워터마크 블록 필터링
    - 페이지 중앙에 큰 글자
    - 반복되는 패턴
    - 투명도 높은 텍스트 (간접 추정)
    """
    if not blocks:
        return blocks
    
    filtered = []
    
    # 중앙 영역 정의 (페이지의 중앙 30%)
    center_x = page_width / 2
    center_y = page_height / 2
    center_tolerance = 0.15  # 중앙의 ±15%
    
    for block in blocks:
        bbox = block['bbox']
        
        # bbox 중심
        block_center_x = (bbox['x0'] + bbox['x1']) / 2
        block_center_y = (bbox['y0'] + bbox['y1']) / 2
        
        # 중앙 근처 여부
        is_center = (
            abs(block_center_x - center_x) / page_width < center_tolerance and
            abs(block_center_y - center_y) / page_height < center_tolerance
        )
        
        # 크기 (워터마크는 보통 큼)
        width = bbox['x1'] - bbox['x0']
        height = bbox['y1'] - bbox['y0']
        is_large = width > page_width * 0.3 or height > page_height * 0.1
        
        # 워터마크 키워드
        text_lower = block['text'].lower()
        watermark_keywords = ['draft', 'confidential', '기밀', '초안', 'watermark']
        has_watermark_keyword = any(kw in text_lower for kw in watermark_keywords)
        
        # 워터마크로 판단되면 제외
        if is_center and is_large and has_watermark_keyword:
            continue
        
        filtered.append(block)
    
    return filtered


# ---------- 내부: EasyOCR로 한 페이지 OCR (bbox 정확도 개선) ----------
def _ocr_page_with_easyocr(img_nd: "np.ndarray", lang: str, gpu: bool) -> Tuple[str, List[Dict]]:
    """
    [개선] EasyOCR bbox 정확도 향상 + Reader 재사용
    """
    import numpy as np
    
    langs = _norm_easyocr_langs(lang)
    # [핵심 수정] Reader 재사용
    reader = _get_easyocr_reader(langs, gpu)
    
    # detail=1로 bbox, text, confidence 받기
    res = reader.readtext(img_nd, detail=1, paragraph=False)
    
    texts = []
    blocks: List[Dict] = []
    
    # 신뢰도 임계값
    confidence_threshold = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.3"))
    
    for item in res:
        if not item or len(item) < 3:
            continue
        
        pts, txt, conf = item[0], (item[1] or "").strip(), item[2]
        
        # 신뢰도 필터링
        if conf < confidence_threshold:
            continue
        
        if not txt:
            continue
        
        try:
            # 폴리곤 포인트에서 bbox 추출
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, y0, x1, y1 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
            
            # bbox 유효성 검증
            if x1 <= x0 or y1 <= y0:
                continue
            
            texts.append(txt)
            blocks.append({
                "text": txt,
                "bbox": {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1
                },
                "confidence": conf
            })
        except Exception:
            pass
    
    # 워터마크 필터링 (선택적)
    if os.getenv("OCR_FILTER_WATERMARKS", "1") == "1" and blocks:
        blocks = _filter_watermark_blocks(blocks, img_nd.shape[1], img_nd.shape[0])
    
    # Y 좌표 기준 정렬 (읽는 순서)
    blocks.sort(key=lambda b: (b['bbox']['y0'], b['bbox']['x0']))
    sorted_texts = [b['text'] for b in blocks]
    
    return ("\n".join(sorted_texts).strip(), blocks)


def _ocr_page_with_tesseract(img_nd: "np.ndarray", lang: str) -> Tuple[str, List[Dict]]:
    """
    [개선] Tesseract bbox 정확도 향상
    - image_to_data로 단어 단위 bbox 추출
    - 신뢰도 필터링
    """
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
    blocks: List[Dict] = []
    confidence_threshold = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.3"))
    
    try:
        data = pytesseract.image_to_data(
            pil, 
            lang=(lang or "kor+eng"), 
            output_type=pytesseract.Output.DICT
        )
        
        n = len(data.get("text", []))
        for i in range(n):
            wtxt = (data["text"][i] or "").strip()
            conf = float(data.get("conf", [0])[i]) / 100.0  # 0-100 -> 0-1
            
            if not wtxt or conf < confidence_threshold:
                continue
            
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            
            blocks.append({
                "text": wtxt, 
                "bbox": {
                    "x0": float(x), 
                    "y0": float(y), 
                    "x1": float(x + w), 
                    "y1": float(y + h)
                },
                "confidence": conf
            })
    except Exception:
        pass
    
    return text, blocks


# ---------- 내부: 한 페이지에 OCR 적용 ----------
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
        gpu  = os.getenv("OCR_EASYOCR_GPU", "1").strip() == "1"
        return _ocr_page_with_easyocr(img, lang, gpu)


# ---------- 공개: 경로 입력을 OCR-융합으로 뽑기 ----------
def extract_pdf_fused(path: str) -> Tuple[List[Tuple[int,str]], Dict[int, List[Dict]]]:
    """
    [개선] PDF 융합 추출 (경로)
    반환:
      pages: [(page_no, text)]
      layout_map: {page_no: [ {"text":..., "bbox":{...}, "confidence":...}, ... ] }
    규칙:
      - OCR_MODE=off → pdfminer 결과 그대로
      - OCR_MODE=force → 모든 페이지를 OCR로 대체(텍스트/박스)
      - OCR_MODE=auto → CID 패턴 또는 텍스트 부족 페이지만 OCR
    """
    import fitz
    pages, layout_map = _pdfminer_pages_and_blocks_from_path(path)

    ocr_mode = os.getenv("OCR_MODE", "auto").strip().lower()  # off|auto|force
    min_chars = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", "50"))

    if ocr_mode == "off":
        return pages, layout_map

    doc = fitz.open(path)
    try:
        for (idx, text), pg in zip(pages, doc, strict=False):
            # 🔥 개선: CID 패턴 명시적 감지
            need_ocr = (ocr_mode == "force") or _needs_ocr_for_page(text, min_chars)
            
            if need_ocr:
                txt, blocks = _ocr_page_image(pg)
                # 교체
                pages[idx-1] = (idx, txt or "")
                layout_map[idx] = blocks or []
    finally:
        doc.close()
    return pages, layout_map


def extract_pdf_fused_from_bytes(pdf_bytes: bytes) -> Tuple[List[Tuple[int,str]], Dict[int, List[Dict]]]:
    """
    [개선] PDF 융합 추출 (바이트)
    """
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
            
            # 🔥 개선: CID 패턴 명시적 감지
            need_ocr = (ocr_mode == "force") or _needs_ocr_for_page(cur_txt, min_chars)
            
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