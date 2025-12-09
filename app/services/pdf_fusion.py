# app/services/pdf_fusion.py
"""
PDF 융합 파서 - SmartOCRRouter 통합 버전
- pdfminer 텍스트 추출 + 지능형 OCR 보강
- 페이지 품질 기반 OCR 라우팅
- bbox 정보 정확도 향상
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


# ========== SmartOCRRouter 통합 ==========
def _get_smart_ocr_decision(
    pages: List[Tuple[int, str]], 
    layout_map: Dict[int, List[Dict]],
    ocr_mode: str
) -> Tuple[bool, List[int], Dict]:
    """
    SmartOCRRouter를 사용한 OCR 결정
    
    Returns:
        (전체_OCR_필요, 부분_OCR_페이지_리스트, 분석_정보)
    """
    # 환경변수로 SmartOCRRouter 활성화 제어
    use_smart_router = os.getenv("OCR_USE_SMART_ROUTER", "1").strip() == "1"
    
    if not use_smart_router:
        # 기존 방식 (하위 호환)
        print("[OCR] SmartOCRRouter disabled, using legacy logic")
        return _legacy_ocr_decision(pages, ocr_mode)
    
    try:
        from app.services.smart_ocr_router import SmartOCRRouter
        
        router = SmartOCRRouter()
        needs_full, partial_pages, analysis = router.should_use_ocr(pages, layout_map)
        
        print(f"[OCR] SmartOCRRouter decision:")
        print(f"  - Full OCR needed: {needs_full}")
        print(f"  - Partial OCR pages: {len(partial_pages)}")
        print(f"  - High priority pages: {analysis.get('high_priority_pages', 0)}")
        print(f"  - Avg confidence: {analysis.get('avg_confidence', 0):.3f}")
        
        # 디버그 모드면 상세 리포트 출력
        if os.getenv("OCR_DEBUG", "0") == "1":
            report = router.get_quality_report(pages, layout_map)
            print("\n[OCR] Quality Report:")
            for page_detail in report.get('page_details', []):
                print(f"  Page {page_detail['page']}: "
                      f"type={page_detail['type']}, "
                      f"needs_ocr={page_detail['needs_ocr']}, "
                      f"priority={page_detail['ocr_priority']}")
            print(f"\n  Recommendations:")
            for rec in report.get('recommendations', []):
                print(f"    {rec}")
        
        return needs_full, partial_pages, analysis
        
    except ImportError as e:
        print(f"[OCR] ⚠️  SmartOCRRouter import failed: {e}, falling back to legacy logic")
        return _legacy_ocr_decision(pages, ocr_mode)
    except Exception as e:
        print(f"[OCR] ⚠️  SmartOCRRouter error: {e}, falling back to legacy logic")
        return _legacy_ocr_decision(pages, ocr_mode)


def _legacy_ocr_decision(
    pages: List[Tuple[int, str]], 
    ocr_mode: str
) -> Tuple[bool, List[int], Dict]:
    """
    기존 OCR 결정 로직 (하위 호환)
    """
    min_chars = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", "50"))
    
    if ocr_mode == "force":
        return True, [], {'mode': 'force'}
    
    ocr_pages = []
    for page_no, text in pages:
        if _needs_ocr_for_page_legacy(text, min_chars):
            ocr_pages.append(page_no)
    
    # 50% 이상이면 전체 OCR
    needs_full = len(ocr_pages) >= len(pages) * 0.5
    
    return needs_full, ([] if needs_full else ocr_pages), {'mode': 'legacy'}


def _needs_ocr_for_page_legacy(text: str, min_chars: int) -> bool:
    """
    기존 페이지별 OCR 필요 여부 판단 (하위 호환)
    """
    if not text or len(text.strip()) < min_chars:
        return True
    
    # CID 패턴 감지
    cid_threshold = float(os.getenv("OCR_CID_THRESHOLD", "0.15"))
    cid_pattern = re.compile(r'\(cid:\d+\)')
    cid_matches = cid_pattern.findall(text)
    cid_char_count = sum(len(m) for m in cid_matches)
    
    if len(text) > 0:
        cid_ratio = cid_char_count / len(text)
        if cid_ratio > cid_threshold:
            return True
    
    # 읽을 수 있는 문자 비율 체크
    min_readable_ratio = float(os.getenv("OCR_MIN_READABLE_RATIO", "0.3"))
    readable_chars = len(re.findall(r'[a-zA-Z0-9가-힣]', text))
    total_chars = len(text)
    
    if total_chars > 0:
        readable_ratio = readable_chars / total_chars
        if readable_ratio < min_readable_ratio:
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


# ---------- 워터마크 필터링 ----------
def _filter_watermark_blocks(blocks: List[Dict], page_width: int, page_height: int) -> List[Dict]:
    """
    워터마크 블록 필터링
    - 중앙 배치 + 대각선 + 반투명 텍스트
    """
    if not blocks:
        return blocks
    
    filtered = []
    center_x = page_width / 2
    center_y = page_height / 2
    
    for block in blocks:
        bbox = block.get('bbox', {})
        if not isinstance(bbox, dict):
            filtered.append(block)
            continue
        
        x0, y0 = bbox.get('x0', 0), bbox.get('y0', 0)
        x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
        
        block_center_x = (x0 + x1) / 2
        block_center_y = (y0 + y1) / 2
        
        # 중앙 근처 + 회전 각도 체크
        dist_from_center = ((block_center_x - center_x)**2 + (block_center_y - center_y)**2)**0.5
        
        if dist_from_center < min(page_width, page_height) * 0.3:
            # 대각선 배치 체크
            width = x1 - x0
            height = y1 - y0
            if width > page_width * 0.5 or height > page_height * 0.5:
                # 워터마크 가능성
                text = block.get('text', '').strip().lower()
                if any(keyword in text for keyword in ['confidential', 'draft', 'copy', '사본', '기밀']):
                    print(f"[OCR] Filtered watermark: {text[:30]}")
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
    [SmartOCRRouter 통합] PDF 융합 추출 (경로)
    
    반환:
      pages: [(page_no, text)]
      layout_map: {page_no: [ {"text":..., "bbox":{...}, "confidence":...}, ... ] }
    
    OCR 전략:
      - OCR_MODE=off → pdfminer 결과 그대로
      - OCR_MODE=force → 모든 페이지를 OCR로 대체
      - OCR_MODE=auto → SmartOCRRouter 기반 지능형 판단
    """
    import fitz
    
    # 1단계: pdfminer로 기본 추출
    pages, layout_map = _pdfminer_pages_and_blocks_from_path(path)
    
    ocr_mode = os.getenv("OCR_MODE", "auto").strip().lower()  # off|auto|force
    
    if ocr_mode == "off":
        print("[OCR] OCR disabled by OCR_MODE=off")
        return pages, layout_map
    
    # 2단계: SmartOCRRouter로 OCR 결정
    needs_full_ocr, partial_ocr_pages, analysis = _get_smart_ocr_decision(
        pages, layout_map, ocr_mode
    )
    
    # 3단계: OCR 실행
    doc = fitz.open(path)
    try:
        if ocr_mode == "force" or needs_full_ocr:
            # 전체 OCR
            print(f"[OCR] Performing full OCR on all {len(pages)} pages")
            for (idx, text), pg in zip(pages, doc, strict=False):
                txt, blocks = _ocr_page_image(pg)
                pages[idx-1] = (idx, txt or "")
                layout_map[idx] = blocks or []
        
        elif partial_ocr_pages:
            # 부분 OCR
            print(f"[OCR] Performing partial OCR on {len(partial_ocr_pages)} pages: {partial_ocr_pages}")
            for page_no in partial_ocr_pages:
                if 1 <= page_no <= len(pages):
                    pg = doc[page_no - 1]
                    txt, blocks = _ocr_page_image(pg)
                    pages[page_no - 1] = (page_no, txt or "")
                    layout_map[page_no] = blocks or []
        else:
            print("[OCR] No OCR needed based on quality analysis")
    
    finally:
        doc.close()
    
    return pages, layout_map


def extract_pdf_fused_from_bytes(pdf_bytes: bytes) -> Tuple[List[Tuple[int,str]], Dict[int, List[Dict]]]:
    """
    [SmartOCRRouter 통합] PDF 융합 추출 (바이트)
    """
    import fitz
    
    # 1단계: pdfminer로 기본 추출
    pages, layout_map = _pdfminer_pages_and_blocks_from_bytes(pdf_bytes)
    
    ocr_mode = os.getenv("OCR_MODE", "auto").strip().lower()
    
    if ocr_mode == "off":
        print("[OCR] OCR disabled by OCR_MODE=off")
        return pages, layout_map
    
    # 2단계: SmartOCRRouter로 OCR 결정
    needs_full_ocr, partial_ocr_pages, analysis = _get_smart_ocr_decision(
        pages, layout_map, ocr_mode
    )
    
    # 3단계: OCR 실행
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if ocr_mode == "force" or needs_full_ocr:
            # 전체 OCR
            print(f"[OCR] Performing full OCR on all {len(pages)} pages")
            for i, pg in enumerate(doc, start=1):
                txt, blocks = _ocr_page_image(pg)
                if i-1 < len(pages):
                    pages[i-1] = (i, txt or "")
                else:
                    pages.append((i, txt or ""))
                layout_map[i] = blocks or []
        
        elif partial_ocr_pages:
            # 부분 OCR
            print(f"[OCR] Performing partial OCR on {len(partial_ocr_pages)} pages: {partial_ocr_pages}")
            for page_no in partial_ocr_pages:
                if 1 <= page_no <= len(doc):
                    pg = doc[page_no - 1]
                    txt, blocks = _ocr_page_image(pg)
                    if page_no - 1 < len(pages):
                        pages[page_no - 1] = (page_no, txt or "")
                    else:
                        pages.append((page_no, txt or ""))
                    layout_map[page_no] = blocks or []
        else:
            print("[OCR] No OCR needed based on quality analysis")
    
    finally:
        doc.close()
    
    return pages, layout_map