from __future__ import annotations
import os, re
from typing import List, Tuple, Dict, Any, Callable, Optional, Set

# ------------------------- 동적 워터마크/헤더 감지 -------------------------
# 기본 패턴들 (최소한만 유지)
RE_PAGE_NUMBER = re.compile(r'^\s*(\d+\s*/\s*\d+|페이지\s*\d+|page\s*\d+)\s*$', re.I)
RE_TABLE_BORDER = re.compile(r'^[\+\-\|\s]*$')  # 테이블 경계선
RE_REPEATED_SYMBOLS = re.compile(r'^[=\-_\*\+]{3,}$')  # 반복되는 구분선
RE_SHORT_CAPS = re.compile(r'^[A-Z\s]{2,20}$')  # 짧은 대문자 텍스트 (헤더 후보)

# 조문/섹션 패턴 강화
RE_CHAPTER = re.compile(r"^\s*(제\s*\d+\s*장|chapter\s+\d+|섹션\s*\d+)\b", re.I)
RE_SECTION = re.compile(r"^\s*(제\s*\d+\s*절|section\s+\d+\.\d+|제\s*\d+조의\d+)\b", re.I)
RE_ARTICLE = re.compile(r"^\s*(제\s*\d+\s*조(\s*\(.+?\))?|article\s+\d+|§\s*\d+)\b", re.I)
RE_CLAUSE = re.compile(r"^\s*((?:\(\d+\))|(?:\d+\.)|[①-⑳]|[가-하]\.)\s*")
RE_SUBITEM = re.compile(r"^\s*([가-하]\)|[a-z]\)|[ⅰ-ⅹ]\))\s*")

# FSE/MACE 패턴 (원자력 규제 문서 특화)
RE_FSE_MACE = re.compile(r"^\s*(FSE\s*\d+|MACE\s*\d+\.\d+)", re.I)
RE_NUCLEAR_TERMS = re.compile(r"(NMACS|IAEA|원자력규제|Nuclear\s+Regulatory|safeguards)", re.I)

# 기존 코드에서 가져온 패턴들
RE_GUTTER_INT = re.compile(r"^\s*\d{1,3}\s*$")
RE_TOC_LINE = re.compile(r"^\s*\d+(?:\.\d+)*\s+.+\s+\d{1,3}\s*$")
RE_TOC_TITLE = re.compile(r"(목차|contents?)", re.I)
RE_OUTLINE = re.compile(r"^\s*\d+(?:\.\d+){0,3}\.?(\))?\s*")
RE_HEADING_HINT = re.compile(r"(개요|목적|정의|범위|서론|서문|적용|규정|요건|원칙|"
                             r"scope|purpose|definition|overview|FSE|MACE)", re.I)
RE_CONT_HEAD = re.compile(r"^\s*[:：\-–—]\s*\S")
RE_TABLELIKE = re.compile(r"(\|.+\|)|(\t)|( {2,}\S+ {2,}\S+)")

# ------------------------- 기존 함수들 유지 -------------------------
BULLETS = "•●○∙·◇◆■□▪▫※▶▷▶︎•‣‒–—・"

def _strip_bullets(s: str) -> str:
    return s.translate({ord(ch): None for ch in BULLETS})

def _normalize_header(s: str) -> str:
    """반복 라인 판정을 위한 보수적 정규화"""
    if not s: return ""
    t = _strip_bullets(s).strip()
    t = re.sub(r"\s*[:：]\s*", ": ", t)      # 콜론 주변 공백 정리
    t = re.sub(r"\s{2,}", " ", t)           # 다중 공백 축소
    t = re.sub(r"\s+\d{1,3}$", "", t)       # 말미 페이지수 제거(약하게)
    return t

def _merge_header_continuations(lines: list[str], *, max_prev_len: int = 64) -> list[str]:
    """상하단 헤더가 2줄로 나뉜 경우 하나로 병합"""
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = (lines[i] or "").strip()
        j = i + 1
        # 빈 줄 1개까지 허용
        if j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            nxt = (lines[j] or "").strip()
            if RE_CONT_HEAD.match(nxt) and 1 <= len(cur) <= max_prev_len:
                cont = re.sub(r"^\s*[:：\-–—]\s*", "", nxt).strip()
                out.append(f"{cur}: {cont}")
                i = j + 1
                continue
        out.append(cur); i += 1
    return out

def _remove_left_gutter_numbers(lines: list[str]) -> list[str]:
    if not lines: return lines
    ints = sum(1 for l in lines if RE_GUTTER_INT.match(l))
    if ints >= max(3, len(lines)//5):               # 20%+
        return [l for l in lines if not RE_GUTTER_INT.match(l)]
    out = []; i = 0
    while i < len(lines):
        l = lines[i]
        if RE_GUTTER_INT.match(l) and (i+1) < len(lines):
            nxt = lines[i+1]
            if nxt.strip():
                out.append(f"{l.strip()} {nxt.strip()}"); i += 2; continue
        out.append(l); i += 1
    return out

def _is_toc_page(lines: list[str]) -> bool:
    if not lines: return False
    head = " ".join(lines[:5]).lower()
    if RE_TOC_TITLE.search(head):
        toc_like = sum(1 for l in lines if RE_TOC_LINE.match(l))
        return toc_like >= max(5, len(lines)//4)
    return False

def _is_outline_heading(line: str) -> bool:
    s = line.strip(); m = RE_OUTLINE.match(s)
    if not m: return False
    rest = RE_OUTLINE.sub("", s).strip()
    return bool(RE_HEADING_HINT.search(rest)) and (8 <= len(rest) <= 80)

def _is_heading(line: str) -> bool:
    """헤딩인지 판단"""
    return bool(RE_CHAPTER.match(line) or RE_SECTION.match(line))

def _is_article(line: str) -> bool:
    """조문인지 판단"""
    return bool(RE_ARTICLE.match(line) or RE_FSE_MACE.match(line))

def _is_table_like(text: str) -> bool:
    """테이블 콘텐츠 판단"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    
    table_indicators = 0
    for line in lines:
        if '|' in line and line.count('|') >= 2:
            table_indicators += 1
        elif re.search(r'\t|\s{3,}', line):  # 탭이나 3개 이상 공백
            table_indicators += 1
        elif RE_TABLE_BORDER.match(line):
            table_indicators += 1
    
    return table_indicators >= max(2, len(lines) // 3)

# ------------------------- OCR 기반 워터마크 감지 -------------------------
def _extract_watermarks_from_ocr_data(layout_blocks: Dict[int, List[Dict]]) -> Set[str]:
    """OCR 블록 데이터에서 워터마크 텍스트 추출"""
    watermarks = set()
    
    if not layout_blocks:
        return watermarks
    
    # 각 페이지별로 처리
    all_texts_by_position = []
    
    for page_no, blocks in layout_blocks.items():
        page_texts = []
        for block in blocks:
            text = block.get('text', '').strip()
            bbox = block.get('bbox', [0, 0, 0, 0])
            
            if not text or len(text) < 3:
                continue
                
            # bbox에서 위치 정보 계산 (상대적 위치)
            if len(bbox) >= 4:
                x0, y0, x1, y1 = bbox[:4]
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2
                width = x1 - x0
                height = y1 - y0
                
                page_texts.append({
                    'text': text,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'area': width * height,
                    'normalized_text': _normalize_text_for_comparison(text)
                })
        
        all_texts_by_position.append(page_texts)
    
    if len(all_texts_by_position) < 2:
        return watermarks
    
    # 반복되는 텍스트 찾기
    text_counts = {}
    position_groups = {}
    
    for page_texts in all_texts_by_position:
        for item in page_texts:
            norm_text = item['normalized_text']
            if len(norm_text) < 3:
                continue
                
            # 텍스트 빈도 계산
            text_counts[norm_text] = text_counts.get(norm_text, 0) + 1
            
            # 위치 그룹화 (상단/중앙/하단)
            y_pos = item['center_y']
            pos_key = 'top' if y_pos < 100 else 'bottom' if y_pos > 700 else 'middle'
            
            if norm_text not in position_groups:
                position_groups[norm_text] = {'top': 0, 'middle': 0, 'bottom': 0}
            position_groups[norm_text][pos_key] += 1
    
    # 워터마크 기준:
    # 1. 최소 3페이지 이상 반복
    # 2. 주로 상단 또는 하단에 위치
    # 3. 텍스트 길이가 적절함 (3-50자)
    min_repeat = max(3, len(all_texts_by_position) // 3)
    
    for norm_text, count in text_counts.items():
        if count >= min_repeat:
            pos_info = position_groups.get(norm_text, {})
            top_count = pos_info.get('top', 0)
            bottom_count = pos_info.get('bottom', 0)
            middle_count = pos_info.get('middle', 0)
            
            # 상단 또는 하단에 집중된 텍스트
            if (top_count + bottom_count) > middle_count:
                # 원본 텍스트 찾아서 추가
                for page_texts in all_texts_by_position:
                    for item in page_texts:
                        if item['normalized_text'] == norm_text:
                            watermarks.add(item['text'])
                            break
    
    return watermarks

def _normalize_text_for_comparison(text: str) -> str:
    """텍스트 비교를 위한 정규화"""
    # 공백, 구두점 정규화하고 소문자 변환
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s가-힣]', '', text)
    return text.lower()

def _is_likely_watermark(text: str, watermark_patterns: Set[str]) -> bool:
    """텍스트가 워터마크일 가능성 판단"""
    text = text.strip()
    
    if not text or len(text) < 3:
        return True
    
    # OCR에서 추출된 워터마크 패턴과 비교
    normalized = _normalize_text_for_comparison(text)
    for pattern in watermark_patterns:
        pattern_norm = _normalize_text_for_comparison(pattern)
        if pattern_norm and normalized == pattern_norm:
            return True
    
    # 기본 패턴들
    if RE_PAGE_NUMBER.match(text):
        return True
    if RE_TABLE_BORDER.match(text) or RE_REPEATED_SYMBOLS.match(text):
        return True
    if len(text) <= 50 and RE_SHORT_CAPS.match(text):
        return True
    
    # 매우 짧거나 긴 텍스트
    if len(text) < 3 or len(text) > 200:
        return True
        
    return False

# ------------------------- 개선된 반복 헤더 수집 (OCR 연동) -------------------------
def _collect_repeating_patterns_with_ocr(
    pages_std, 
    layout_blocks: Dict[int, List[Dict]], 
    top_k=3, 
    bottom_k=3
) -> Set[str]:
    """OCR 데이터와 텍스트 데이터를 결합하여 워터마크/반복 패턴 감지"""
    
    # 1. OCR 데이터에서 워터마크 추출
    ocr_watermarks = _extract_watermarks_from_ocr_data(layout_blocks)
    
    # 2. 텍스트에서 반복 패턴 수집 (기존 로직 간소화)
    text_patterns = set()
    freq: Dict[str, int] = {}
    n_pages = 0
    
    for _, txt in (pages_std or []):
        lines = [l.strip() for l in (txt or "").splitlines() if l and l.strip()]
        if not lines:
            continue
            
        n_pages += 1
        
        # 상단/하단 라인들만 검사
        tops = lines[:top_k] if len(lines) > top_k else []
        bots = lines[-bottom_k:] if len(lines) > bottom_k and bottom_k > 0 else []
        
        for line in [*tops, *bots]:
            normalized = _normalize_text_for_comparison(line)
            if len(normalized) >= 3:  # 최소 길이만 체크
                freq[normalized] = freq.get(normalized, 0) + 1
    
    if n_pages >= 3:
        min_repeat = max(2, n_pages // 4)
        for norm_text, count in freq.items():
            if count >= min_repeat:
                # 원본 텍스트 찾아서 추가
                for _, txt in pages_std:
                    for line in txt.splitlines():
                        if _normalize_text_for_comparison(line.strip()) == norm_text:
                            text_patterns.add(line.strip())
                            break
    
    # 3. OCR 워터마크와 텍스트 패턴 결합
    all_patterns = ocr_watermarks.union(text_patterns)
    
    print(f"[WATERMARK] OCR patterns: {len(ocr_watermarks)}, Text patterns: {len(text_patterns)}, Total: {len(all_patterns)}")
    
    return all_patterns

# ------------------------- 청킹 크기 조절 함수 -------------------------
def _adjust_chunk_size_by_content(text: str, base_tokens: int) -> int:
    """내용에 따라 청크 크기 동적 조절"""
    # FSE/MACE 등 중요 섹션은 더 큰 청크
    if RE_FSE_MACE.search(text):
        return int(base_tokens * 1.5)
    
    # 조문/조항은 적당한 크기
    if RE_ARTICLE.search(text) or RE_SECTION.search(text):
        return int(base_tokens * 1.2)
    
    # 일반 텍스트
    return base_tokens

def _should_merge_chunks(chunk1: str, chunk2: str, min_chunk_size: int) -> bool:
    """청크 병합 여부 결정"""
    # 둘 다 매우 짧으면 병합
    if len(chunk1.split()) < min_chunk_size // 2 and len(chunk2.split()) < min_chunk_size // 2:
        return True
    
    # 첫 번째가 헤더/제목 형태면 병합
    if len(chunk1.split()) < 20 and (RE_CHAPTER.match(chunk1) or RE_SECTION.match(chunk1)):
        return True
    
    return False

def _tail_by_tokens(enc: Callable[[str], List[int]], text: str, want: int) -> str:
    """텍스트 끝부분에서 지정된 토큰 수만큼 가져오기"""
    if want <= 0 or not text:
        return ""
    
    toks = enc(text) or []
    if len(toks) <= want:
        return text
    
    lines = [l for l in text.splitlines()]
    acc = []
    total = 0
    
    for l in reversed(lines):
        acc.append(l)
        total = len(enc("\n".join(reversed(acc))))
        if total >= want:
            break
    
    return "\n".join(reversed(acc)).strip()

def _pack_tokens_enhanced(enc, parts, target, overlap, min_tokens, section_title):
    """개선된 토큰 패킹"""
    out = []
    cur = []
    cur_pages = []
    cur_tok = 0
    
    def flush(force=False):
        nonlocal cur, cur_pages, cur_tok
        if not cur:
            return
            
        text = "\n".join(cur).strip()
        toks = enc(text) or []
        
        if not force and len(toks) < min_tokens and out:
            # 이전 청크와 병합
            last_text, last_meta = out[-1]
            combined_text = last_text + "\n\n" + text
            combined_pages = sorted(set(last_meta.get("pages", []) + cur_pages))
            
            out[-1] = (combined_text, {
                "page": min(combined_pages) if combined_pages else 0,
                "pages": combined_pages,
                "section": section_title[:160],
                "bboxes": {},
            })
        else:
            out.append((text, {
                "page": min(cur_pages) if cur_pages else 0,
                "pages": sorted(set(cur_pages)) if cur_pages else [],
                "section": section_title[:160],
                "bboxes": {},
            }))
        
        # 오버랩 처리
        if overlap > 0 and len(toks) > overlap:
            keep_text = _tail_by_tokens(enc, text, overlap)
            cur = [keep_text] if keep_text else []
            cur_pages = list(sorted(set(cur_pages))) if cur else []
            cur_tok = len(enc(keep_text) or []) if keep_text else 0
        else:
            cur, cur_pages, cur_tok = [], [], 0
    
    for t, pno in parts:
        if not t.strip():
            continue
            
        toks = enc(t) or []
        if cur_tok + len(toks) <= target:
            cur.append(t)
            cur_pages.append(int(pno) if pno else 0)
            cur_tok += len(toks)
        else:
            flush(force=True)
            cur = [t]
            cur_pages = [int(pno) if pno else 0]
            cur_tok = len(toks)
    
    flush(force=True)
    return out

# ------------------------- 메인 청킹 함수 -------------------------
def law_chunk_pages_enhanced(
    pages_std: List[Tuple[int, str]],
    enc: Callable[[str], List[int]],
    target_tokens: int = 512,
    overlap_tokens: int = 96,
    *,
    layout_blocks: Optional[Dict[int, List[Dict]]] = None,
    min_chunk_tokens: int = 150,
) -> List[Tuple[str, Dict[str, Any]]]:
    """OCR 데이터를 활용한 개선된 법령 청킹"""
    layout_blocks = layout_blocks or {}
    chunks: List[Tuple[str, Dict[str, Any]]] = []

    # OCR 데이터와 텍스트를 결합한 워터마크/반복 패턴 감지
    detected_patterns = _collect_repeating_patterns_with_ocr(
        pages_std, 
        layout_blocks,
        top_k=int(os.getenv("LAW_HEADER_TOP_K", "3")),
        bottom_k=int(os.getenv("LAW_HEADER_BOTTOM_K", "3"))
    )

    cur_section = ""
    cur_article = ""
    buf: List[Tuple[str, int]] = []

    def flush_section():
        nonlocal buf, cur_section, cur_article, chunks
        if not buf:
            return
            
        section_title = (cur_section + (" | " if cur_section and cur_article else "") + cur_article).strip()
        
        # 동적 청크 크기 조절
        combined_text = "\n".join([text for text, _ in buf])
        adjusted_target = _adjust_chunk_size_by_content(combined_text, target_tokens)
        
        section_chunks = _pack_tokens_enhanced(
            enc, buf, adjusted_target, overlap_tokens, min_chunk_tokens, section_title
        )
        chunks.extend(section_chunks)
        buf = []

    for page_no, page_text in (pages_std or []):
        text = (page_text or "").strip()
        if not text:
            continue

        # 테이블 처리
        if _is_table_like(text):
            flush_section()
            sec = (cur_section + (f" | {cur_article}" if cur_article else "")).strip() or "표"
            chunks.append((text, {
                "page": int(page_no), 
                "pages": [int(page_no)], 
                "section": sec, 
                "type": "table", 
                "bboxes": {}
            }))
            continue

        # 라인별 처리 - 감지된 워터마크 패턴 사용
        lines = [l.strip() for l in text.splitlines() if l and l.strip()]
        
        # 기존 전처리
        lines = _remove_left_gutter_numbers(lines)
        lines = _merge_header_continuations(lines, max_prev_len=90)
        
        # 워터마크/헤더 제거 (동적 패턴 기반)
        cleaned_lines = []
        for line in lines:
            if _is_likely_watermark(line, detected_patterns):
                continue
            cleaned_lines.append(line)
        
        if not cleaned_lines:
            continue

        # TOC 처리
        if _is_toc_page(cleaned_lines):
            flush_section()
            chunks.append(("\n".join(cleaned_lines), {
                "page": int(page_no), 
                "pages": [int(page_no)], 
                "section": "목차", 
                "type": "toc", 
                "bboxes": {}
            }))
            continue

        # 구조 분석 및 청킹
        for line in cleaned_lines:
            if _is_heading(line) or _is_outline_heading(line):
                flush_section()
                cur_section = line.strip()
                continue
            
            if _is_article(line):
                flush_section()
                cur_article = line.strip()
                buf.append((line.strip(), int(page_no)))
                continue
                
            buf.append((line.strip(), int(page_no)))

    flush_section()

    # 빈 결과인 경우 폴백
    if not chunks:
        big = [(t.strip(), int(p)) for p, t in (pages_std or []) if t and t.strip()]
        if big:
            chunks = _pack_tokens_enhanced(enc, big, target_tokens, overlap_tokens, min_chunk_tokens, "")

    # 후처리: 너무 작은 청크들 병합
    final_chunks = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        
        # 다음 청크와 병합 가능한지 확인
        if (i + 1 < len(chunks) and 
            _should_merge_chunks(current[0], chunks[i+1][0], min_chunk_tokens)):
            
            next_chunk = chunks[i+1]
            merged_text = current[0] + "\n\n" + next_chunk[0]
            merged_meta = current[1].copy()
            merged_meta["pages"] = sorted(set(
                merged_meta.get("pages", []) + next_chunk[1].get("pages", [])
            ))
            
            final_chunks.append((merged_text, merged_meta))
            i += 2  # 두 청크를 모두 처리했으므로
        else:
            final_chunks.append(current)
            i += 1

    # 최종 정규화 및 중복 제거
    final = []
    last = None
    for text, meta in final_chunks:
        meta = dict(meta or {})
        pg = meta.get("page") or (meta.get("pages")[0] if meta.get("pages") else 0)
        norm_meta = {
            "page": int(pg) if pg else 0,
            "pages": meta.get("pages") or ([int(pg)] if pg else []),
            "section": (meta.get("section") or "")[:160],
            "bboxes": meta.get("bboxes") or {}
        }
        item = (text, norm_meta)
        if text and item != last:
            final.append(item)
            last = item
    
    return final

# 기존 함수명도 유지 (하위 호환성)
def law_chunk_pages(
    pages_std: List[Tuple[int, str]],
    enc: Callable[[str], List[int]],
    target_tokens: int = 512,
    overlap_tokens: int = 96,
    *,
    layout_blocks: Optional[Dict[int, List[Dict]]] = None,
    min_chunk_tokens: int = 100,
) -> List[Tuple[str, Dict[str, Any]]]:
    """기존 법령 청킹 (하위 호환성 유지)"""
    return law_chunk_pages_enhanced(
        pages_std, enc, target_tokens, overlap_tokens,
        layout_blocks=layout_blocks, min_chunk_tokens=min_chunk_tokens
    )