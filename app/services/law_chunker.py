from __future__ import annotations
import os, re
from typing import List, Tuple, Dict, Any, Callable, Optional, Set

# ------------------------- Title/structure patterns -------------------------
RE_CHAPTER   = re.compile(r"^\s*(제\s*\d+\s*장|chapter\s+\d+)\b", re.I)
RE_SECTION   = re.compile(r"^\s*(제\s*\d+\s*절|section\s+\d+)\b", re.I)
RE_ARTICLE   = re.compile(r"^\s*(제\s*\d+\s*조(\s*\(.+?\))?)\b", re.I)           # 제12조(정의)
RE_CLAUSE    = re.compile(r"^\s*((?:\(\d+\))|(?:\d+\.)|[①-⑳])\s*")
RE_SUBITEM   = re.compile(r"^\s*([가-하]\)|[a-z]\))\s*")
RE_TABLELIKE = re.compile(r"(\|.+\|)|(\t)|( {2,}\S+ {2,}\S+)")

# ------------------------- Header/footer helpers ----------------------------
RE_GUTTER_INT = re.compile(r"^\s*\d{1,3}\s*$")
RE_TOC_LINE   = re.compile(r"^\s*\d+(?:\.\d+)*\s+.+\s+\d{1,3}\s*$")
RE_TOC_TITLE  = re.compile(r"(목차|contents?)", re.I)
RE_OUTLINE    = re.compile(r"^\s*\d+(?:\.\d+){0,3}\.?(\))?\s*")
RE_HEADING_HINT = re.compile(r"(개요|목적|정의|범위|서론|서문|적용|규정|요건|원칙|"
                             r"scope|purpose|definition|overview|FSE|MACE)", re.I)
# 두 줄 헤더의 2번째 줄: ': 입문서', '— Appendix' 등
RE_CONT_HEAD  = re.compile(r"^\s*[:：\-–—]\s*\S")

# ------------------------- Normalizers -------------------------------------
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

def _normalize_fuzzy(s: str) -> str:
    """조금 더 공격적인 유사도용 정규화(공백/구두점 축소, 소문자)"""
    t = _normalize_header(s)
    t = re.sub(r"[\s\p{P}]+", "", t, flags=re.UNICODE)
    return t.lower()

def _merge_header_continuations(lines: list[str], *, max_prev_len: int = 64) -> list[str]:
    """
    상하단 헤더가 2줄(가끔 중간에 빈 줄 포함)로 나뉜 경우 하나로 병합.
    ex)
      다른 국가와의 원자력 협력
      :   입문서
    -> '다른 국가와의 원자력 협력: 입문서'
    """
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

# ------------------------- Token helpers -----------------------------------
def _tail_by_tokens(enc: Callable[[str], List[int]], text: str, want: int) -> str:
    if want <= 0 or not text: return ""
    toks = enc(text) or []
    if len(toks) <= want: return text
    lines = [l for l in text.splitlines()]
    acc = []; total = 0
    for l in reversed(lines):
        acc.append(l)
        total = len(enc("\n".join(reversed(acc))))
        if total >= want: break
    return "\n".join(reversed(acc)).strip()

def _pack_tokens(enc, parts, target, overlap, min_tokens, section_title):
    out = []; cur = []; cur_pages = []; cur_tok = 0
    def flush(force=False):
        nonlocal cur, cur_pages, cur_tok
        if not cur: return
        text = "\n".join(cur).strip()
        toks = enc(text) or []
        if not force and len(toks) < min_tokens and out:
            last_text, last_meta = out[-1]
            out[-1] = ((last_text + "\n" + text).strip(), last_meta)
        else:
            out.append((text, {
                "page": min(cur_pages) if cur_pages else 0,
                "pages": sorted(set(cur_pages)) if cur_pages else [],
                "section": section_title[:160],
                "bboxes": {},
            }))
        if overlap > 0:
            keep = _tail_by_tokens(enc, text, overlap)
            cur = [keep] if keep else []; cur_pages = list(sorted(set(cur_pages))) if cur else []; cur_tok = len(enc(keep) or []) if keep else 0
        else:
            cur, cur_pages, cur_tok = [], [], 0
    for t, pno in parts:
        toks = enc(t) or []
        if cur_tok + len(toks) <= max(target, min_tokens):
            cur.append(t); cur_pages.append(int(pno) if pno else 0); cur_tok += len(toks)
        else:
            flush(force=True); cur = [t]; cur_pages = [int(pno) if pno else 0]; cur_tok = len(toks)
    flush(force=True); return out

# ------------------------- Repeating detectors ------------------------------
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

def _collect_repeating_headers(pages_std, top_k=4, bottom_k=4, min_repeat: Optional[int]=None) -> Set[str]:
    freq: Dict[str,int] = {}; n_pages = 0
    for _, txt in (pages_std or []):
        lines = [l.strip() for l in (txt or "").splitlines() if l and l.strip()]
        if not lines: continue
        n_pages += 1
        lines = _merge_header_continuations(lines ,max_prev_len=90)
        tops = lines[:max(0, top_k)]
        bots = lines[-max(0, bottom_k):] if bottom_k>0 else []
        for c in [*tops, *bots]:
            norm = _normalize_header(c)
            if norm: freq[norm] = freq.get(norm,0)+1
    if n_pages == 0: return set()
    if min_repeat is None: min_repeat = max(3, n_pages//5)
    return {line for line,cnt in freq.items() if cnt >= min_repeat}

def _collect_repeating_windows_global(pages_std, window_sizes=(2,3), min_repeat=3, min_len=3, max_len=120) -> Set[str]:
    """
    페이지 경계가 없어도 잘 잡히도록, 2~3줄 묶음 슬라이딩 윈도우를 문서 전역에서 집계.
    """
    freq: Dict[str,int] = {}
    for _, txt in (pages_std or []):
        raw = [l for l in (txt or "").splitlines()]
        lines = [l.strip() for l in raw if l and l.strip()]
        lines = _merge_header_continuations(lines, max_prev_len=90)
        n = len(lines)
        for w in window_sizes:
            if n < w: continue
            for i in range(0, n-w+1):
                block = " ".join(lines[i:i+w]).strip()
                bnorm = _normalize_header(block)
                if not bnorm: continue
                if not (min_len <= len(bnorm) <= max_len): continue
                # 숫자만, 표 행 등은 스킵
                if RE_GUTTER_INT.match(bnorm): continue
                freq[bnorm] = freq.get(bnorm,0)+1
    return {k for k,v in freq.items() if v >= min_repeat}

# ------------------------- Main chunker -------------------------------------
def law_chunk_pages(
    pages_std: List[Tuple[int, str]],
    enc: Callable[[str], List[int]],
    target_tokens: int = 512,
    overlap_tokens: int = 96,
    *,
    layout_blocks: Optional[Dict[int, List[Dict]]] = None,
    min_chunk_tokens: int = 100,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    - 러닝 헤더/푸터: 라인/2~3줄 윈도우 전역 빈도 기반 자동 제거
    - 2줄 헤더 병합(빈 줄 끼어도 병합)
    - 조문/항/목 유지, 표는 통청크
    - 너무 작은 청크는 앞 청크와 병합
    """
    layout_blocks = layout_blocks or {}
    chunks: List[Tuple[str, Dict[str, Any]]] = []

    # A) 반복 헤더/푸터 후보 수집(페이지 상하단 라인)
    min_repeat_env = os.getenv("LAW_MIN_HEADER_REPEAT")
    min_repeat = int(min_repeat_env) if (min_repeat_env or "").isdigit() else None
    repeating_lines = _collect_repeating_headers(pages_std, top_k=int(os.getenv("LAW_HEADER_TOP_K","4")),
                                                 bottom_k=int(os.getenv("LAW_HEADER_BOTTOM_K","4")),
                                                 min_repeat=min_repeat)
    repeating_lines_norm = {_normalize_header(s) for s in repeating_lines}

    # B) 전역 슬라이딩 윈도우(2~3줄) 반복 헤더/푸터
    global_blocks = _collect_repeating_windows_global(
        pages_std,
        window_sizes=(2,3),
        min_repeat=int(os.getenv("LAW_GLOBAL_MIN_REPEAT","3")),
        min_len=int(os.getenv("LAW_GLOBAL_MINLEN","3")),
        max_len=int(os.getenv("LAW_GLOBAL_MAXLEN","120")),
    )

    cur_section = ""; cur_article = ""; buf: List[Tuple[str,int]] = []

    def flush_article():
        nonlocal buf, cur_section, cur_article, chunks
        if not buf: return
        section = (cur_section + (" | " if cur_section and cur_article else "") + cur_article).strip()
        chunks.extend(_pack_tokens(enc, buf, target_tokens, overlap_tokens, min_chunk_tokens, section))
        buf = []

    for page_no, page_text in (pages_std or []):
        text = (page_text or "").strip()
        if not text: continue

        if _is_table_like(text):
            flush_article()
            sec = (cur_section + (f" | {cur_article}" if cur_article else "")).strip() or "표"
            chunks.append((text, {"page": int(page_no), "pages":[int(page_no)], "section": sec, "type":"table", "bboxes":{}}))
            continue

        # --- per-line cleanup ---
        lines = [l.strip() for l in (text.splitlines()) if l and l.strip()]
        lines = _remove_left_gutter_numbers(lines)
        lines = _merge_header_continuations(lines, max_prev_len=90)

        # 1) 2~3줄 블록 단위 제거(슬라이딩 윈도우)
        i = 0; cleaned: List[str] = []
        while i < len(lines):
            # try 3줄, then 2줄
            matched = False
            for w in (3,2):
                if i + w <= len(lines):
                    block = " ".join(lines[i:i+w]).strip()
                    if _normalize_header(block) in global_blocks:
                        i += w; matched = True; break
            if matched: continue
            # 개별 라인 반복 제거
            ln = lines[i]
            if _normalize_header(ln) in repeating_lines_norm:
                i += 1; continue
            cleaned.append(ln); i += 1
        lines = cleaned

        # TOC?
        if _is_toc_page(lines):
            flush_article()
            chunks.append(("\n".join(lines), {"page": int(page_no), "pages":[int(page_no)], "section":"목차", "type":"toc", "bboxes":{}}))
            continue

        # body
        for ln in lines:
            if _is_heading(ln) or _is_outline_heading(ln):
                flush_article(); cur_section = ln.strip(); continue
            if _is_article(ln):
                flush_article(); cur_article = ln.strip(); buf.append((ln.strip(), int(page_no))); continue
            buf.append((ln.strip(), int(page_no)))

    flush_article()

    if not chunks:
        big = [(t.strip(), int(p)) for p,t in (pages_std or []) if t and t.strip()]
        if big: chunks = _pack_tokens(enc, big, target_tokens, overlap_tokens, min_chunk_tokens, "")

    # normalize & dedup
    final = []
    last = None
    for text, meta in chunks:
        meta = dict(meta or {})
        pg = meta.get("page") or (meta.get("pages")[0] if meta.get("pages") else 0)
        norm_meta = {"page": int(pg) if pg else 0,
                     "pages": meta.get("pages") or ([int(pg)] if pg else []),
                     "section": (meta.get("section") or "")[:160],
                     "bboxes": meta.get("bboxes") or {}}
        item = (text, norm_meta)
        if text and item != last:
            final.append(item); last = item
    return final
