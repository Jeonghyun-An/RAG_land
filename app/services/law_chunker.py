from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any, Callable, Optional

# =========================
# Regex patterns
# =========================
RE_CHAPTER   = re.compile(r"^\s*(제\s*\d+\s*장|chapter\s+\d+)\b", re.I)
RE_SECTION   = re.compile(r"^\s*(제\s*\d+\s*절|section\s+\d+)\b", re.I)
RE_ARTICLE   = re.compile(r"^\s*(제\s*\d+\s*조(\s*\(.+?\))?)\b", re.I)   # e.g., 제12조(정의)
RE_CLAUSE    = re.compile(r"^\s*((?:\(\d+\))|(?:\d+\.)|[①-⑳])\s*")      # (1), 1., ① …
RE_SUBITEM   = re.compile(r"^\s*([가-하]\)|[a-z]\))\s*")                 # 가), a) …
RE_TABLELIKE = re.compile(r"(\|.+\|)|(\t)|( {2,}\S+ {2,}\S+)")           # '|' or tabs or aligned spaces

# Gutter / TOC / Outline helpers
RE_GUTTER_INT = re.compile(r"^\s*\d{1,3}\s*$")
RE_TOC_LINE   = re.compile(r"^\s*\d+(?:\.\d+)*\s+.+\s+\d{1,3}\s*$")
RE_TOC_TITLE  = re.compile(r"(목차|contents?)", re.I)
RE_OUTLINE    = re.compile(r"^\s*\d+(?:\.\d+){0,3}\.?(\))?\s*")          # 1. / 1.1. / 1.1.1)
RE_HEADING_HINT = re.compile(
    r"(개요|목적|정의|범위|FSE|MACE|서론|서문|적용|규정|요건|원칙|"
    r"scope|purpose|definition|overview)",
    re.I
)

# =========================
# Predicates
# =========================
def _is_heading(line: str) -> bool:
    s = (line or "").strip()
    return bool(RE_CHAPTER.match(s) or RE_SECTION.match(s))

def _is_article(line: str) -> bool:
    return bool(RE_ARTICLE.match((line or "").strip()))

def _is_clause(line: str) -> bool:
    s = (line or "").strip()
    return bool(RE_CLAUSE.match(s))

def _is_subitem(line: str) -> bool:
    s = (line or "").strip()
    return bool(RE_SUBITEM.match(s))

def _is_table_like(block_text: str) -> bool:
    if not block_text:
        return False
    lines = [l.strip() for l in block_text.splitlines() if l.strip()]
    if len(lines) <= 1:
        return False
    hits = sum(1 for l in lines if RE_TABLELIKE.search(l))
    return hits >= max(2, len(lines)//3)

# =========================
# Token helpers
# =========================
def _tail_by_tokens(enc: Callable[[str], List[int]], text: str, want: int) -> str:
    if want <= 0 or not text:
        return ""
    toks = enc(text) or []
    if len(toks) <= want:
        return text
    # approximate: take from end by lines
    lines = [l for l in text.splitlines() if l is not None]
    acc = []
    total = 0
    for l in reversed(lines):
        acc.append(l)
        total = len(enc("\n".join(reversed(acc)))) if acc else 0
        if total >= want:
            break
    return "\n".join(reversed(acc)).strip()

def _pack_tokens(
    enc: Callable[[str], List[int]],
    parts: List[Tuple[str, int]],
    target: int,
    overlap: int,
    min_tokens: int,
    section_title: str
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    parts: [(text, page_no)]
    토큰 길이 기반 패킹 + overlap. 너무 작으면 이전 청크와 병합.
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    cur: List[str] = []
    cur_pages: List[int] = []
    cur_tok = 0

    def flush(force: bool = False):
        nonlocal cur, cur_pages, cur_tok
        if not cur:
            return
        text = "\n".join(cur).strip()
        toks = enc(text) or []
        if not force and len(toks) < min_tokens and out:
            # merge into previous
            last_text, last_meta = out[-1]
            merged = (last_text + "\n" + text).strip()
            out[-1] = (merged, last_meta)
        else:
            meta = {
                "page": min(cur_pages) if cur_pages else 0,
                "pages": sorted(set(cur_pages)) if cur_pages else [],
                "section": section_title[:160],
                "bboxes": {},
            }
            out.append((text, meta))

        # overlap tail
        if overlap > 0:
            keep_text = _tail_by_tokens(enc, text, overlap)
            cur = [keep_text] if keep_text else []
            cur_pages = list(sorted(set(cur_pages))) if cur else []
            cur_tok = len(enc(keep_text) or []) if keep_text else 0
        else:
            cur, cur_pages, cur_tok = [], [], 0

    for t, pno in parts:
        toks = enc(t) or []
        if cur_tok + len(toks) <= max(target, min_tokens):
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

# =========================
# Gutter / TOC / Outline
# =========================
def _remove_left_gutter_numbers(lines: list[str]) -> list[str]:
    """
    페이지 내 '숫자만 라인' 비율이 높으면 거터로 간주하여 제거.
    약한 케이스는 번호를 다음 줄에 부착.
    """
    if not lines:
        return lines
    ints = sum(1 for l in lines if RE_GUTTER_INT.match(l))
    if ints >= max(3, len(lines)//5):  # 20%+
        return [l for l in lines if not RE_GUTTER_INT.match(l)]

    out: list[str] = []
    i = 0
    while i < len(lines):
        l = lines[i]
        if RE_GUTTER_INT.match(l) and (i + 1) < len(lines):
            nxt = lines[i + 1]
            if nxt.strip():
                out.append(f"{l.strip()} {nxt.strip()}")
                i += 2
                continue
        out.append(l)
        i += 1
    return out

def _is_toc_page(lines: list[str]) -> bool:
    if not lines:
        return False
    head = " ".join(lines[:5]).lower()
    if RE_TOC_TITLE.search(head):
        toc_like = sum(1 for l in lines if RE_TOC_LINE.match(l))
        return toc_like >= max(5, len(lines)//4)
    return False

def _is_outline_heading(line: str) -> bool:
    """
    소수점 아웃라인 번호(1., 1.1., 1.1.1)를
    '제목성 단서 + 길이'가 있을 때만 헤딩으로 인정.
    """
    s = line.strip()
    m = RE_OUTLINE.match(s)
    if not m:
        return False
    rest = RE_OUTLINE.sub("", s).strip()
    return bool(RE_HEADING_HINT.search(rest)) and (8 <= len(rest) <= 80)

# =========================
# Main chunker
# =========================
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
    법령/규정 전용 청커:
      - 헤딩(장/절/소수점 아웃라인) → 섹션 타이틀 업데이트
      - '제n조(…)'에서 신규 청크 시작
      - 항/목은 같은 조문에서 토큰 기반 패킹
      - 표(의심)는 통으로 묶어 별도 청크(meta.type='table')
      - 최소 토큰 미만은 앞 청크와 병합
    입력 형식: pages_std = [(page_no, text), ...]
    """
    layout_blocks = layout_blocks or {}
    chunks: List[Tuple[str, Dict[str, Any]]] = []

    cur_section = ""   # 장/절/헤딩
    cur_article = ""   # 제n조(…)
    buf: List[Tuple[str, int]] = []  # (line, page)

    def flush_article():
        nonlocal buf, cur_section, cur_article, chunks
        if not buf:
            return
        section_title = (cur_section + (" | " if cur_section and cur_article else "") + cur_article).strip()
        packed = _pack_tokens(enc, buf, target_tokens, overlap_tokens, min_chunk_tokens, section_title)
        chunks.extend(packed)
        buf = []

    for page_no, page_text in (pages_std or []):
        text = (page_text or "").strip()
        if not text:
            continue

        # Whole-page table? (heuristic)
        if _is_table_like(text):
            flush_article()
            section_title = (cur_section + (f" | {cur_article}" if cur_article else "")).strip() or "표"
            chunks.append((
                text,
                {"page": int(page_no), "pages": [int(page_no)], "section": section_title, "type": "table", "bboxes": {}}
            ))
            continue

        # Line-level processing
        lines = [l for l in text.splitlines() if l is not None]
        lines = [l.strip() for l in lines if l.strip()]

        # Gutter cleanup
        lines = _remove_left_gutter_numbers(lines)

        # TOC page?
        if _is_toc_page(lines):
            flush_article()
            toc_text = "\n".join(lines)
            chunks.append((
                toc_text,
                {"page": int(page_no), "pages": [int(page_no)], "section": "목차", "type": "toc", "bboxes": {}}
            ))
            continue

        for ln in lines:
            # Headings (chapter/section) or outline-style headings with title hints
            if _is_heading(ln) or _is_outline_heading(ln):
                flush_article()
                cur_section = ln.strip()
                continue

            # Article start
            if _is_article(ln):
                flush_article()
                cur_article = ln.strip()
                buf.append((ln.strip(), int(page_no)))
                continue

            # Default: accumulate under current article/body
            buf.append((ln.strip(), int(page_no)))

    # Tail flush
    flush_article()

    # Fallback: if nothing produced, pack big text
    if not chunks:
        big = []
        for p, t in (pages_std or []):
            if t and t.strip():
                big.append((t.strip(), int(p)))
        if big:
            chunks = _pack_tokens(enc, big, target_tokens, overlap_tokens, min_chunk_tokens, "")

    # Normalize meta (page/pages/section/bboxes)
    safe: List[Tuple[str, Dict[str, Any]]] = []
    for text, meta in chunks:
        meta = dict(meta or {})
        pg = 0
        if "pages" in meta and meta["pages"]:
            try:
                pg = int(meta["pages"][0])
            except Exception:
                pg = 0
        elif "page" in meta:
            try:
                pg = int(meta["page"])
            except Exception:
                pg = 0
        safe.append((
            text,
            {
                "page": pg,
                "pages": meta.get("pages") or ([pg] if pg else []),
                "section": (meta.get("section") or "")[:160],
                "bboxes": meta.get("bboxes") or {},
            }
        ))

    # De-dup consecutive identical chunks
    final: List[Tuple[str, Dict[str, Any]]] = []
    last: Optional[Tuple[str, Dict[str, Any]]] = None
    for it in safe:
        if it[0] and it != last:
            final.append(it)
            last = it
    return final
