# app/services/layout_chunker.py
from __future__ import annotations
import re, json, os
from typing import List, Tuple, Dict, Any, Callable, Optional

# 표는 “행=정확 데이터”라 원자화 + 인접행 보조청크가 필수.
# 조항/불릿은 문맥형이므로 제목 단위로 묶고 토큰 오버랩.
# 본문은 기존 token_safe_chunks로 안정 패킹.
# 각 청크 맨 앞에 META 1줄을 넣어 타입/섹션을 보존(정확성·추적성↑).

# --- 패턴들 ---
BULLET_LINE = re.compile(
    r"^\s*[-*•·]\s+\S"            # 기호 불릿
    r"|^\s*\d+\.\s+\S"            # 숫자. 항목
    r"|^\s*\([a-z]\)\s+\S"        # (a) (b) ...
    r"|^\s*\([ivx]+\)\s+\S",      # (i) (ii) ...
    re.IGNORECASE
)
BULLET_CONT = re.compile(r"^\s{2,}\S")  # 들여쓰기된 불릿 연속 줄(설명 줄)
GRID_HINT   = re.compile(r"[│┃┆┇┋┊\|]")  # 표의 세로선 문자 or 파이프
HEADING_RE  = re.compile(
    r'^\s*(?:'
    r'제\s*\d+\s*(?:조|장|절)[\.\)]?\s*$'
    r'|[0-9]+[.)]\s*[^\n]{0,80}\s*$'
    r'|[IVXLCM]+[.)]\s*[^\n]{0,80}\s*$'
    r'|[A-Z][A-Za-z0-9\s\-]{0,40}:?\s*$'
    r')',
    re.IGNORECASE
)
# 러닝헤더: "5.3 … | 168" 등 (가운뎃점, 점 모두 허용)
RUNHDR_RE = re.compile(
    r"^\s*(?:[A-Z][A-Za-z0-9\.\s]{0,40}\s+)?\d+(?:\.\d+)*\s+[^\n]{1,60}\s[.\·]\s*\d{1,4}\s*$"
)

def _is_heading(line: str) -> bool:
    return bool(HEADING_RE.match(line.strip()))

def _is_bullet(line: str) -> bool:
    return bool(BULLET_LINE.match(line))

def _is_tableish_block(lines: List[str]) -> bool:
    # 라인 수가 너무 적으면 표로 보지 않음
    if len(lines) < 4:
        return False
    # 세로선/파이프 또는 2칸 이상 공백이 '충분히' 반복
    marks = sum(1 for ln in lines if GRID_HINT.search(ln) or ("  " in ln))
    return marks >= max(4, len(lines) // 2)


# --- 조문 단위 패킹 옵션 ---
ARTICLE_PACK   = os.getenv("RAG_PACK_BY_ARTICLE", "0") == "1"
ARTICLE_TARGET = int(os.getenv("RAG_ARTICLE_TARGET_TOKENS", "800"))

# 조문/ARTICLE 제목(행 전체) 감지
_HEAD_RE = re.compile(
    r"(?mi)^\s*(?:제\s*\d+\s*조[\.\)]?\s*$|ARTICLE\s+(?:[IVXLC]+|\d+)[\.\)]?\s*$)"
)

def _pack_blocks_by_article(blocks: list[tuple[int, dict]], encode) -> list[tuple[int, dict]]:  # encode 미사용이지만 시그니처 보존
    """제목(제 n조/ARTICLE …)을 만나면 다음 제목 전까지를 하나의 블록으로 묶는다.
    pages를 누적해 메타에 보존한다. 기본은 비활성(ARTICLE_PACK=0)."""
    if not ARTICLE_PACK:
        return blocks
    out: list[tuple[int, dict]] = []
    buf: list[tuple[int, dict]] = []
    head: Optional[str] = None
    pages_acc: list[int] = []
    for page, b in blocks:
        first = (b.get("body", "").splitlines() or [""])[0].strip()
        if _HEAD_RE.match(first):
            if buf:
                body = "\n".join(x[1]["body"] for x in buf).strip()
                out.append((buf[0][0], {"type": "text", "title": head or "", "body": body, "pages": pages_acc[:]}))
                buf, head, pages_acc = [], None, []
            head = first
        buf.append((page, b))
        pages_acc.append(int(page))
    if buf:
        body = "\n".join(x[1]["body"] for x in buf).strip()
        out.append((buf[0][0], {"type": "text", "title": head or "", "body": body, "pages": pages_acc[:]}))
    return out

# --- 여러 페이지의 BBox를 모아서 붙이는 헬퍼 ---
def _collect_bboxes_for_pages(text: str, pages_list: List[int], layout_blocks: Optional[dict[int, list[dict]]]) -> dict[int, list[list[float]]]:
    bmap: dict[int, list[list[float]]] = {}
    for p in (pages_list or []):
        p_bbs = (layout_blocks or {}).get(int(p), [])
        bb = _attach_bboxes_simple(text, p_bbs)
        if bb:
            bmap[int(p)] = bb
    return bmap

# --- 섹션 타이틀 선택 ---
SECTION_CAP    = int(os.getenv("RAG_SECTION_MAX", "160"))
SECTION_SCAN_N = int(os.getenv("RAG_HEADING_SCAN_LINES", "8"))
IGNORE_RUNNING_HDR = os.getenv("RAG_IGNORE_RUNNING_HEADER", "1") == "1"

def _pick_section_title(title: str, body: str) -> str:
    t = (title or "").strip()
    if IGNORE_RUNNING_HDR and t and RUNHDR_RE.match(t):
        t = ""  # 러닝헤더는 섹션으로 쓰지 않음
    if t:
        return t[:SECTION_CAP]

    b = (body or "").strip()
    if not b:
        return ""

    # 본문 첫 N줄에서 "제 n조/장/절 ..." 같은 표제 찾기
    lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
    for ln in lines[:SECTION_SCAN_N]:
        if HEADING_RE.match(ln):
            return ln[:SECTION_CAP]

    # 폴백: 첫 줄 80자
    return lines[0][:80] if lines else ""

# --- 페이지 말단 제목을 다음 청크 머리로 이관 ---
_FIX_TAIL_HEADING = os.getenv("RAG_FIX_TAIL_HEADING", "1") == "1"

def _move_tail_heading_to_next(chunks: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """
    케이스: [ ... 본문 ... '제 7 조'] | ['1. 양 당사국은 ...'] 처럼
    '제 7 조'가 앞 청크의 꼬리에만 존재할 때 → 다음 청크의 머리로 이동.
    - 임베딩 포맷(META 라인 포함)은 그대로 유지
    - 앞 청크 본문이 비면 드롭(빈 청크 제거)
    """
    if not _FIX_TAIL_HEADING:
        return chunks

    out: list[tuple[str, dict]] = []
    i = 0
    while i < len(chunks):
        text, meta = chunks[i]
        body = text.split("\n", 1)[1] if "\n" in text else text  # META 라인 이후
        lines = [ln for ln in body.splitlines() if ln.strip()]
        if lines and _HEAD_RE.match(lines[-1]) and i + 1 < len(chunks):
            # 뒤 청크 준비
            ntext, nmeta = chunks[i + 1]
            nbody = ntext.split("\n", 1)[1] if "\n" in ntext else ntext
            nlines = [ln for ln in nbody.splitlines()]
            # 앞 청크에서 제목 제거
            moved = lines[-1].strip()
            new_body = "\n".join(lines[:-1]).rstrip()
            # 다음 청크에 제목 붙이기 (이미 같은 제목이면 중복 방지)
            if not (nlines and nlines[0].strip() == moved):
                nbody2 = f"{moved}\n{nbody}".strip()
                # 다음 청크의 section/title도 갱신
                nmeta2 = dict(nmeta)
                nmeta2["section"] = moved[:SECTION_CAP]
                # 텍스트(메타라인 유지) 재구성
                head  = text.split("\n", 1)[0] if text.startswith("META:") else ""
                nhead = ntext.split("\n", 1)[0] if ntext.startswith("META:") else ""
                text2  = f"{head}\n{new_body}".strip() if head else new_body
                ntext2 = f"{nhead}\n{nbody2}".strip() if nhead else nbody2
                if not new_body.strip():
                    # 앞 청크 본문이 비면 앞 청크 드롭
                    chunks[i + 1] = (ntext2, nmeta2)
                    i += 1
                    continue
                # 반영
                out.append((text2, meta))
                chunks[i + 1] = (ntext2, nmeta2)
                i += 1
                continue
        # 변경 없음
        out.append((text, meta))
        i += 1
    return out

# --- BBOX 안전화 ---
def _clamp_bbox(bb: list[float]) -> list[float]:
    x0, y0, x1, y1 = map(float, bb[:4])
    x0 = max(0.0, x0); y0 = max(0.0, y0)
    x1 = max(x0, x1); y1 = max(y0, y1)
    return [x0, y0, x1, y1]

# --- 페이지 → 블록 분해 ---
def _split_pages_to_blocks(pages: List[Tuple[int, str]]) -> List[Tuple[int, Dict[str, Any]]]:
    """페이지 텍스트를 (type,title,body,page) 블록으로 분해"""
    blocks: List[Tuple[int, Dict[str, Any]]] = []
    for page_no, text in pages:
        raw_lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
        # 러닝헤더/푸터 제거
        lines = [ln for ln in raw_lines if not RUNHDR_RE.match(ln)]
        cur: Dict[str, Any] = {"type": "text", "title": "", "lines": [], "page": page_no}

        def flush():
            nonlocal cur
            if cur["lines"]:
                body = "\n".join(cur["lines"]).strip()
                if body:
                    cur2 = dict(cur); cur2["body"] = body; del cur2["lines"]
                    blocks.append((page_no, cur2))
            cur = {"type": "text", "title": "", "lines": [], "page": page_no}

        i = 0
        while i < len(lines):
            ln = lines[i]
            # 새 구획의 제목
            if _is_heading(ln):
                flush()
                cur["title"] = ln.strip()
                # 다음 줄이 표의 시작처럼 보이면 table 모드로
                j = i + 1
                tail: List[str] = []
                while j < len(lines) and not _is_heading(lines[j]):
                    tail.append(lines[j]); j += 1
                if _is_tableish_block(tail):
                    blocks.append((page_no, {"type": "table", "title": cur["title"], "body": "\n".join(tail).strip(), "page": page_no}))
                    cur = {"type": "text", "title": "", "lines": [], "page": page_no}
                    i = j; continue
                # 그렇지 않으면 텍스트/불릿을 모음

            # 불릿 블록 (연속 줄 포함)
            if _is_bullet(ln):
                flush()
                j = i; tail: List[str] = []
                while j < len(lines) and (_is_bullet(lines[j]) or BULLET_CONT.match(lines[j])):
                    tail.append(lines[j]); j += 1
                blocks.append((page_no, {"type": "bullet", "title": cur["title"], "body": "\n".join(tail), "page": page_no}))
                i = j; continue

            # 일반 텍스트 누적
            cur["lines"].append(ln); i += 1

        flush()
    return blocks

# --- 간이 표 파서 ---
def _parse_table_rows(block_body: str) -> List[Dict[str, str]]:
    """| 또는 다중 공백으로 열 분리. 헤더가 있으면 키로 매핑. 실패 시 []"""
    rows = [re.split(r"\s*\|\s*|\s{2,}", ln.strip()) for ln in block_body.splitlines() if ln.strip()]
    if not rows or max(len(r) for r in rows) < 2:
        return []
    head = rows[0]
    # 헤더가 설명형이면 이후 행과 길이를 비교하여 결정
    if len(head) >= 2 and all(len(r) == len(head) for r in rows[1:3] if r):
        cols = head; data = rows[1:]
    else:
        # 헤더 불명확: 1행도 데이터 취급
        cols = [f"col{i+1}" for i in range(max(len(r) for r in rows))]
        data = rows
    out: List[Dict[str, str]] = []
    for r in data:
        cells = r + [""] * (len(cols) - len(r))
        out.append({k: v for k, v in zip(cols, cells)})
    return out

def _windowed(lst: List, k: int, step: int = 1):
    for i in range(0, max(0, len(lst) - k + 1), step):
        yield lst[i:i + k]

# --- 간단 BBox 부착(부분 문자열 매칭) ---
def _attach_bboxes_simple(text: str, page_blocks: list[dict]) -> list[list[float]]:
    if not text or not page_blocks:
        return []
    t = text.replace("\n", " ").strip()
    out: List[List[float]] = []
    for blk in page_blocks:
        bt = (blk.get("text") or "").strip()
        probe = bt if len(bt) <= 80 else bt[:80]  # 매칭 내구성 ↑
        if probe and probe in t:
            bb = blk.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                out.append(_clamp_bbox(list(bb)))

    # 중복 제거
    seen, uniq = set(), []
    for bb in out:
        key = tuple(round(v, 3) for v in bb)  # 너무 미세한 차이는 동일 처리
        if key not in seen:
            seen.add(key)
            uniq.append(bb)
    return uniq

# --- 메인 ---
def layout_aware_chunks(
    pages: List[Tuple[int, str]],
    encode: Callable[[str], list],
    target_tokens: int,
    overlap_tokens: int,
    slide_rows: int = 4,
    layout_blocks: Optional[dict[int, list[dict]]] = None
) -> List[Tuple[str, dict]]:

    blocks = _split_pages_to_blocks(pages)
    blocks = _pack_blocks_by_article(blocks, encode)  # 기본 비활성(ARTICLE_PACK=0)
    out: List[Tuple[str, dict]] = []

    def meta_line(d: Dict[str, Any], page: int, title: str) -> str:
        """pages는 d.get('pages') 우선, 단일 page 폴백"""
        section = _pick_section_title(title, d.get("body", ""))[:SECTION_CAP]
        pages_field = d.get("pages") or [page]
        meta = {
            "type": d["type"],
            "section": section,
            "pages": [int(p) for p in pages_field],
            "bboxes": {}  # 실제 박스는 호출부 meta에 채움
        }
        return "META: " + json.dumps(meta, ensure_ascii=False)

    for page, b in blocks:
        title = (b.get("title") or "").strip()
        section = _pick_section_title(title, b.get("body", ""))[:SECTION_CAP]
        pages_list = b.get("pages") or [page]

        if b["type"] == "table":
            rows = _parse_table_rows(b["body"])
            if not rows:
                payload = b["body"]
                bmap = _collect_bboxes_for_pages(payload, pages_list, layout_blocks)
                head = f"\n[{section}]\n" if section else "\n"
                text = f"{meta_line(b, page, title)}{head}{payload}"
                out.append((text, {"page": page, "section": section, "pages": pages_list, "bboxes": bmap}))
                continue

            # (1) 원자 행
            for idx, row in enumerate(rows):
                alias = ", ".join([v for v in row.values() if v])
                head = f"[{section}]\n" if section else ""
                cols = ", ".join(row.keys())
                payload = f"{head}행#{idx+1}\n열: {cols}\n값: {alias}"
                bmap = _collect_bboxes_for_pages(alias, pages_list, layout_blocks)
                text = f"{meta_line(b, page, title)}\n{payload}"
                out.append((text, {"page": page, "section": section, "pages": pages_list, "bboxes": bmap}))

            # (2) 인접 행 보조
            for win in _windowed(rows, slide_rows, step=1):
                alias = "\n".join([", ".join(r.values()) for r in win])
                head2 = f"[{section}] 인접행 묶음\n" if section else "인접행 묶음\n"
                payload2 = f"{head2}{alias}"
                bmap = _collect_bboxes_for_pages(alias, pages_list, layout_blocks)
                text = f"{meta_line(b, page, title)}\n{payload2}"
                out.append((text, {"page": page, "section": section, "pages": pages_list, "bboxes": bmap}))

        elif b["type"] == "bullet":
            payload = b["body"]
            bmap = _collect_bboxes_for_pages(payload, pages_list, layout_blocks)
            head = f"\n[{section}]\n" if section else "\n"
            text = f"{meta_line(b, page, title)}{head}{payload}"
            out.append((text, {"page": page, "section": section, "pages": pages_list, "bboxes": bmap}))

        else:
            from app.services.chunker import token_safe_chunks
            is_article = ARTICLE_PACK and _HEAD_RE.match((b.get("title") or "") + "\n" + b.get("body", ""))
            limit = ARTICLE_TARGET if is_article else target_tokens

            # ✅ encode 넘겨주기: token_safe_chunks가 encode 파라미터를 지원하지 않으면
            # 내부에서 try/except로 처리해서 기존 시그니처와도 호환되게 만든다.
            chunks_made = 0
            try:
                gen = token_safe_chunks(b["body"], limit, overlap_tokens, encode)  # 선호
            except TypeError:
                gen = token_safe_chunks(b["body"], limit, overlap_tokens)          # 구버전 호환

            for ch in gen:
                payload = ch
                bmap = _collect_bboxes_for_pages(payload, pages_list, layout_blocks)
                head = f"\n[{section}]\n" if section else "\n"
                text = f"{meta_line(b, page, title)}{head}{payload}"
                out.append((text, {"page": page, "section": section, "pages": pages_list, "bboxes": bmap}))
                chunks_made += 1

            # 비상 방출: 본문이 있는데도 아무 청크가 없으면 그대로 1개라도 만든다
            if chunks_made == 0 and (b.get("body") or "").strip():
                payload = (b.get("body") or "").strip()
                bmap = _collect_bboxes_for_pages(payload, pages_list, layout_blocks)
                head = f"\n[{section}]\n" if section else "\n"
                text = f"{meta_line(b, page, title)}{head}{payload}"
                out.append((text, {"page": page, "section": section, "pages": pages_list, "bboxes": bmap}))

    # 제목이 페이지 꼬리에만 있는 경우 다음 청크로 이관
    return _move_tail_heading_to_next(out)
