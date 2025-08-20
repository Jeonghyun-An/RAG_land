# app/services/layout_chunker.py
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any, Callable

#표는 “행=정확 데이터”라 원자화 + 인접행 보조청크가 필수.
# 조항/불릿은 문맥형이므로 제목 단위로 묶고 토큰 오버랩.
# 본문은 기존 token_safe_chunks로 안정 패킹.
# 각 청크 맨 앞에 META 1줄을 넣어 타입/섹션을 보존(정확성·추적성↑).
# 간단 패턴
HEADING_LINE = re.compile(r"^\s*(표\s*[A-Z]-?\d+\.?|[A-Za-z가-힣0-9].*?:\s*$)")
BULLET_LINE  = re.compile(r"^\s*[-*•·]\s+\S|^\s*\d+\.\s+\S")
GRID_HINT    = re.compile(r"[│┃┆┇┋┊┋|]")  # 표의 세로선 문자 or 파이프

def _is_heading(line: str) -> bool:
    # 예: "표 C-5. ...", "비즈니스 규칙: PIL", "데이터 필드: ..."
    return bool(HEADING_LINE.match(line.strip()))

def _is_bullet(line: str) -> bool:
    return bool(BULLET_LINE.match(line))

def _is_tableish_block(lines: List[str]) -> bool:
    # 라인에 세로선/구분기호가 반복되거나, 2칸 이상 공백 분할이 지속되면 표로 간주
    marks = sum(1 for ln in lines if GRID_HINT.search(ln) or ("  " in ln))
    return marks >= max(3, len(lines)//3)

def _split_pages_to_blocks(pages: List[Tuple[int,str]]) -> List[Tuple[int, Dict[str,Any]]]:
    """페이지 텍스트를 (타입,본문,제목) 블록으로 분해"""
    blocks = []
    for page_no, text in pages:
        lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
        cur: Dict[str,Any] = {"type":"text","title":"","lines":[],"page":page_no}
        def flush():
            nonlocal cur
            if cur["lines"]:
                body = "\n".join(cur["lines"]).strip()
                if body:
                    cur2 = dict(cur); cur2["body"]=body; del cur2["lines"]
                    blocks.append((page_no, cur2))
            cur = {"type":"text","title":"","lines":[],"page":page_no}

        i=0
        while i < len(lines):
            ln = lines[i]
            # 새 구획의 제목
            if _is_heading(ln):
                flush()
                cur["title"] = ln.strip()
                # 다음 줄이 표의 시작처럼 보이면 table 모드로
                j = i+1
                tail = []
                while j < len(lines) and not _is_heading(lines[j]):
                    tail.append(lines[j]); j += 1
                if _is_tableish_block(tail):
                    blocks.append((page_no, {"type":"table","title":cur["title"],"body":"\n".join(tail).strip(),"page":page_no}))
                    cur = {"type":"text","title":"","lines":[],"page":page_no}
                    i = j; continue
                # 그렇지 않으면 텍스트/불릿을 모음
            # 불릿 블록
            if _is_bullet(ln):
                flush()
                j=i; tail=[]
                while j < len(lines) and _is_bullet(lines[j]):
                    tail.append(lines[j]); j+=1
                blocks.append((page_no, {"type":"bullet","title":cur["title"],"body":"\n".join(tail),"page":page_no}))
                i=j; continue
            # 일반 텍스트 누적
            cur["lines"].append(ln); i+=1
        flush()
    return blocks

def _parse_table_rows(block_body: str) -> List[Dict[str,str]]:
    """
    간이 표 파서: | 또는 다중 공백으로 열 분리. 헤더가 있으면 키로 매핑.
    실패시 전체 본문 하나로 반환.
    """
    rows = [re.split(r"\s*\|\s*|\s{2,}", ln.strip()) for ln in block_body.splitlines() if ln.strip()]
    if not rows or max(len(r) for r in rows) < 2:
        return []
    head = rows[0]
    # 헤더가 설명형이면 이후 행과 길이를 비교하여 결정
    if len(head) >= 2 and all(len(r)==len(head) for r in rows[1:3] if r):
        cols = head
        data = rows[1:]
    else:
        # 헤더 불명확: 1행도 데이터 취급
        cols = [f"col{i+1}" for i in range(max(len(r) for r in rows))]
        data = rows
    out=[]
    for r in data:
        cells = r + [""]*(len(cols)-len(r))
        out.append({k:v for k,v in zip(cols, cells)})
    return out

def _windowed(lst: List, k: int, step: int=1):
    for i in range(0, max(0, len(lst)-k+1), step):
        yield lst[i:i+k]

def layout_aware_chunks(
    pages: List[Tuple[int,str]],
    encode: Callable[[str], list],
    target_tokens: int,
    overlap_tokens: int,
    slide_rows: int = 4
) -> List[Tuple[str, dict]]:
    """
    표/불릿/텍스트를 타입별로 나눠 원자청크 생성 + 표는 보조(슬라이딩) 청크.
    반환: List[(text, {"page","section"})]
    """
    blocks = _split_pages_to_blocks(pages)
    out: List[Tuple[str,dict]] = []

    def meta_line(d: Dict[str,Any]) -> str:
        m = {"type": d["type"], "section": (d.get("title") or "").strip()}
        import json
        return "META: " + json.dumps(m, ensure_ascii=False)

    for page, b in blocks:
        title = (b.get("title") or "").strip()
        if b["type"] == "table":
            rows = _parse_table_rows(b["body"])
            if not rows:
                # 파싱 실패 → 본문 통짜로
                text = f"{meta_line(b)}\n[{title}]\n{b['body']}"
                out.append((text, {"page":page, "section":title})); continue
            # (1) 행 원자청크
            for idx, row in enumerate(rows):
                alias = ", ".join([v for v in row.values() if v])
                text = f"""{meta_line(b)}
[{title}] 행#{idx+1}
열: {", ".join(row.keys())}
값: {alias}"""
                out.append((text, {"page":page, "section":title}))
            # (2) 보조: 인접 행 묶음(오버랩)
            for win in _windowed(rows, slide_rows, step=1):
                alias = "\n".join([", ".join(r.values()) for r in win])
                text = f"""{meta_line(b)}
[{title}] 인접행 묶음
{alias}"""
                out.append((text, {"page":page, "section":title}))
        elif b["type"] == "bullet":
            text = f"{meta_line(b)}\n[{title}]\n{b['body']}"
            out.append((text, {"page":page, "section":title}))
        else:
            # 일반 텍스트는 기존 청킹으로 토큰 기준 패킹
            from app.services.chunker import token_safe_chunks
            for ch in token_safe_chunks(b["body"], target_tokens, overlap_tokens):
                text = f"{meta_line(b)}\n[{title}]\n{ch}"
                out.append((text, {"page":page, "section":title}))
    return out
