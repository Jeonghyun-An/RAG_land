# app/services/chunker.py
import re
from typing import List, Tuple, Callable

def chunk_text(text: str, max_length: int = 500) -> list[str]:
    lines = text.split("\n")
    chunks = []
    chunk = ""
    for line in lines:
        if len(chunk) + len(line) < max_length:
            chunk += line + "\n"
        else:
            chunks.append(chunk.strip())
            chunk = line + "\n"
    if chunk:
        chunks.append(chunk.strip())
    return chunks



HEADING_RE = re.compile(r"^\s*(?:\d+(\.\d+)*|[IVXLC]+\.|[A-Z]\))\s+\S|^\s*#{1,6}\s+\S", re.M)
LIST_RE    = re.compile(r"^\s*[-*•·]\s+\S|^\s*\d+\.\s+\S", re.M)

def _split_to_paragraphs(text: str) -> List[str]:
    blocks = re.split(r"\n{2,}", text.strip())
    out = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        pieces = re.split(r"(?=\n?(?:%s|%s))" % (HEADING_RE.pattern, LIST_RE.pattern), b)
        out.extend([p.strip() for p in pieces if p.strip()])
    return out

def pack_by_tokens(paras: List[str], encode: Callable[[str], List[int]],
                   target_tokens=350, overlap_tokens=40) -> List[str]:
    chunks, cur, cur_ids = [], [], []
    for p in paras:
        ids = encode(p)
        if not cur:
            cur, cur_ids = [p], ids
            continue
        if len(cur_ids) + len(ids) <= target_tokens:
            cur.append(p); cur_ids += ids
        else:
            chunks.append("\n\n".join(cur))
            tail = cur_ids[-overlap_tokens:] if overlap_tokens>0 else []
            cur, cur_ids = [p], tail + ids
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

def smart_chunk_pages(pages: List[Tuple[int, str]], encode) -> List[Tuple[str, dict]]:
    results = []
    for page_no, text in pages:
        paras = _split_to_paragraphs(text)
        chs = pack_by_tokens(paras, encode, target_tokens=350, overlap_tokens=32)
        for i, ch in enumerate(chs):
            # 첫 제목 라인 추출(있으면)
            section = ""
            for line in ch.splitlines():
                if HEADING_RE.match(line):
                    section = line.strip(); break
            meta = {"page": page_no, "section": section, "idx": i}
            results.append((ch, meta))
    return results
