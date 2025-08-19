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

def pack_by_tokens(
    paras: List[str],
    encode: Callable[[str], List[int]],
    target_tokens: int = 128,
    overlap_tokens: int = 32,
    overlap_paras: int = 1,   # 새로 추가: 텍스트도 겹치기
) -> List[str]:
    chunks: List[str] = []
    cur_paras: List[str] = []
    cur_ids: List[int] = []

    for p in paras:
        ids = encode(p)
        if not cur_paras:
            cur_paras, cur_ids = [p], ids
            continue

        if len(cur_ids) + len(ids) <= target_tokens:
            cur_paras.append(p)
            cur_ids += ids
        else:
            # 현재 청크 확정
            chunks.append("\n\n".join(cur_paras))

            # 텍스트도 겹치기: 마지막 overlap_paras 문단 carry
            if overlap_paras > 0:
                carry_paras = cur_paras[-overlap_paras:]
                # carry 토큰 재계산 (정확)
                carry_ids: List[int] = []
                for cp in carry_paras:
                    carry_ids += encode(cp)
            else:
                carry_paras, carry_ids = [], []

            # 새 청크 시작: carry + 현재 문단
            cur_paras = carry_paras + [p]
            cur_ids = carry_ids + ids

            # 혹시 carry+현재가 너무 크면, 현재만 넣는 안전장치
            if len(cur_ids) > target_tokens:
                chunks.append("\n\n".join(cur_paras[:-1]))  # carry만
                cur_paras = [p]
                cur_ids = ids

    if cur_paras:
        chunks.append("\n\n".join(cur_paras))
    return chunks


def smart_chunk_pages(pages: List[Tuple[int, str]], encode) -> List[Tuple[str, dict]]:
    # 임베딩 max 길이 감지
    try:
        from app.services.embedding_model import get_embedding_model
        max_len = getattr(get_embedding_model(), "max_seq_length", 128)
    except Exception:
        max_len = 128
    target = max(64, max_len - 16)     # 여유 16토큰
    overlap_tok = min(32, target // 4) # 겹침 25% 이내

    results = []
    for page_no, text in pages:
        paras = _split_to_paragraphs(text)
        chs = pack_by_tokens(
            paras, encode,
            target_tokens=target,
            overlap_tokens=overlap_tok,
            overlap_paras=1,          # 겹침 문단 1개 권장
        )
        for i, ch in enumerate(chs):
            section = ""
            for line in ch.splitlines():
                if HEADING_RE.match(line):
                    section = line.strip(); break
            results.append((ch, {"page": page_no, "section": section, "idx": i}))
    return results

def token_safe_chunks(text: str, target_tokens: int | None = None, overlap_tokens: int = 32) -> list[str]:
    """
    임베딩 모델의 max_seq_length에 맞춰 토큰 기준으로 안전하게 청킹.
    - 기본 target_tokens: (max_len - 16)로 자동 산정
    - overlap은 기본 32 토큰(필요시 조정)
    """
    # 지연 import로 순환참조 방지
    from app.services.embedding_model import get_embedding_model

    model = get_embedding_model()
    tok = getattr(model, "tokenizer", None)
    if tok is None or not hasattr(tok, "encode"):
        # 토크나이저가 없으면 기존 문자기반 청킹으로 폴백
        return chunk_text(text, max_length=1000)

    max_len = getattr(model, "max_seq_length", 128)
    if target_tokens is None:
        target_tokens = max(64, max_len - 16)

    # 문단/제목 경계 보존
    paras = _split_to_paragraphs(text)
    def _encode(s: str):
        return tok.encode(s, add_special_tokens=False)
    return pack_by_tokens(paras, _encode, target_tokens=target_tokens, overlap_tokens=min(overlap_tokens, target_tokens//4))
