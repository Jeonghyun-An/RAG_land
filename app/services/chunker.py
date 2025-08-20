import re
from typing import List, Tuple, Callable, Iterable, Any
from typing import Iterable, Any, List

def _ensure_encode(encode):
    def _enc(s: str) -> List[Any]:
        try:
            out = encode(s) or []
        except Exception:
            out = []
        if not isinstance(out, list):
            out = list(out)
        return out
    return _enc

def _toklen(enc, s: str) -> int:
    return len(enc(s)) if s else 0


# -------- 기본 작은 청크러 (fallback) --------
def chunk_text(text: str, max_length: int = 500) -> list[str]:
    lines = (text or "").split("\n")
    chunks, buf = [], ""
    for line in lines:
        if len(buf) + len(line) < max_length:
            buf += line + "\n"
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = line + "\n"
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

# -------- 패턴들 --------
HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+\.|[A-Z]\))\s+\S|^\s*#{1,6}\s+\S", re.M)
LIST_RE    = re.compile(r"^\s*[-*•·]\s+\S|^\s*\d+\.\s+\S", re.M)

# -------- 문단 분리 (제목/리스트 경계 고려) --------
def _split_to_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    # 1차: 빈 줄로 크게 나누기
    blocks = re.split(r"\n{2,}", text.strip())
    out: List[str] = []
    for b in blocks:
        b = (b or "").strip()
        if not b:
            continue
        # 제목/리스트 경계는 '줄 시작 패턴'을 스캔하며 조각내기
        lines = b.splitlines()
        piece = []
        def _emit():
            if piece:
                seg = "\n".join(piece).strip()
                if seg:
                    out.append(seg)
        for ln in lines:
            if HEADING_RE.match(ln) or LIST_RE.match(ln):
                _emit()
                piece = [ln]
            else:
                piece.append(ln)
        _emit()
    return out


# -------- 문장 분리 --------
SENT_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+|(?<=[。！？])|(?<=\n)")

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [s.strip() for s in SENT_SPLIT_RE.split(text) if s and s.strip()]
    fused: List[str] = []
    buf: List[str] = []
    for p in parts:
        buf.append(p)
        if p.endswith(("다.", "요.", ".", "!", "?", "…")):
            fused.append(" ".join(buf).strip()); buf = []
    if buf:
        fused.append(" ".join(buf).strip())
    return fused

# -------- 토크나이저 보정 --------
def _ensure_encode(encode: Callable[[str], Iterable[Any]]) -> Callable[[str], List[Any]]:
    def _enc(s: str) -> List[Any]:
        try:
            out = encode(s) or []
        except Exception:
            out = []
        if not isinstance(out, list):
            out = list(out)
        return out
    return _enc

def _toklen(encode: Callable[[str], List[Any]], s: str) -> int:
    return len(encode(s)) if s else 0

# -------- 과대 문단 세분화 --------
def _split_oversize_to_tokens(text: str, encode: Callable[[str], List[Any]], target_tokens: int) -> List[str]:
    out: List[str] = []
    sents = _split_sentences(text)
    cur = ""
    for s in sents:
        if _toklen(encode, s) > target_tokens:
            if cur:
                out.append(cur.strip()); cur = ""
            words = re.split(r"(\s+)", s)  # 공백 보존
            seg = ""
            for w in words:
                candidate = (seg + w) if seg else w
                if _toklen(encode, candidate) > target_tokens:
                    if seg.strip():
                        out.append(seg.strip())
                    # w 자체가 target을 초과하는 극단 케이스 -> 강제 쪼개기
                    if _toklen(encode, w) > target_tokens:
                        # 고정 길이 조각
                        chars = list(w)
                        buf = ""
                        for ch in chars:
                            cand = buf + ch
                            if _toklen(encode, cand) > target_tokens:
                                if buf.strip():
                                    out.append(buf.strip())
                                buf = ch
                            else:
                                buf = cand
                        if buf.strip():
                            out.append(buf.strip())
                        seg = ""
                    else:
                        seg = w
                else:
                    seg = candidate
            if seg.strip():
                out.append(seg.strip())
            continue
        # 문장 합치기
        joined = (cur + " " + s).strip() if cur else s
        if _toklen(encode, joined) <= target_tokens:
            cur = joined
        else:
            if cur.strip():
                out.append(cur.strip())
            cur = s
    if cur.strip():
        out.append(cur.strip())
    return out

def _normalize_paras(paras: List[str], encode: Callable[[str], List[Any]], target_tokens: int) -> List[str]:
    out: List[str] = []
    for p in paras:
        if _toklen(encode, p) <= target_tokens:
            if p.strip():
                out.append(p.strip())
        else:
            out.extend(_split_oversize_to_tokens(p, encode, target_tokens))
    return out

def _tail_paras_by_tokens(paras: List[str], encode: Callable[[str], List[Any]], max_tokens: int) -> tuple[List[str], List[Any]]:
    tail: List[str] = []
    tail_ids: List[Any] = []
    for p in reversed(paras):
        ids = encode(p)
        if len(ids) + len(tail_ids) > max_tokens:
            break
        tail.insert(0, p)
        tail_ids = ids + tail_ids
    return tail, tail_ids

def pack_by_tokens(
    paras: List[str],
    encode: Callable[[str], List[Any]],
    target_tokens: int = 128,
    overlap_tokens: int = 32,
) -> List[str]:
    chunks: List[str] = []
    cur_paras: List[str] = []
    cur_ids: List[Any] = []

    def _emit():
        if not cur_paras:
            return
        piece = "\n\n".join(cur_paras).strip()
        if piece and (not chunks or chunks[-1] != piece):
            chunks.append(piece)

    for p in paras:
        if not p or not p.strip():
            continue
        ids = encode(p)
        if not cur_paras:
            cur_paras, cur_ids = [p], ids
            continue
        if len(cur_ids) + len(ids) <= target_tokens:
            cur_paras.append(p)
            cur_ids += ids
        else:
            _emit()
            tail_paras, tail_ids = _tail_paras_by_tokens(cur_paras, encode, max(0, overlap_tokens))
            if len(tail_ids) + len(ids) > target_tokens:
                cur_paras, cur_ids = [p], ids
            else:
                cur_paras, cur_ids = tail_paras + [p], tail_ids + ids
    if cur_paras:
        _emit()
    # 완전히 동일한 인접 청크 제거(안전망)
    dedup: List[str] = []
    for c in chunks:
        if not dedup or dedup[-1] != c:
            dedup.append(c)
    return dedup

def smart_chunk_pages(
    pages: List[Tuple[int, str]],
    encode,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> List[Tuple[str, dict]]:
    enc = _ensure_encode(encode)  # ← 추가

    # target/overlap 계산은 기존 로직 유지
    if target_tokens is None or overlap_tokens is None:
        try:
            from app.services.embedding_model import get_embedding_model
            m = get_embedding_model()
            max_len = int(getattr(m, "max_seq_length", 128))
        except Exception:
            max_len = 128
        if target_tokens is None:
            target_tokens = max(64, max_len - 16)
        if overlap_tokens is None:
            overlap_tokens = min(32, target_tokens // 4)

    # 과도한 값 방지
    target_tokens = int(max(16, target_tokens))
    overlap_tokens = int(max(0, min(overlap_tokens, target_tokens // 2)))

    results: List[Tuple[str, dict]] = []
    for item in pages:
        # (page_no, text) 이외 포맷 방어
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            page_no, text = item[0], item[1]
        elif isinstance(item, dict):
            page_no, text = item.get("page", 1), item.get("text", "")
        else:
            continue

        try:
            page_no = int(page_no)
        except Exception:
            page_no = 1
        text = str(text or "").strip()
        if not text:
            continue

        paras = _split_to_paragraphs(text)

        # 길이 초과 문단을 미리 세분화
        safe_paras: List[str] = []
        for p in paras:
            if _toklen(enc, p) <= target_tokens:
                if p.strip():
                    safe_paras.append(p.strip())
            else:
                # 문장→단어→문자 단위로라도 잘라 target 이하로
                safe_paras.extend(_split_oversize_to_tokens(p, enc, target_tokens))

        # 토큰 패킹 (중복 방지)
        chs = pack_by_tokens(safe_paras, enc, target_tokens=target_tokens, overlap_tokens=overlap_tokens)

        for i, ch in enumerate(chs):
            section = ""
            for line in ch.splitlines():
                if HEADING_RE.match(line):
                    section = line.strip(); break
            results.append((ch, {"page": page_no, "section": section, "idx": i}))
    return results
def token_safe_chunks(text: str, target_tokens: int | None = None, overlap_tokens: int = 32) -> list[str]:
    from app.services.embedding_model import get_embedding_model
    try:
        model = get_embedding_model()
        tok = getattr(model, "tokenizer", None)
        max_len = int(getattr(model, "max_seq_length", 128))
    except Exception:
        tok, max_len = None, 128

    if tok is None or not hasattr(tok, "encode"):
        return chunk_text(text, max_length=1000)

    if target_tokens is None:
        target_tokens = max(64, max_len - 16)
    target_tokens = int(max(16, target_tokens))
    overlap_tokens = int(max(0, min(overlap_tokens, target_tokens // 4)))

    def _enc(s: str) -> List[int]:
        return tok.encode(s, add_special_tokens=False) or []

    paras = _split_to_paragraphs(text or "")
    safe_paras = _normalize_paras(paras, _enc, target_tokens)
    return pack_by_tokens(safe_paras, _enc, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
