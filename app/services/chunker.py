# app/services/chunker.py
from __future__ import annotations
import re, hashlib
from typing import List, Tuple, Callable, Iterable, Any, Dict

# ===================== 유틸 =====================

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

def _toklen(enc: Callable[[str], List[Any]], s: str) -> int:
    return len(enc(s)) if s else 0

def _norm_text(s: str) -> str:
    """
    전역 중복 판정을 위한 강한 정규화:
    - 공백/개행 압축
    - 페이지바(… | 12) 제거
    - 연속 구두점/공백 정리
    """
    if not s:
        return ""
    t = s
    # 양쪽 공백 정리
    t = t.strip()
    # 페이지 바 패턴 제거: "… | 12" 또는 "… | 6" 등
    t = re.sub(r"\s*\|\s*\d+\s*$", "", t, flags=re.M)
    # 연속 공백 압축
    t = re.sub(r"[ \t]+", " ", t)
    # 줄단위 트림 후 빈줄 1개로
    lines = [ln.strip() for ln in t.splitlines()]
    lines = [ln for ln in lines if ln]
    t = "\n".join(lines)
    return t

def _hash_text(s: str) -> str:
    return hashlib.sha1(_norm_text(s).encode("utf-8", errors="ignore")).hexdigest()

# ===================== 패턴 =====================

HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+\.|[A-Z]\))\s+\S|^\s*#{1,6}\s+\S", re.M)
LIST_RE    = re.compile(r"^\s*[-*•·]\s+\S|^\s*\d+\.\s+\S", re.M)
SENT_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+|(?<=[。！？])|(?<=\n)")

# ===================== 문장/문단 분리 =====================

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

def _split_to_paragraphs(text: str) -> List[str]:
    """빈 줄로 크게 나누고, 블록 내부는 제목/리스트 시작줄 기준으로 쪼갬."""
    if not text:
        return []
    blocks = re.split(r"\n{2,}", text.strip())
    out: List[str] = []
    for b in blocks:
        b = (b or "").strip()
        if not b:
            continue
        lines = b.splitlines()
        piece: List[str] = []

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

# ===================== 과대 문단 세분화 =====================

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
                    # 단어 자체가 큰 경우 문자 단위로 강제 분절
                    if _toklen(encode, w) > target_tokens:
                        buf = ""
                        for ch in list(w):
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

# ===================== 페이지 헤더/푸터 제거 =====================

def _detect_repeating_lines(pages: List[Tuple[int, str]], head_n: int = 3, tail_n: int = 3, min_ratio: float = 0.2) -> Dict[str, int]:
    """
    여러 페이지에서 반복되는 상/하단 라인을 찾아낸다.
    - 각 페이지 앞 head_n줄 + 뒤 tail_n줄만 샘플링
    - 전체 페이지의 min_ratio 이상에서 등장하면 반복 라인으로 간주
    """
    freq: Dict[str, int] = {}
    total = max(1, len(pages))
    for _, txt in pages:
        lines = [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]
        heads = lines[:head_n]
        tails = lines[-tail_n:] if len(lines) >= tail_n else []
        for ln in heads + tails:
            # 페이지바 같은 것들 정리해서 안정화
            key = re.sub(r"\s*\|\s*\d+\s*$", "", ln)
            if len(key) <= 2:
                continue
            freq[key] = freq.get(key, 0) + 1
    # 임계치 이상만 남김
    cut = int(total * min_ratio)
    return {k: v for k, v in freq.items() if v >= cut and len(k) > 2}

def _strip_repeating_lines(text: str, repeating: Dict[str, int]) -> str:
    if not text or not repeating:
        return text or ""
    out: List[str] = []
    for ln in (text or "").splitlines():
        key = re.sub(r"\s*\|\s*\d+\s*$", "", ln.strip())
        if key in repeating:
            continue
        out.append(ln)
    return "\n".join(out)

# ===================== 토큰 패킹(오버랩/중복 보호) =====================

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
    """
    오버랩은 유지하되, '완전히 동일한 문단'이 두 번 들어가면 방지.
    """
    chunks: List[str] = []
    cur_paras: List[str] = []
    cur_ids: List[Any] = []

    def _emit():
        if not cur_paras:
            return
        piece = "\n\n".join(cur_paras).strip()
        if not piece:
            return
        # 바로 직전 청크와 완전 동일하면 스킵
        if chunks and _norm_text(chunks[-1]) == _norm_text(piece):
            return
        chunks.append(piece)

    for p in paras:
        if not p or not p.strip():
            continue
        ids = encode(p)
        # 문단이 target보다 크면 단독 청크로(단, 직전과 동일시 스킵)
        if len(ids) > target_tokens:
            _emit()
            candidate = p.strip()
            if not (chunks and _norm_text(chunks[-1]) == _norm_text(candidate)):
                chunks.append(candidate)
            cur_paras, cur_ids = [], []
            continue

        if not cur_paras:
            cur_paras, cur_ids = [p], ids
            continue

        if len(cur_ids) + len(ids) <= target_tokens:
            cur_paras.append(p)
            cur_ids += ids
        else:
            _emit()
            tail_paras, tail_ids = _tail_paras_by_tokens(cur_paras, encode, max(0, overlap_tokens))
            # tail + p 가 target 초과하면 p를 단독으로
            if len(tail_ids) + len(ids) > target_tokens:
                candidate = p.strip()
                if not (chunks and _norm_text(chunks[-1]) == _norm_text(candidate)):
                    chunks.append(candidate)
                cur_paras, cur_ids = [], []
            else:
                cur_paras, cur_ids = tail_paras + [p], tail_ids + ids
    _emit()

    # 인접 동일 제거(이미 했지만 2차 안전망)
    dedup: List[str] = []
    for c in chunks:
        if not dedup or _norm_text(dedup[-1]) != _norm_text(c):
            dedup.append(c)
    return dedup

# ===================== 퍼블릭 API =====================

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

def smart_chunk_pages(
    pages: List[Tuple[int, str]],
    encode,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> List[Tuple[str, dict]]:
    """
    입력: [(page_no, text)]
    출력: [(chunk_text, {"page":int, "section":str, "idx":int})]
    - 페이지 헤더/푸터 자동 제거
    - 전역 중복 제거(강한 정규화 해시)
    """
    enc = _ensure_encode(encode)

    # 기본 길이
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

    target_tokens = int(max(16, target_tokens))
    overlap_tokens = int(max(0, min(overlap_tokens, target_tokens // 2)))

    # 1) 반복 라인 감지
    repeating = _detect_repeating_lines(pages, head_n=3, tail_n=3, min_ratio=0.2)

    results: List[Tuple[str, dict]] = []
    seen_hashes: set[str] = set()

    for item in pages:
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

        # 2) 페이지 반복 상/하단 제거
        text = _strip_repeating_lines(text, repeating)

        # 3) 문단 → 과대 문단 세분화 → 토큰 패킹
        paras = _split_to_paragraphs(text)
        safe_paras: List[str] = []
        for p in paras:
            if _toklen(enc, p) <= target_tokens:
                if p.strip():
                    safe_paras.append(p.strip())
            else:
                safe_paras.extend(_split_oversize_to_tokens(p, enc, target_tokens))

        chs = pack_by_tokens(safe_paras, enc, target_tokens=target_tokens, overlap_tokens=overlap_tokens)

        # 4) 전역 중복 제거
        for i, ch in enumerate(chs):
            norm = _norm_text(ch)
            if not norm:
                continue
            h = _hash_text(norm)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # section(제목) 추출
            section = ""
            for line in ch.splitlines():
                if HEADING_RE.match(line):
                    section = line.strip()
                    break

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
    chs = pack_by_tokens(safe_paras, _enc, target_tokens=target_tokens, overlap_tokens=overlap_tokens)

    # 전역 중복 한 번 더 방지(함수 단위)
    dedup: List[str] = []
    seen: set[str] = set()
    for c in chs:
        h = _hash_text(c)
        if h in seen:
            continue
        seen.add(h)
        if not dedup or _norm_text(dedup[-1]) != _norm_text(c):
            dedup.append(c)
    return dedup
