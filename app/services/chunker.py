# app/services/chunker.py
from __future__ import annotations
import re, hashlib, os, itertools
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

# ---- bbox helper ----
def _clamp_bbox(bb: List[float]) -> List[float]:
    x0, y0, x1, y1 = map(float, bb[:4])
    x0 = max(0.0, x0); y0 = max(0.0, y0)
    x1 = max(x0, x1); y1 = max(y0, y1)
    return [x0, y0, x1, y1]
# ===================== 패턴 =====================

HEADING_RE = re.compile(
    r"""^
        \s*
        (?:                               # 대표 케이스들
            (?:제?\s*\d+(?:\.\d+)*\s*장?) |    # 제1장, 1.2, 1.2.3, 1 장
            (?:[IVXLC]+\.?) |                 # 로마숫자 I. II. 등
            (?:[■□◦\-*•·]\s*\d+) |            # 글머리표 + 숫자
            (?:\d+\s*[)\]]\s*) |               # 1)  1]
            (?:#{1,6}\s+)                      # markdown 헤딩
        )
        \s*\S+
    """,
    re.X | re.M
)
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
from collections import defaultdict
import re
from typing import List, Tuple, Set, Dict, Iterable, Any

_WS_RE = re.compile(r"\s+")

def _normalize_line(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())

def _iter_nonempty_lines(text: str) -> List[str]:
    return [ln for ln in (text or "").splitlines() if ln.strip()]

def _extract_page_texts(pages: Iterable[Any]) -> List[str]:
    """pages가 str / (page_no, text) / {'page':..,'text':..} 등을 섞어도 텍스트만 뽑아냄."""
    out: List[str] = []
    for it in pages or []:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, (list, tuple)):
            # (page_no, text, ...) 형태 지원
            if len(it) >= 2:
                out.append(str(it[1] or ""))
            else:
                out.append(str(it[0] or ""))
        elif isinstance(it, dict):
            out.append(str(it.get("text", "") or ""))
        else:
            out.append(str(it or ""))
    return out

def _detect_repeating_lines(
    pages: List[Any],
    *,
    # 신 파라미터명
    k_head: int = 3,
    k_tail: int = 3,
    min_len: int = 6,
    max_len: int = 120,
    min_support: float = 0.8,
    # 구 파라미터명(호환용)
    head_n: int | None = None,
    tail_n: int | None = None,
    min_ratio: float | None = None,
) -> List[Tuple[str, int]]:
    """
    여러 페이지에 반복되는 헤더/푸터 라인을 감지.
    반환: [(line, page_count), ...]
    - pages: str 또는 (page_no, text) 또는 {'page':..,'text':..} 섞여 있어도 OK
    - head_n/tail_n/min_ratio(구명)도 받으며, 있으면 우선 적용
    """
    # 구 파라미터명 우선 적용
    if head_n is not None:
        k_head = int(head_n)
    if tail_n is not None:
        k_tail = int(tail_n)
    if min_ratio is not None:
        min_support = float(min_ratio)

    texts = _extract_page_texts(pages)
    if not texts:
        return []

    N = len(texts)
    threshold = max(2, int(N * min_support))

    page_appear: Dict[str, set[int]] = defaultdict(set)

    for pi, pg_text in enumerate(texts):
        lines = _iter_nonempty_lines(pg_text)
        if not lines:
            continue

        head = lines[:k_head] if k_head > 0 else []
        tail = lines[-k_tail:] if k_tail > 0 else []
        candidates = head + tail

        for raw in candidates:
            line = _normalize_line(raw)
            if not (min_len <= len(line) <= max_len):
                continue
            page_appear[line].add(pi)

    repeating = [(line, len(idx_set)) for line, idx_set in page_appear.items() if len(idx_set) >= threshold]
    repeating.sort(key=lambda x: (-x[1], x[0]))
    return repeating

def _repeating_to_set(repeating: Any) -> Set[str]:
    """
    repeating이 set[str] / dict[str,int] / list[tuple[str,int]] / list[str]
    어느 것이든 정규화된 set[str]로 변환
    """
    if not repeating:
        return set()
    if isinstance(repeating, set):
        return { _normalize_line(x) for x in repeating }
    if isinstance(repeating, dict):
        return { _normalize_line(k) for k in repeating.keys() }
    if isinstance(repeating, (list, tuple)):
        out: Set[str] = set()
        for it in repeating:
            if isinstance(it, (list, tuple)) and it:
                out.add(_normalize_line(str(it[0])))
            else:
                out.add(_normalize_line(str(it)))
        return out
    # 기타 타입
    return { _normalize_line(str(repeating)) }


def _strip_repeating_lines(text: str, repeating_any: Any) -> str:
    """
    text에서 반복 라인 제거. repeating_any는 set/dict/list[tuple]/list[str] 등 무엇이든 OK.
    """
    if not text:
        return ""
    repeating = _repeating_to_set(repeating_any)
    if not repeating:
        return text

    out: List[str] = []
    for ln in (text or "").splitlines():
        # 페이지바 " | 12" 같은 꼬리 제거 후 비교
        key = re.sub(r"\s*\|\s*\d+\s*$", "", ln.strip())
        if _normalize_line(key) in repeating:
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
    너무 작은 청크(하한 미만)는 다음/이전과 자동 병합
    """
    MIN_TOK = int(os.getenv("RAG_MIN_CHUNK_TOKENS", "0") or 0)
    chunks: List[str] = []
    cur_paras: List[str] = []
    cur_ids: List[Any] = []

    def _emit(final: bool=False):
        if not cur_paras:
            return
        piece = "\n\n".join(cur_paras).strip()
        if not piece:
            return
        # 하한 미만인데 최종 emit이 아니면 일단 보류(다음 문단과 합치기 대기)
        if MIN_TOK > 0 and _toklen(encode, piece) < MIN_TOK and not final:
            return

        # 최종 emit인데 하한 미만이면 직전 청크에 병합 시도
        if MIN_TOK > 0 and _toklen(encode, piece) < MIN_TOK and final and chunks:
            prev = chunks[-1]
            merged = (prev + "\n\n" + piece).strip()
            if _toklen(encode, merged) <= target_tokens and _norm_text(prev) != _norm_text(merged):
                chunks[-1] = merged
                cur_paras.clear(); cur_ids.clear()
                return
            # 병합 불가면 그냥 내보냄 (유실 방지)

        # 바로 직전과 완전 동일하면 스킵
        if chunks and _norm_text(chunks[-1]) == _norm_text(piece):
            cur_paras.clear(); cur_ids.clear()
            return
        chunks.append(piece)
        cur_paras.clear(); cur_ids.clear()

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
            # ---- 하한 보호: 끊기 직전에 새 문단을 더 붙여도 target을 넘지 않으면 붙여서 자투리 방지 ----
            if MIN_TOK > 0 and cur_paras:
                tmp = "\n\n".join(cur_paras + [p]).strip()
                if _toklen(encode, tmp) <= target_tokens:
                    cur_paras.append(p)
                    cur_ids += ids
                    continue
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
    _emit(final=True)

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
    repeating = _detect_repeating_lines(pages)

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


def _guess_section_for_paragraph(paragraph: str, last_section: str) -> str:
    # 문단 맨 앞 라인이 제목 패턴이면 해당 라인을 섹션으로, 아니면 직전 섹션 계승
    for ln in (paragraph or "").splitlines():
        if HEADING_RE.match(ln):
            return ln.strip()
        break
    return last_section

def _clean(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\s\p{P}]+", " ", s, flags=re.UNICODE)
    return s.strip()

def _token_set(s: str) -> set[str]:
    return set(_clean(s).split())

def _similar(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A or not B: 
        return 0.0
    inter = len(A & B)
    return inter / min(len(A), len(B))   # 부분일치에 관대

def _attach_bboxes_to_paragraph(para: str, page_blocks: list[dict]) -> list[list[float]]:
    if not para or not page_blocks:
        return []
    p = para.replace("\n", " ").strip()
    if not p:
        return []
    bboxes: list[list[float]] = []
    # 임계치: 0.5 정도면 보수적, 0.3이면 관대
    THRESH = 0.35
    for blk in page_blocks:
        t = (blk.get("text") or "").strip()
        if not t:
            continue
        if _similar(p[:200], t[:200]) >= THRESH or _similar(p, t) >= THRESH:
            bb = blk.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                bboxes.append(_clamp_bbox([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]))
    # 중복 제거
    seen, uniq = set(), []
    for bb in bboxes:
        key = tuple(round(v, 2) for v in bb)  # 좌표 반올림으로 근접 중복 제거
        if key not in seen:
            seen.add(key); uniq.append(bb)
    return uniq

def smart_chunk_pages_plus(
    pages: List[Tuple[int, str]],
    encode,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
    layout_blocks: dict[int, list[dict]] | None = None,
) -> List[Tuple[str, dict]]:
    """
    기존 smart_chunk_pages를 확장:
      - 섹션(제목) 계승
      - 페이지 경계 오버랩(옵션)
      - 각 청크에 포함된 모든 페이지 목록 + 페이지별 BBOX 리스트 포함
    반환: [(chunk_text, {"page":int, "section":str, "pages":[int], "bboxes":{page:[bbox...]}})]
    """
    # 0) 토크나이저/기본 길이
    enc = _ensure_encode(encode)
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
            overlap_tokens = min(96, target_tokens // 3)

    target_tokens = int(max(16, target_tokens))
    overlap_tokens = int(max(0, min(overlap_tokens, target_tokens // 2)))

    cross_page = os.getenv("RAG_CROSS_PAGE_CHUNK", "1") == "1"

    # 1) 반복 라인 보수 제거
    repeating = _detect_repeating_lines(pages, head_n=3, tail_n=3, min_ratio=0.2)

    # 2) 페이지→문단(para) 단위로 전개 + 섹션 계승 + para별 bbox 붙이기
    para_items: list[dict] = []
    last_section = ""
    for page_no, text in pages:
        txt = _strip_repeating_lines(text, repeating)
        paras = _split_to_paragraphs(txt)
        # 과대 문단 세분화
        safe_paras: List[str] = []
        for p in paras:
            if _toklen(enc, p) <= target_tokens:
                if p.strip():
                    safe_paras.append(p.strip())
            else:
                safe_paras.extend(_split_oversize_to_tokens(p, enc, target_tokens))
        # para each → attach section/bbox
        blocks = (layout_blocks or {}).get(int(page_no), [])
        for p in safe_paras:
            sec = _guess_section_for_paragraph(p, last_section)
            if sec != last_section and sec:
                last_section = sec
            # 새 페이지 첫 문단이 직전 섹션 제목과 동일한 헤딩이면, 헤딩 라인은 버리고 본문만 사용
            lines = p.splitlines()
            if lines and HEADING_RE.match(lines[0]) and lines[0].strip() == last_section:
                p = "\n".join(lines[1:]).strip()
                if not p:
                    continue
            bbs = _attach_bboxes_to_paragraph(p, blocks)
            para_items.append({
                "page": int(page_no),
                "section": last_section,
                "text": p,
                "bboxes": bbs,
            })

    # 3) 토큰 패킹(페이지 경계 오버랩 허용)
    chunks: List[Tuple[str, dict]] = []
    cur_texts: List[str] = []
    cur_ids: List[Any] = []
    cur_pages: List[int] = []
    cur_bboxes: dict[int, list[list[float]]] = {}
    cur_section: str = ""
    SECTION_CAP = int(os.getenv("RAG_SECTION_MAX", "160"))
    seen_text_hashes: set[str] = set()

    def _emit():
        nonlocal cur_texts, cur_ids, cur_pages, cur_bboxes, cur_section
        if not cur_texts:
            return
        piece = "\n\n".join(cur_texts).strip()
        if not piece:
            return
        # 텍스트만 기준으로 전역 dedup (META/페이지 리스트 차이 때문에 생기는 중복 방지)
        norm = _norm_text(piece)
        h = _hash_text(norm)
        if h in seen_text_hashes:
            cur_texts, cur_ids, cur_pages, cur_bboxes, cur_section = [], [], [], {}, ""
            return
        seen_text_hashes.add(h)
        # META 라인 (텍스트 최상단에 삽입)
        meta = {
            "type": "text",
            "section": (cur_section or "")[:SECTION_CAP],
            "pages": sorted(list(dict.fromkeys(cur_pages))),  # unique-preserving order
            # 페이지별 bbox 중복 제거
            "bboxes": {
                int(k): (lambda lst: (lambda seen=set(), u=[]: ([
                    (seen.add(tuple(round(v,3) for v in bb)) or u.append(bb))
                    for bb in lst if tuple(round(v,3) for v in bb) not in seen
                ], u)[1])())(v) for k, v in cur_bboxes.items() if v
            },
        }
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        chunk_text = meta_line + "\n" + piece

        # 첫 페이지(호환) 및 section 저장
        comp_meta = {
            "page": int(meta["pages"][0]) if meta["pages"] else (cur_pages[0] if cur_pages else 0),
            "section": meta["section"],
            "pages": meta["pages"],
            "bboxes": meta["bboxes"],
        }
        # 인접 완전중복 방지
        if not chunks or _norm_text(chunks[-1][0]) != _norm_text(chunk_text):
            chunks.append((chunk_text, comp_meta))

        cur_texts, cur_ids, cur_pages, cur_bboxes, cur_section = [], [], [], {}, ""

    import json
    for it in para_items:
        ids = enc(it["text"])
        # 현재 버퍼가 비어있다면 초기화
        if not cur_texts:
            cur_texts = [it["text"]]
            cur_ids = list(ids)
            cur_pages = [it["page"]]
            if it["bboxes"]:
                cur_bboxes[it["page"]] = list(it["bboxes"])
            cur_section = it["section"]
            continue

        # 같은 섹션이면 그대로 잇고, 섹션 바뀌면 지금까지 emit
        if it["section"] and it["section"] != cur_section:
            _emit()
            cur_texts = [it["text"]]
            cur_ids = list(ids)
            cur_pages = [it["page"]]
            cur_bboxes = {it["page"]: list(it["bboxes"])} if it["bboxes"] else {}
            cur_section = it["section"]
            continue

        # 토큰 용량 검사
        if len(cur_ids) + len(ids) <= target_tokens:
            # cross_page가 꺼져 있고 새 페이지라면 이전 청크를 마무리하고 끊는다.
            if (not cross_page) and (it["page"] != cur_pages[-1]):
                _emit()
                cur_texts = [it["text"]]
                cur_ids   = list(ids)
                cur_pages = [it["page"]]
                cur_bboxes = {it["page"]: list(it["bboxes"])} if it["bboxes"] else {}
                cur_section = it["section"]
                continue
            # 같은 페이지이거나 cross_page가 켜져 있을 때만 이어붙임
            cur_texts.append(it["text"])
            cur_ids += ids
            # 페이지/박스 갱신
            if (cross_page or it["page"] == cur_pages[-1]) and (it["page"] not in cur_pages):
                cur_pages.append(it["page"])
            if it["bboxes"]:
                cur_bboxes.setdefault(it["page"], []).extend(it["bboxes"])
        else:
            # 오버랩 적용
            _emit()
            # 새 버퍼 시작
            cur_texts = [it["text"]]
            cur_ids = list(ids)
            cur_pages = [it["page"]]
            cur_bboxes = {it["page"]: list(it["bboxes"])} if it["bboxes"] else {}
            cur_section = it["section"]

    _emit()
    return chunks
