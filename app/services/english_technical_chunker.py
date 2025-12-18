# app/services/english_technical_chunker.py
"""
English technical papers chunker (v2)
- 빠른 휴리스틱(정규식) 기반: 토크나이저 호출 없이 초고속
- 섹션 헤더(1., 1.1., I., A., APPENDIX …) 단위로 블록화
- 문단 복구(줄바꿈 래핑/하이픈 교정), 불릿/서브불릿은 부모 문단과 같은 청크로
- 페이지 경계는 무시하고 같은 섹션이면 자동 병합(跨-page continuity)
- target_tokens≈800, max≈1800 권장
- Milvus VARCHAR 8192 길이 제한 안전장치 추가
"""
from __future__ import annotations
import os
import re
from typing import List, Tuple, Dict, Optional, Callable

# -------------------- 헤더/불릿/각주 패턴 --------------------
EN_HEADER_RE = re.compile(
    r"""^(
        (?:\d+\.){1,4}\s+\S.+                  # 1. , 1.1. , 2.1.3. Title
      | [A-Z]\.\s+\S.+                         # A. Title
      | (?:Appendix|APPENDIX)\s+[A-Z0-9]+(?:\s*[:\-]\s*\S.+)?  # Appendix A: ...
      | (?:[1-9]\d*|I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+[A-Z][A-Z ,\-()’'\/:&]+$  # ALL CAPS header
    )$""", re.VERBOSE
)

EN_BULLET_RE = re.compile(
    r'^\s*(?:[\-\u2013\u2014\*•]|(?:\([a-zA-Zivx]+\)|\(\d+\)|\d+\)|\d+\.\d*\)))\s+'
)

FOOTRULE_RE = re.compile(r'^[ _]{5,}$')           # 하단 긴 밑줄
FOOTNOTE_LINE_RE = re.compile(r'^\s*\d+\s+.+')     # "1 some note..."
PAGENO_RE = re.compile(r'^\s*\d+\s*$')             # 페이지 번호 단독 라인

def _take_first_paragraph(text: str) -> tuple[str, str]:
    """text에서 첫 번째 문단만 분리해 (first, rest)로 반환."""
    m = re.search(r'\n\s*\n', text)
    if not m:
        return text.strip(), ""
    i = m.start()
    first = text[:i].strip()
    rest = text[m.end():].lstrip()
    return first, rest


# Milvus VARCHAR 길이 제한
MILVUS_VARCHAR_MAX = int(os.getenv("MILVUS_VARCHAR_MAX", "8192"))
# 바이트 기준 안전 제한 (환경변수로 조정 가능)
# 영어: 1글자 ≈ 1바이트, 한글: 1글자 ≈ 3바이트
# 5500자는 영어 문서 기준으로 충분히 안전한 크기
# 실제 임베딩 전 단계에서 바이트 기준으로 다시 한번 체크됨
SAFE_TEXT_LIMIT = int(os.getenv("RAG_SAFE_TEXT_LIMIT", "2500"))

def _split_by_limit(text: str, limit: int = SAFE_TEXT_LIMIT) -> list[str]:
    """
    텍스트가 limit을 초과하면 문장 단위로 안전하게 분할.
    - 문장 단위 분할 우선
    - 개별 문장도 limit 초과 시 강제 분할
    """
    if len(text) <= limit:
        return [text]
    
    parts, buf = [], ""
    # 문장 분리: 마침표/물음표/느낌표 뒤 공백 + 대문자/괄호
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z(])', text)
    
    for s in sents:
        s = s.strip()
        if not s:
            continue
        
        # 현재 버퍼에 추가 가능한지 체크
        if len(buf) + 1 + len(s) <= limit:
            buf = (buf + " " + s).strip()
        else:
            # 버퍼가 있으면 먼저 저장
            if buf:
                parts.append(buf)
            
            # 개별 문장도 limit 초과 시 강제 분할
            if len(s) > limit:
                for i in range(0, len(s), limit):
                    parts.append(s[i:i+limit])
                buf = ""
            else:
                buf = s
    
    if buf:
        parts.append(buf)
    
    return parts


# -------------------- 본 클래스 --------------------
class EnglishTechnicalChunker:
    def __init__(
        self,
        encoder_fn: Callable,
        target_tokens: int = 500,
        overlap_tokens: int = 0,
        cross_page_merge: bool = True,
    ):
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens  # v2에선 0 권장
        self.min_chunk_tokens = 100
        self.max_chunk_tokens = int(target_tokens * 2.25)  # ≈1125
        self.cross_page_merge = cross_page_merge

    # -------- 외부 진입점 --------
    def chunk_pages(
        self,
        pages_std: List[Tuple[int, str]],
        layout_blocks: Optional[Dict[int, List[Dict]]] = None,
    ) -> List[Tuple[str, Dict]]:
        if not pages_std:
            return []

        # 빠른 영어 비율 체크 (비영문이면 폴백을 위해 빈 리스트)
        sample = " ".join(text[:600] for _, text in pages_std[:3])
        eng = len(re.findall(r'[a-zA-Z]', sample))
        tot = max(1, len(sample.strip()))
        if eng / tot < 0.3:
            return []  # 영어 비율 30% 미만이면 포기

        all_chunks: List[Tuple[str, Dict]] = []

        for page_no, raw_text in pages_std:
            if not raw_text.strip():
                continue

            # 1) 줄바꿈 래핑 / 하이픈 교정
            clean = self._recover_line_wrapped_text(raw_text)

            # 2) 각주 제거
            clean = self._remove_footnotes(clean)

            # 3) 헤더 기반 블록화
            blocks = self._split_blocks_by_headers(clean)

            # 4) 각 블록을 문단/불릿 단위로 패킹
            for blk in blocks:
                paras = self._paragraphs_keep_bullets(blk)
                chs = self._pack_paragraphs(paras, page_no)
                all_chunks.extend(chs)

        # 5) 페이지 넘어서 같은 섹션 병합 (cross-page continuity)
        if self.cross_page_merge:
            all_chunks = self._merge_same_section_neighbors(all_chunks)

        # 6) 최종 정리 및 길이 제한 적용
        return self._finalize_chunks(all_chunks)

    # -------- 줄바꿈 래핑 복구 (line-wrapping) --------
    def _recover_line_wrapped_text(self, text: str) -> str:
        """
        OCR 등에서 한 문장이 여러 줄로 깨진 경우 복구:
          - 하이픈 제거 + 단어 결합
          - 문장 중간 줄바꿈 제거
        """
        # 하이픈 라인 결합: "exam- ple" → "example"
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

        # 문장 중간 줄바꿈 제거: 소문자로 끝나고 다음 줄이 소문자 시작 → 공백으로
        text = re.sub(r'([a-z,])\n\s+([a-z])', r'\1 \2', text)

        # 중복 공백 정리
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)
        return text

    # -------- 각주 제거 --------
    def _remove_footnotes(self, text: str) -> str:
        lines = text.split("\n")
        out = []
        in_footnote = False
        for ln in lines:
            if FOOTRULE_RE.match(ln.strip()):
                in_footnote = True
                continue
            if in_footnote:
                if FOOTNOTE_LINE_RE.match(ln.strip()):
                    continue
                if PAGENO_RE.match(ln.strip()):
                    continue
                # 일반 문장 재등장 → 각주 종료
                if ln.strip() and not ln.strip().isdigit():
                    in_footnote = False
            if not in_footnote:
                out.append(ln)
        return "\n".join(out)

    # -------- 헤더 기반 블록화 --------
    def _split_blocks_by_headers(self, text: str) -> List[str]:
        lines = [l for l in text.split("\n") if l.strip()]
        blocks, cur = [], []
        for ln in lines:
            if EN_HEADER_RE.match(ln.strip()):
                if cur:
                    blocks.append("\n".join(cur).strip())
                    cur = []
            cur.append(ln)
        if cur:
            blocks.append("\n".join(cur).strip())
        return blocks or [text]

    # -------- 문단/불릿 묶기 --------
    def _paragraphs_keep_bullets(self, block: str) -> List[str]:
        raw = [l.strip() for l in block.split("\n") if l.strip()]
        paras, buf = [], []

        def flush():
            nonlocal buf, paras
            if buf:
                # "\n\n" 마커는 실제 두 줄 공백으로 유지
                chunk = " ".join(buf).replace(" ¶¶ ", "\n\n").strip()
                if chunk:
                    paras.append(chunk)
                buf = []

        for ln in raw:
            if EN_BULLET_RE.match(ln):
                # 이전이 불릿이 아니면 문단 경계 마커 삽입하여 같은 청크 내 계층 유지
                if buf and not EN_BULLET_RE.match(buf[-1]):
                    buf.append(" ¶¶ ")  # 문단 경계 마커
                buf.append(ln)
                continue

            # 문장 경계 휴리스틱: 직전이 종결부호면 새 단락 가능
            if buf:
                prev = buf[-1]
                if re.search(r'[.!?"\'»\)\]}\->:]$', prev):
                    flush()
            buf.append(ln)
        flush()

        # 블록 첫 줄이 헤더면 다음 문단과 결합(짧은 헤더만)
        if paras:
            first_line = raw[0]
            if EN_HEADER_RE.match(first_line) and len(paras[0].split()) <= 12:
                if len(paras) >= 2:
                    paras[1] = first_line + "\n\n" + paras[1]
                    paras = paras[1:]
        return paras

    # -------- 토큰 예산 패킹 --------
    def _pack_paragraphs(self, paras: List[str], page_no: int) -> List[Tuple[str, Dict]]:
        out: List[Tuple[str, Dict]] = []
        cur_list: List[str] = []
        cur_tokens = 0

        for para in paras:
            t = self._estimate_tokens(para)

            # 아주 큰 문단은 문장 분할
            if t > self.max_chunk_tokens:
                if cur_list:
                    out.append(self._create_chunk("\n\n".join(cur_list), page_no))
                    cur_list, cur_tokens = [], 0
                out.extend(self._split_large_paragraph(para, page_no))
                continue

            # 불릿은 같은 청크로 최대한 붙이기
            is_bullet = EN_BULLET_RE.match(para) is not None

            if not cur_list:
                cur_list, cur_tokens = [para], t
                continue

            if cur_tokens + t <= self.target_tokens or is_bullet:
                cur_list.append(para)
                cur_tokens += t
            else:
                out.append(self._create_chunk("\n\n".join(cur_list), page_no))
                cur_list, cur_tokens = [para], t

        if cur_list:
            out.append(self._create_chunk("\n\n".join(cur_list), page_no))
        return out

    # -------- 페이지跨 섹션 연속 병합 (길이 체크 강화) --------
    def _merge_same_section_neighbors(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """
        같은 섹션이 페이지 경계를 넘어 연속되면 병합.
        
        **중요**: 병합 시 SAFE_TEXT_LIMIT(5500자)을 초과하지 않도록 체크.
        초과 시 _split_by_limit()로 안전하게 분할.
        """
        if not chunks:
            return chunks

        merged: List[Tuple[str, Dict]] = []

        def first_header(text: str) -> Optional[str]:
            head = text.split("\n", 1)[0].strip()
            return head if EN_HEADER_RE.match(head) else None

        def ends_mid_sentence(s: str) -> bool:
            s = s.rstrip()
            if not s:
                return False
            # 문장 종결 기호 체크
            return s[-1] not in {'.', '?', '!', ')', ']', '}', '-', ':', '"', "'"}

        def looks_like_continuation(s: str) -> bool:
            head = s.lstrip()[:20]
            return bool(re.match(r'^[a-z0-9(]', head)) or re.match(
                r'^\s*(?:and|or|but|however|moreover|furthermore|thus|therefore|additionally)\b',
                head, re.IGNORECASE
            )

        prev_text, prev_meta = chunks[0]
        prev_header = prev_meta.get("section") or first_header(prev_text)

        for i in range(1, len(chunks)):
            text, meta = chunks[i]
            cur_header = meta.get("section") or first_header(text)

            # 같은 섹션 & 연속 조건
            same_section = (prev_header and cur_header and prev_header == cur_header)
            is_continuation = ends_mid_sentence(prev_text) and looks_like_continuation(text)

            if same_section or is_continuation:
                # 병합 시도
                first, rest = _take_first_paragraph(text)
                candidate = (prev_text + "\n\n" + first).strip()

                # ===== 길이 체크 로직 =====
                if len(candidate) <= SAFE_TEXT_LIMIT:
                    # 안전하게 병합 가능
                    rep_meta = dict(prev_meta)
                    rep_meta["pages"] = sorted(
                        set(rep_meta.get("pages", [prev_meta.get("page", 0)])) |
                        {meta.get("page", 0)}
                    )
                    rep_meta["token_count"] = self._estimate_tokens(candidate)
                    prev_text, prev_meta = candidate, rep_meta

                    # 남은 텍스트(rest) 처리
                    if rest:
                        # rest도 길이 체크 후 추가
                        if len(rest) <= SAFE_TEXT_LIMIT:
                            rest_meta = dict(meta)
                            rest_meta["section"] = prev_header
                            rest_meta["token_count"] = self._estimate_tokens(rest)
                            # rest는 다음 반복에서 병합 시도하도록 prev로 설정
                            prev_text, prev_meta = rest, rest_meta
                        else:
                            # rest가 길이 초과 → 분할 후 추가
                            rest_parts = _split_by_limit(rest, SAFE_TEXT_LIMIT)
                            for j, part in enumerate(rest_parts):
                                part_meta = dict(meta)
                                part_meta["section"] = prev_header
                                part_meta["token_count"] = self._estimate_tokens(part)
                                if len(rest_parts) > 1:
                                    part_meta["split_index"] = j
                                merged.append((part, part_meta))
                            # 다음 반복을 위해 prev 초기화
                            if i + 1 < len(chunks):
                                prev_text, prev_meta = chunks[i + 1]
                                prev_header = prev_meta.get("section") or first_header(prev_text)
                            continue
                else:
                    # 병합 시 길이 초과 → 이전 청크 저장 후 분할 처리
                    # 1) 이전 청크 저장
                    if len(prev_text) <= SAFE_TEXT_LIMIT:
                        merged.append((prev_text, prev_meta))
                    else:
                        # 이전 청크도 길이 초과 → 분할
                        prev_parts = _split_by_limit(prev_text, SAFE_TEXT_LIMIT)
                        for j, part in enumerate(prev_parts):
                            part_meta = dict(prev_meta)
                            part_meta["token_count"] = self._estimate_tokens(part)
                            if len(prev_parts) > 1:
                                part_meta["split_index"] = j
                            merged.append((part, part_meta))

                    # 2) 현재 청크를 분할 (first + rest)
                    full_text = (first + "\n\n" + rest).strip() if rest else first
                    if len(full_text) <= SAFE_TEXT_LIMIT:
                        full_meta = dict(meta)
                        full_meta["section"] = prev_header or cur_header
                        full_meta["token_count"] = self._estimate_tokens(full_text)
                        prev_text, prev_meta = full_text, full_meta
                        prev_header = prev_header or cur_header
                    else:
                        # 분할 후 추가
                        parts = _split_by_limit(full_text, SAFE_TEXT_LIMIT)
                        for j, part in enumerate(parts):
                            part_meta = dict(meta)
                            part_meta["section"] = prev_header or cur_header
                            part_meta["token_count"] = self._estimate_tokens(part)
                            if len(parts) > 1:
                                part_meta["split_index"] = j
                            merged.append((part, part_meta))
                        # 다음 반복을 위해 prev 초기화
                        if i + 1 < len(chunks):
                            prev_text, prev_meta = chunks[i + 1]
                            prev_header = prev_meta.get("section") or first_header(prev_text)
                        continue
                continue

            # 연속 아님 → 이전 청크 저장
            if len(prev_text) <= SAFE_TEXT_LIMIT:
                merged.append((prev_text, prev_meta))
            else:
                # 길이 초과 시 분할
                parts = _split_by_limit(prev_text, SAFE_TEXT_LIMIT)
                for j, part in enumerate(parts):
                    part_meta = dict(prev_meta)
                    part_meta["token_count"] = self._estimate_tokens(part)
                    if len(parts) > 1:
                        part_meta["split_index"] = j
                    merged.append((part, part_meta))

            # 현재 청크를 다음 반복의 prev로 설정
            prev_text, prev_meta = text, meta
            prev_header = cur_header or prev_header

        # 마지막 청크 처리
        if len(prev_text) <= SAFE_TEXT_LIMIT:
            merged.append((prev_text, prev_meta))
        else:
            parts = _split_by_limit(prev_text, SAFE_TEXT_LIMIT)
            for j, part in enumerate(parts):
                part_meta = dict(prev_meta)
                part_meta["token_count"] = self._estimate_tokens(part)
                if len(parts) > 1:
                    part_meta["split_index"] = j
                merged.append((part, part_meta))

        return merged

    # -------- 유틸 --------
    @staticmethod
    def _estimate_tokens(s: str) -> int:
        # 영어 대략 1 token ≈ 0.75 words → 1.3 multiplier로 넉넉히 계산
        return int(len(s.split()) * 1.3)

    def _split_large_paragraph(self, paragraph: str, page_no: int) -> List[Tuple[str, Dict]]:
        """큰 문단을 문장 단위로 분할"""
        out: List[Tuple[str, Dict]] = []
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z(])', paragraph)
        cur, cur_t = "", 0
        for s in sents:
            s = s.strip()
            if not s:
                continue
            t = self._estimate_tokens(s)
            if cur_t + t <= self.target_tokens:
                cur = (cur + " " + s).strip()
                cur_t += t
            else:
                if cur:
                    out.append(self._create_chunk(cur, page_no))
                cur, cur_t = s, t
        if cur:
            out.append(self._create_chunk(cur, page_no))
        return out

    def _create_chunk(self, text: str, page_no: int) -> Tuple[str, Dict]:
        meta = {
            "type": "paragraph_group",
            "page": page_no,
            "pages": [page_no],
            "token_count": self._estimate_tokens(text),
        }
        # 섹션 헤더를 발견하면 메타에 기록(이후 병합 단계에서 활용)
        first = text.split("\n", 1)[0].strip()
        if EN_HEADER_RE.match(first):
            meta["section"] = first
        return (text.strip(), meta)

    def _finalize_chunks(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """
        최종 정리:
        - 중복 공백 제거
        - SAFE_TEXT_LIMIT(5500자) 길이 제한 적용 (안전장치)
        - chunk_index 부여
        """
        finalized = []
        idx = 0
        for _, (text, meta) in enumerate(chunks):
            if not text.strip():
                continue
            
            # 공백 정리
            clean = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            clean = re.sub(r'[ \t]+', ' ', clean)
            clean = re.sub(r' +$', '', clean, flags=re.MULTILINE).strip()
            
            # 길이 체크 및 분할 (최종 안전장치)
            pieces = _split_by_limit(clean, SAFE_TEXT_LIMIT)
            for j, piece in enumerate(pieces):
                m = dict(meta)
                m["chunk_index"] = idx
                idx += 1
                if len(pieces) > 1:
                    m["sub_index"] = j
                    m["orig_chunk_chars"] = len(clean)
                m["token_count"] = self._estimate_tokens(piece)
                finalized.append((piece, m))
        
        return finalized


# -------- 외부 함수 --------
def english_technical_chunk_pages(
    pages_std: List[Tuple[int, str]],
    encoder_fn: Callable,
    target_tokens: int = 800,
    overlap_tokens: int = 0,
    layout_blocks: Optional[Dict[int, List[Dict]]] = None,
) -> List[Tuple[str, Dict]]:
    """
    외부 진입점: 영어 기술 문서 청킹
    
    Args:
        pages_std: [(page_no, text), ...]
        encoder_fn: 토큰 인코더 함수
        target_tokens: 목표 토큰 수 (기본 800)
        overlap_tokens: 오버랩 토큰 수 (기본 0)
        layout_blocks: 레이아웃 정보 (현재 미사용)
    
    Returns:
        [(chunk_text, metadata_dict), ...]
    """
    chunker = EnglishTechnicalChunker(
        encoder_fn,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        cross_page_merge=True,
    )
    return chunker.chunk_pages(pages_std, layout_blocks)