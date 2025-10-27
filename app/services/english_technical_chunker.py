# app/services/english_technical_chunker.py
"""
English technical papers chunker (v2)
- 빠른 휴리스틱(정규식) 기반: 토크나이저 호출 없이 초고속
- 섹션 헤더(1., 1.1., I., A., APPENDIX …) 단위로 블록화
- 문단 복구(줄바꿈 래핑/하이픈 교정), 불릿/서브불릿은 부모 문단과 같은 청크로
- 페이지 경계는 무시하고 같은 섹션이면 자동 병합(跨-page continuity)
- target_tokens≈800, max≈1800 권장
"""
from __future__ import annotations
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

# -------------------- 본 클래스 --------------------
class EnglishTechnicalChunker:
    def __init__(
        self,
        encoder_fn: Callable,
        target_tokens: int = 800,
        overlap_tokens: int = 0,
        cross_page_merge: bool = True,
    ):
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens  # v2에선 0 권장
        self.min_chunk_tokens = 100
        self.max_chunk_tokens = int(target_tokens * 2.25)  # ≈1800
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
        if eng / tot < 0.30:
            return []

        # 1) 페이지별 정리 → 2) 헤더로 블록화 → 3) 문단/불릿 묶음 → 4) 토큰 예산 패킹
        prelim_chunks: List[Tuple[str, Dict]] = []
        for page_no, raw in pages_std:
            if not raw or not raw.strip():
                continue
            text = self._normalize_page_text(raw)
            for block in self._split_blocks_by_headers(text):
                paras = self._paragraphs_keep_bullets(block)
                prelim_chunks.extend(
                    self._pack_paragraphs(paras, page_no)
                )

        # 5) 페이지跨 섹션 연속 병합
        if self.cross_page_merge:
            prelim_chunks = self._merge_same_section_neighbors(prelim_chunks)

        # 6) 마무리 정리
        return self._finalize_chunks(prelim_chunks)

    # -------- 페이지 정리 --------
    def _normalize_page_text(self, text: str) -> str:
        # 하단 각주/밑줄/페이지번호 제거
        lines = text.splitlines()
        cleaned, footnote_mode = [], False
        for i, ln in enumerate(lines):
            s = ln.strip()
            if FOOTRULE_RE.match(s):
                footnote_mode = True
                continue
            if footnote_mode:
                if not s or FOOTNOTE_LINE_RE.match(ln):
                    continue
            if PAGENO_RE.match(s):
                # 고립된 페이지 번호 라인 제거
                continue
            cleaned.append(ln)
        text = "\n".join(cleaned).strip()

        # 하이픈 줄바꿈 교정, 일반 줄바꿈 래핑 복구
        text = re.sub(r'(\w)-\n(\w)', r'\1-\2', text)   # 진짜 하이픈 단어
        text = re.sub(r'(\w)\n(\w)', r'\1 \2', text)    # 나머지는 공백 결합

        # 중복 공백 정리
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)
        return text

    # -------- 헤더 기반 블록화 --------
    def _split_blocks_by_headers(self, text: str) -> List[str]:
        lines = [l for l in text.split("\n") if l.strip()]
        blocks, cur = [], []
        for ln in lines:
            if EN_HEADER_RE.match(ln.strip()):
                if cur:
                    blocks.append("\n".join(cur).strip()); cur = []
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
                if re.search(r'[.!?]["\')]*$', prev):
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
                cur_list.append(para); cur_tokens += t
            else:
                out.append(self._create_chunk("\n\n".join(cur_list), page_no))
                cur_list, cur_tokens = [para], t

        if cur_list:
            out.append(self._create_chunk("\n\n".join(cur_list), page_no))
        return out

    # -------- 페이지跨 섹션 연속 병합 --------
    def _merge_same_section_neighbors(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        if not chunks:
            return chunks
    
        merged: List[Tuple[str, Dict]] = []
        current_section: Optional[str] = None
    
        def first_header(text: str) -> Optional[str]:
            first = text.split("\n", 1)[0].strip()
            return first if EN_HEADER_RE.match(first) else None
    
        def ends_mid_sentence(s: str) -> bool:
            s = s.rstrip()
            if not s:
                return False
            # 문장 종결부호/따옴표/괄호로 끝나지 않으면 미완성 가능성↑
            return s[-1] not in {".", "?", "!", "”", "’", '"', "'", ")", "]", "}"}
    
        def looks_like_continuation(s: str) -> bool:
            # 소문자/숫자/괄호로 시작하거나, 접속사·부사로 시작하면 이어쓰기 가능성↑
            head = s.lstrip()[:12]
            return bool(re.match(r'^[a-z0-9(]', head)) or re.match(
                r'^\s*(?:and|or|but|so|also|as|well|thus|therefore|hence|however|while|whereas|in|under|with|for)\b',
                s, flags=re.I
            )
    
        for text, meta in chunks:
            header = first_header(text)
    
            if header:
                # 새 섹션 시작
                current_section = header
                # 메타에 섹션 저장
                if meta.get("section") != header:
                    meta = dict(meta, section=header)
    
                # 단순 push 또는 직전과 같은 섹션이면 이어붙일지 검사
                if not merged:
                    merged.append((text, meta))
                    continue
                
                prev_text, prev_meta = merged[-1]
                prev_header = prev_meta.get("section") or first_header(prev_text)
                if prev_header and prev_header == header:
                    # 같은 헤더가 연속으로 등장하면 이어붙임
                    new_text = prev_text + "\n\n" + text
                    new_meta = dict(prev_meta)
                    new_meta["pages"] = sorted(set(prev_meta.get("pages", []) + meta.get("pages", [])))
                    new_meta["token_count"] = self._estimate_tokens(new_text)
                    new_meta["section"] = header
                    merged[-1] = (new_text, new_meta)
                else:
                    merged.append((text, meta))
                continue
            
            # header 없는 청크(= 섹션 본문 지속)
            if not merged:
                merged.append((text, meta))
                continue
            
            prev_text, prev_meta = merged[-1]
            prev_section = prev_meta.get("section") or first_header(prev_text)
    
            if prev_section:
                # ① 이전이 문장 중간에서 끝났거나 ② 현재가 이어지는 형태면 병합
                if ends_mid_sentence(prev_text) or looks_like_continuation(text):
                    new_text = prev_text + "\n\n" + text
                    new_meta = dict(prev_meta)
                    new_meta["pages"] = sorted(set(prev_meta.get("pages", []) + meta.get("pages", [])))
                    new_meta["token_count"] = self._estimate_tokens(new_text)
                    new_meta["section"] = prev_section
                    merged[-1] = (new_text, new_meta)
                    continue
                else:
                    # 섹션은 유지하되 새 청크로 둠
                    meta = dict(meta, section=prev_section)
    
            merged.append((text, meta))
    
        return merged


    # -------- 유틸 --------
    @staticmethod
    def _estimate_tokens(s: str) -> int:
        # 영어 대략 1 token ≈ 0.75 words → 1.3 multiplier로 넉넉히 계산
        return int(len(s.split()) * 1.3)

    def _split_large_paragraph(self, paragraph: str, page_no: int) -> List[Tuple[str, Dict]]:
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
        finalized = []
        for i, (text, meta) in enumerate(chunks):
            if not text.strip():
                continue
            clean = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            clean = re.sub(r'[ \t]+', ' ', clean)
            clean = re.sub(r' +$', '', clean, flags=re.MULTILINE).strip()
            m = dict(meta)
            m["chunk_index"] = i
            m.setdefault("type", "paragraph_group")
            m.setdefault("page", meta.get("page", 1))
            m.setdefault("pages", meta.get("pages", [m["page"]]))
            m["token_count"] = self._estimate_tokens(clean)
            finalized.append((clean, m))
        return finalized

# -------- 외부 함수 --------
def english_technical_chunk_pages(
    pages_std: List[Tuple[int, str]],
    encoder_fn: Callable,
    target_tokens: int = 800,
    overlap_tokens: int = 0,
    layout_blocks: Optional[Dict[int, List[Dict]]] = None,
) -> List[Tuple[str, Dict]]:
    chunker = EnglishTechnicalChunker(
        encoder_fn,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        cross_page_merge=True,
    )
    return chunker.chunk_pages(pages_std, layout_blocks)
