# app/services/english_technical_chunker.py (초고속 버전 - 섹션 파싱 제거)
"""
영어 기술 문서 특화 청킹 - 초고속 버전
- 섹션 파싱 완전 제거 (병목 제거)
- 문단 단위로만 분할
- 5분 → 10초 이내
"""
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Optional, Callable

EN_HEADER_RE = re.compile(
    r"""^(
        (?:\d+\.){1,4}\s+\S+                       # 1. , 1.1. , 2.1.3. Title
      | [A-Z]\.\s+\S+                              # A. Title
      | (?:APPENDIX|Appendix)\s+[A-Z0-9]+(?:\s*[:\-]\s*\S+)?  # Appendix A: ...
      | (?:[1-9]\d*|I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+[A-Z][A-Z ,\-()’']+$  # ALL CAPS header
    )$""", re.VERBOSE
)

EN_BULLET_RE = re.compile(
    r'^\s*(?:[\-\u2013\u2014\*•]|(?:\(\d+|\([a-zA-Zivx]+\)|\d+\)|\d+\.\d*\)))\s+'
)

FOOTRULE_RE = re.compile(r'^[ _]{5,}$')  # 페이지 하단 긴 밑줄
FOOTNOTE_LINE_RE = re.compile(r'^\s*\d+\s+.+')  # "1 some note..."


class EnglishTechnicalChunker:
    """영어 기술 문서 전용 청킹 - 초고속"""
    
    def __init__(self, encoder_fn: Callable, target_tokens: int = 800, overlap_tokens: int = 0):
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = 0  # 교정교열용
        self.min_chunk_tokens = 100
        self.max_chunk_tokens = target_tokens * 3
        
    def chunk_pages(
        self, 
        pages_std: List[Tuple[int, str]], 
        layout_blocks: Optional[Dict[int, List[Dict]]] = None
    ) -> List[Tuple[str, Dict]]:
        """페이지별 영어 문서 청킹 - 초고속 문단 기반"""
        if not pages_std:
            return []
        
        # ⭐ 영어 문서 여부만 빠르게 체크 (섹션 파싱 안 함!)
        full_text_sample = " ".join(text[:500] for _, text in pages_std[:3])
        
        # 영어 비율 체크 (빠른 휴리스틱)
        english_chars = len(re.findall(r'[a-zA-Z]', full_text_sample))
        total_chars = len(full_text_sample.strip())
        
        if total_chars == 0 or english_chars / total_chars < 0.3:
            # 영어 비율 30% 미만이면 빈 결과 (다른 청커로 폴백)
            return []
        
        print(f"[EN-CHUNK] English document detected ({english_chars/total_chars*100:.0f}% English)")
        
        # ⭐ 간단한 문단 기반 청킹만 수행
        return self._fast_paragraph_chunking(pages_std)
    def _fast_paragraph_chunking(self, pages_std):
        out = []
        for page_no, raw in pages_std:
            if not raw or not raw.strip():
                continue
            text = self._normalize_page_text(raw)
            for block in self._split_blocks_by_headers(text):
                paras = self._paragraphs_keep_bullets(block)

                cur, cur_tokens = [], 0
                for para in paras:
                    t = int(len(para.split()) * 1.3)
                    if t > self.max_chunk_tokens:
                        # 너무 큰 문단은 문장 단위로 쪼개서 넣기
                        out.extend(self._split_large_paragraph(para, page_no))
                        continue

                    if not cur:
                        cur, cur_tokens = [para], t
                        continue

                    if cur_tokens + t <= self.target_tokens or EN_BULLET_RE.match(para):
                        # 불릿은 같은 청크에 붙인다
                        cur.append(para); cur_tokens += t
                    else:
                        out.append(self._create_chunk("\n\n".join(cur), page_no))
                        cur, cur_tokens = [para], t

                if cur:
                    out.append(self._create_chunk("\n\n".join(cur), page_no))

        return self._finalize_chunks(out)

    def _normalize_page_text(self, text: str) -> str:
        # 하단 각주 블록 제거 (밑줄 이후 짧은 각주들)
        lines = text.splitlines()
        cleaned = []
        footnote_mode = False
        for i, ln in enumerate(lines):
            if FOOTRULE_RE.match(ln.strip()):
                footnote_mode = True
                continue
            if footnote_mode:
                # 각주: 보통 숫자 + 짧은 문장, 페이지 끝까지 skip
                if FOOTNOTE_LINE_RE.match(ln) or not ln.strip():
                    continue
            cleaned.append(ln)
        text = "\n".join(cleaned).strip()

        # 단어 중간 하이픈 줄바꿈 수선 (Non-\nProliferation -> Non-Proliferation)
        text = re.sub(r'(\w)-\n(\w)', r'\1-\2', text)         # 진짜 하이픈 단어
        text = re.sub(r'(\w)\n(\w)', r'\1 \2', text)          # 단순 줄바꿈은 공백

        # 단락 내 불필요 공백 정리
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)
        return text
    
    def _split_blocks_by_headers(self, text: str) -> list[str]:
        """헤더를 만나면 새 블록 시작 (1., 1.1., ALL-CAPS 등)"""
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

    def _paragraphs_keep_bullets(self, block: str) -> list[str]:
        """빈 줄이 없어도 '문장 경계'와 불릿을 이용해 문단 복구"""
        raw_lines = [l.strip() for l in block.split("\n") if l.strip()]

        # 1) 불릿 덩어리 유지
        paras, buf = [], []
        def flush():
            nonlocal buf, paras
            if buf:
                paras.append(" ".join(buf).strip()); buf = []

        for ln in raw_lines:
            if EN_BULLET_RE.match(ln):
                # 앞 문단과 붙여 동일 청크로 유지 (헤더/문단 다음에 연속 불릿 가능)
                if buf and not EN_BULLET_RE.match(buf[-1]):
                    buf.append("\n\n")  # 문단 경계 표시 (나중 join 시 두 줄 공백으로 바뀜)
                buf.append(ln)
                continue
            # 문장 경계 휴리스틱: 끝이 .?! or .)” 등이고, 다음이 대문자로 시작하면 단락 후보
            if buf:
                prev = buf[-1]
                if re.search(r'[.!?]["\')]*$', prev):
                    # 새 문단 시작
                    flush()
            buf.append(ln)
        flush()

        # 2) 헤더 첫 줄은 다음 문단과 합쳐 섹션 단위 유지
        if paras:
            first_line = raw_lines[0]
            if EN_HEADER_RE.match(first_line):
                # 첫 문단이 헤더만 있으면 다음 것과 합쳐줌
                if len(paras) >= 2 and len(paras[0].split()) <= 12:
                    paras[1] = first_line + "\n\n" + paras[1]
                    paras = paras[1:]
        return [p.replace("\n\n", "\n\n") for p in paras]

    
    def _split_paragraphs(self, text: str) -> List[str]:
        """문단 분리 - 빈 줄 기준"""
        # 빈 줄(2개 이상 개행)로 분리
        paragraphs = re.split(r'\n\s*\n+', text)
        
        # 빈 문단 제거
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_large_paragraph(
        self,
        paragraph: str,
        page_no: int
    ) -> List[Tuple[str, Dict]]:
        """큰 문단을 문장으로 분할"""
        chunks = []
        
        # 문장 분리 (간단한 패턴)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
        
        current_chunk = ""
        current_tokens = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_tokens = len(sent.split()) * 1.3
            
            if current_tokens + sent_tokens <= self.target_tokens:
                if current_chunk:
                    current_chunk += " " + sent
                else:
                    current_chunk = sent
                current_tokens += sent_tokens
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, page_no))
                
                current_chunk = sent
                current_tokens = sent_tokens
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, page_no))
        
        return chunks
    
    def _create_chunk(self, text: str, page_no: int) -> Tuple[str, Dict]:
        """청크 생성"""
        # ⭐ 간단한 메타데이터만
        metadata = {
            "type": "paragraph_group",
            "page": page_no,
            "pages": [page_no],
            "token_count": int(len(text.split()) * 1.3),  # 빠른 추정
        }
        
        return (text, metadata)
    
    def _finalize_chunks(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """최종 청크 정리"""
        finalized = []
        
        for i, (text, meta) in enumerate(chunks):
            if not text.strip():
                continue
            
            # 텍스트 정리
            text = self._clean_text(text)
            
            meta["chunk_index"] = i
            meta.setdefault("type", "paragraph_group")
            meta.setdefault("page", 1)
            meta.setdefault("pages", [1])
            
            finalized.append((text, meta))
        
        return finalized
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 과도한 공백 제거
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)
        
        return text.strip()


# ========== 외부 인터페이스 ==========

def english_technical_chunk_pages(
    pages_std: List[Tuple[int, str]], 
    encoder_fn: Callable,
    target_tokens: int = 800,
    overlap_tokens: int = 0,
    layout_blocks: Optional[Dict[int, List[Dict]]] = None
) -> List[Tuple[str, Dict]]:
    """영어 기술 문서 전용 청킹 - 초고속"""
    if not pages_std:
        return []
    
    chunker = EnglishTechnicalChunker(encoder_fn, target_tokens, overlap_tokens)
    return chunker.chunk_pages(pages_std, layout_blocks)