# app/services/english_technical_chunker.py (교정교열용 - 깔끔한 문단 분할)
"""
영어 기술 문서 특화 청킹 모듈 - 교정교열용
- 오버랩 없음 (중복 제거)
- 문단 단위로 깔끔하게 분할
- 섹션 구조 보존
- 성능 최적화
"""
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class SectionInfo:
    """섹션 정보"""
    number: str
    title: str
    level: int
    start_line: int
    content: str = ""


class EnglishTechnicalChunker:
    """영어 기술 문서 전용 청킹 - 교정교열 최적화"""
    
    def __init__(self, encoder_fn: Callable, target_tokens: int = 800, overlap_tokens: int = 0):
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = 0  # ⭐ 교정교열용이므로 오버랩 없음
        self.min_chunk_tokens = 100
        self.max_chunk_tokens = target_tokens * 3
        
        # 토큰 카운트 캐시
        self._token_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 섹션 번호 패턴
        self.section_pattern = re.compile(
            r'^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s,\-\']+?)(?:\s*\n|$)',
            re.MULTILINE
        )
        
        # 대문자 제목 패턴
        self.title_pattern = re.compile(
            r'^([A-Z][A-Z\s]{3,})(?:\s*\n|$)',
            re.MULTILINE
        )
        
        # 불릿 포인트 패턴
        self.bullet_patterns = [
            re.compile(r'^[\s]*[\-•●○▪▫]\s+', re.MULTILINE),
            re.compile(r'^[\s]*\([a-z]\)\s+', re.MULTILINE),
            re.compile(r'^[\s]*\([ivxIVX]+\)\s+', re.MULTILINE),
            re.compile(r'^[\s]*[a-z]\)\s+', re.MULTILINE),
        ]
        
        self.sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
    def chunk_pages(
        self, 
        pages_std: List[Tuple[int, str]], 
        layout_blocks: Optional[Dict[int, List[Dict]]] = None
    ) -> List[Tuple[str, Dict]]:
        """페이지별 영어 기술 문서 청킹 - 교정교열용"""
        if not pages_std:
            return []
        
        all_chunks = []
        
        # 전체 문서를 하나로 결합
        full_text = ""
        page_boundaries = []
        
        for page_no, text in pages_std:
            start_pos = len(full_text)
            full_text += text + "\n\n"
            end_pos = len(full_text)
            page_boundaries.append((start_pos, end_pos, page_no))
        
        # 섹션 파싱
        sections = self._parse_sections(full_text)
        
        if not sections:
            # 섹션 없으면 문단 기반
            return self._paragraph_based_chunking(pages_std, layout_blocks)
        
        # 각 섹션 청킹 (오버랩 없이)
        for section in sections:
            section_chunks = self._chunk_section(
                section, 
                page_boundaries,
                layout_blocks
            )
            all_chunks.extend(section_chunks)
        
        # 디버그 정보
        if self._cache_hits + self._cache_misses > 0:
            hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) * 100
            print(f"[CHUNKER] Token cache hit rate: {hit_rate:.1f}%")
        print(f"[CHUNKER] Total chunks: {len(all_chunks)} (no overlap)")
        
        return self._finalize_chunks(all_chunks)
    
    def _parse_sections(self, text: str) -> List[SectionInfo]:
        """문서 섹션 구조 파싱"""
        sections = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            match = self.section_pattern.match(line)
            if match:
                section_num = match.group(1)
                section_title = match.group(2).strip()
                level = section_num.count('.') + 1
                
                content_lines = [line]
                i += 1
                
                while i < len(lines):
                    next_line = lines[i]
                    if self.section_pattern.match(next_line):
                        break
                    content_lines.append(next_line)
                    i += 1
                
                sections.append(SectionInfo(
                    number=section_num,
                    title=section_title,
                    level=level,
                    start_line=len('\n'.join(lines[:i-len(content_lines)])),
                    content='\n'.join(content_lines)
                ))
            else:
                i += 1
        
        return sections
    
    def _chunk_section(
        self,
        section: SectionInfo,
        page_boundaries: List[Tuple[int, int, int]],
        layout_blocks: Optional[Dict[int, List[Dict]]]
    ) -> List[Tuple[str, Dict]]:
        """섹션 청킹 - 깔끔하게 분할"""
        chunks = []
        
        section_header = f"{section.number} {section.title}"
        
        # 빠른 길이 추정
        estimated_tokens = self._estimate_tokens_fast(section.content)
        
        if estimated_tokens <= self.max_chunk_tokens:
            actual_tokens = self._count_tokens(section.content)
            
            if actual_tokens <= self.max_chunk_tokens:
                pages = self._get_pages_for_text(
                    section.content, 
                    section.start_line,
                    page_boundaries
                )
                
                metadata = {
                    "type": "section",
                    "section": section_header,
                    "section_number": section.number,
                    "section_level": section.level,
                    "page": pages[0] if pages else 1,
                    "pages": pages,
                    "token_count": actual_tokens,
                }
                
                chunks.append((section.content, metadata))
                return chunks
        
        # 큰 섹션은 하위 단위로 분할
        subsections = self._find_subsections(section.content, section.number)
        
        if subsections:
            for subsec in subsections:
                subsec_chunks = self._chunk_subsection(
                    subsec,
                    section.start_line,
                    page_boundaries,
                    layout_blocks
                )
                chunks.extend(subsec_chunks)
        else:
            para_chunks = self._chunk_section_by_paragraphs(
                section,
                page_boundaries,
                layout_blocks
            )
            chunks.extend(para_chunks)
        
        return chunks
    
    def _find_subsections(self, text: str, parent_num: str) -> List[SectionInfo]:
        """하위 섹션 찾기"""
        subsections = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = self.section_pattern.match(line)
            
            if match:
                section_num = match.group(1)
                if section_num.startswith(parent_num + '.'):
                    section_title = match.group(2).strip()
                    level = section_num.count('.') + 1
                    
                    content_lines = [line]
                    i += 1
                    
                    while i < len(lines):
                        next_line = lines[i]
                        next_match = self.section_pattern.match(next_line)
                        
                        if next_match:
                            next_num = next_match.group(1)
                            if next_num.count('.') + 1 <= level:
                                break
                        
                        content_lines.append(next_line)
                        i += 1
                    
                    subsections.append(SectionInfo(
                        number=section_num,
                        title=section_title,
                        level=level,
                        start_line=i - len(content_lines),
                        content='\n'.join(content_lines)
                    ))
            else:
                i += 1
        
        return subsections
    
    def _chunk_subsection(
        self,
        subsection: SectionInfo,
        base_start: int,
        page_boundaries: List[Tuple[int, int, int]],
        layout_blocks: Optional[Dict[int, List[Dict]]]
    ) -> List[Tuple[str, Dict]]:
        """하위 섹션 청킹"""
        estimated_tokens = self._estimate_tokens_fast(subsection.content)
        
        if estimated_tokens <= self.max_chunk_tokens:
            content_tokens = self._count_tokens(subsection.content)
            
            if content_tokens <= self.max_chunk_tokens:
                pages = self._get_pages_for_text(
                    subsection.content,
                    base_start + subsection.start_line,
                    page_boundaries
                )
                
                metadata = {
                    "type": "subsection",
                    "section": f"{subsection.number} {subsection.title}",
                    "section_number": subsection.number,
                    "section_level": subsection.level,
                    "page": pages[0] if pages else 1,
                    "pages": pages,
                    "token_count": content_tokens,
                }
                
                return [(subsection.content, metadata)]
        
        # 크면 문단으로 분할
        return self._chunk_section_by_paragraphs(
            subsection,
            page_boundaries,
            layout_blocks,
            base_start
        )
    
    def _chunk_section_by_paragraphs(
        self,
        section: SectionInfo,
        page_boundaries: List[Tuple[int, int, int]],
        layout_blocks: Optional[Dict[int, List[Dict]]],
        base_start: int = 0
    ) -> List[Tuple[str, Dict]]:
        """섹션을 문단으로 청킹 - 깔끔하게"""
        chunks = []
        
        paragraphs = self._split_into_semantic_paragraphs(section.content)
        
        current_chunk = ""
        current_tokens = 0
        chunk_start_line = 0
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens_fast(para)
            
            if para_tokens > self.max_chunk_tokens:
                # 현재 청크 저장
                if current_chunk:
                    pages = self._get_pages_for_text(
                        current_chunk,
                        base_start + section.start_line + chunk_start_line,
                        page_boundaries
                    )
                    
                    actual_tokens = self._count_tokens(current_chunk)
                    
                    metadata = {
                        "type": "section_part",
                        "section": f"{section.number} {section.title}",
                        "section_number": section.number,
                        "page": pages[0] if pages else 1,
                        "pages": pages,
                        "token_count": actual_tokens,
                    }
                    chunks.append((current_chunk, metadata))
                    current_chunk = ""
                    current_tokens = 0
                
                # 큰 문단은 문장으로 분할
                sent_chunks = self._split_large_paragraph(para, section)
                chunks.extend(sent_chunks)
                
            elif current_tokens + para_tokens <= self.target_tokens:
                # 현재 청크에 추가
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    chunk_start_line = section.content.find(para)
                current_tokens += para_tokens
                
            else:
                # 현재 청크 완료하고 새 청크 시작
                if current_chunk:
                    pages = self._get_pages_for_text(
                        current_chunk,
                        base_start + section.start_line + chunk_start_line,
                        page_boundaries
                    )
                    
                    actual_tokens = self._count_tokens(current_chunk)
                    
                    metadata = {
                        "type": "section_part",
                        "section": f"{section.number} {section.title}",
                        "section_number": section.number,
                        "page": pages[0] if pages else 1,
                        "pages": pages,
                        "token_count": actual_tokens,
                    }
                    chunks.append((current_chunk, metadata))
                
                # 새 청크 시작
                current_chunk = para
                current_tokens = para_tokens
                chunk_start_line = section.content.find(para)
        
        # 마지막 청크
        if current_chunk:
            pages = self._get_pages_for_text(
                current_chunk,
                base_start + section.start_line + chunk_start_line,
                page_boundaries
            )
            
            actual_tokens = self._count_tokens(current_chunk)
            
            metadata = {
                "type": "section_part",
                "section": f"{section.number} {section.title}",
                "section_number": section.number,
                "page": pages[0] if pages else 1,
                "pages": pages,
                "token_count": actual_tokens,
            }
            chunks.append((current_chunk, metadata))
        
        return chunks
    
    def _split_into_semantic_paragraphs(self, text: str) -> List[str]:
        """의미론적 문단 분리"""
        paragraphs = []
        lines = text.split('\n')
        
        current_para = []
        in_bullet_group = False
        
        for line in lines:
            stripped = line.strip()
            
            # 빈 줄 처리
            if not stripped:
                if current_para and not in_bullet_group:
                    paragraphs.append('\n'.join(current_para))
                    current_para = []
                continue
            
            # 불릿 포인트 확인
            is_bullet = any(p.match(line) for p in self.bullet_patterns)
            
            if is_bullet:
                in_bullet_group = True
                current_para.append(line)
            elif in_bullet_group:
                # 들여쓰기 있으면 계속 같은 그룹
                if line.startswith('  ') or line.startswith('\t'):
                    current_para.append(line)
                else:
                    # 불릿 그룹 종료
                    if current_para:
                        paragraphs.append('\n'.join(current_para))
                        current_para = []
                    in_bullet_group = False
                    current_para.append(line)
            else:
                current_para.append(line)
        
        # 마지막 문단
        if current_para:
            paragraphs.append('\n'.join(current_para))
        
        return paragraphs
    
    def _split_large_paragraph(
        self,
        paragraph: str,
        section: SectionInfo
    ) -> List[Tuple[str, Dict]]:
        """큰 문단을 문장으로 분할"""
        chunks = []
        sentences = self.sentence_end.split(paragraph)
        
        current_chunk = ""
        current_tokens = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_tokens = self._estimate_tokens_fast(sent)
            
            if current_tokens + sent_tokens <= self.target_tokens:
                if current_chunk:
                    current_chunk += " " + sent
                else:
                    current_chunk = sent
                current_tokens += sent_tokens
            else:
                if current_chunk:
                    actual_tokens = self._count_tokens(current_chunk)
                    metadata = {
                        "type": "section_fragment",
                        "section": f"{section.number} {section.title}",
                        "section_number": section.number,
                        "page": 1,
                        "pages": [1],
                        "token_count": actual_tokens,
                    }
                    chunks.append((current_chunk, metadata))
                
                current_chunk = sent
                current_tokens = sent_tokens
        
        if current_chunk:
            actual_tokens = self._count_tokens(current_chunk)
            metadata = {
                "type": "section_fragment",
                "section": f"{section.number} {section.title}",
                "section_number": section.number,
                "page": 1,
                "pages": [1],
                "token_count": actual_tokens,
            }
            chunks.append((current_chunk, metadata))
        
        return chunks
    
    def _paragraph_based_chunking(
        self,
        pages_std: List[Tuple[int, str]],
        layout_blocks: Optional[Dict[int, List[Dict]]]
    ) -> List[Tuple[str, Dict]]:
        """섹션 구조 없는 경우 문단 기반"""
        chunks = []
        
        for page_no, text in pages_std:
            paragraphs = self._split_into_semantic_paragraphs(text)
            
            current_chunk = ""
            current_tokens = 0
            
            for para in paragraphs:
                para_tokens = self._estimate_tokens_fast(para)
                
                if current_tokens + para_tokens <= self.target_tokens:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                    current_tokens += para_tokens
                else:
                    if current_chunk:
                        actual_tokens = self._count_tokens(current_chunk)
                        metadata = {
                            "type": "paragraph_group",
                            "page": page_no,
                            "pages": [page_no],
                            "token_count": actual_tokens,
                        }
                        chunks.append((current_chunk, metadata))
                    
                    current_chunk = para
                    current_tokens = para_tokens
            
            if current_chunk:
                actual_tokens = self._count_tokens(current_chunk)
                metadata = {
                    "type": "paragraph_group",
                    "page": page_no,
                    "pages": [page_no],
                    "token_count": actual_tokens,
                }
                chunks.append((current_chunk, metadata))
        
        return chunks
    
    def _get_pages_for_text(
        self,
        text: str,
        text_start: int,
        page_boundaries: List[Tuple[int, int, int]]
    ) -> List[int]:
        """텍스트가 걸쳐있는 페이지 번호 리스트"""
        text_end = text_start + len(text)
        pages = []
        
        for start, end, page_no in page_boundaries:
            if not (text_end <= start or text_start >= end):
                pages.append(page_no)
        
        return sorted(set(pages)) if pages else [1]
    
    def _estimate_tokens_fast(self, text: str) -> int:
        """빠른 토큰 수 추정"""
        if not text:
            return 0
        words = len(text.split())
        return int(words * 1.3)
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산 (캐싱)"""
        if not text:
            return 0
        
        text_hash = hash(text)
        if text_hash in self._token_cache:
            self._cache_hits += 1
            return self._token_cache[text_hash]
        
        self._cache_misses += 1
        
        try:
            tokens = len(self.encoder(text))
        except:
            tokens = int(len(text.split()) * 1.3)
        
        if len(self._token_cache) < 10000:
            self._token_cache[text_hash] = tokens
        
        return tokens
    
    def _finalize_chunks(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """최종 청크 정리"""
        finalized = []
        
        for i, (text, meta) in enumerate(chunks):
            if not text.strip():
                continue
            
            # 청크 텍스트 정리
            text = self._clean_chunk_text(text)
            
            meta["chunk_index"] = i
            meta.setdefault("type", "unknown")
            meta.setdefault("page", 1)
            meta.setdefault("pages", [1])
            meta.setdefault("token_count", self._count_tokens(text))
            
            finalized.append((text, meta))
        
        return finalized
    
    def _clean_chunk_text(self, text: str) -> str:
        """청크 텍스트 정리 - 교정교열용"""
        # 과도한 공백 제거 (3줄 이상 → 2줄로)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # 연속된 공백/탭 → 단일 공백
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 줄 끝 공백 제거
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)
        
        return text.strip()


# ========== 외부 인터페이스 ==========

def english_technical_chunk_pages(
    pages_std: List[Tuple[int, str]], 
    encoder_fn: Callable,
    target_tokens: int = 800,
    overlap_tokens: int = 0,  # ⭐ 기본값 0
    layout_blocks: Optional[Dict[int, List[Dict]]] = None
) -> List[Tuple[str, Dict]]:
    """영어 기술 문서 전용 청킹 - 교정교열 최적화 (오버랩 없음)"""
    if not pages_std:
        return []
    
    chunker = EnglishTechnicalChunker(encoder_fn, target_tokens, overlap_tokens)
    return chunker.chunk_pages(pages_std, layout_blocks)