# app/services/english_technical_chunker.py
"""
영어 기술 문서 (IAEA, Nuclear Standards 등) 특화 청킹 모듈
- 섹션 번호 인식 (1., 1.1, 1.1.1, 3.2.1 등)
- 긴 문단 보존 (페이지 넘어가도 OK)
- 계층 구조 유지 (bullet points, sub-items)
- 문맥 연속성 최우선
"""
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from app.services.enhanced_table_detector import EnhancedTableDetector, TableRegion


@dataclass
class SectionInfo:
    """섹션 정보"""
    number: str  # "3.1", "3.2.1" 등
    title: str
    level: int  # 1, 2, 3 (depth)
    start_line: int
    content: str = ""


class EnglishTechnicalChunker:
    """영어 기술 문서 전용 청킹 클래스"""
    
    def __init__(self, encoder_fn: Callable, target_tokens: int = 800, overlap_tokens: int = 100):
        """
        Args:
            encoder_fn: 토큰 인코더 함수
            target_tokens: 목표 토큰 수 (영어 문서는 더 길게 설정 권장: 600-1000)
            overlap_tokens: 오버랩 토큰 수
        """
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = 100  # 최소 청크 크기
        self.max_chunk_tokens = target_tokens * 3  # 최대 청크 크기 (더 관대하게)
        self.table_detector = EnhancedTableDetector()
        
        # 섹션 번호 패턴 (1. 1.1 1.1.1 등)
        self.section_pattern = re.compile(
            r'^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s,\-\']+?)(?:\s*\n|$)',
            re.MULTILINE
        )
        
        # 대문자로 시작하는 제목 패턴 (SECTION TITLE, CHAPTER 등)
        self.title_pattern = re.compile(
            r'^([A-Z][A-Z\s]{3,})(?:\s*\n|$)',
            re.MULTILINE
        )
        
        # 불릿 포인트 패턴
        self.bullet_patterns = [
            re.compile(r'^[\s]*[\-•●○▪▫]\s+', re.MULTILINE),  # - • ● 등
            re.compile(r'^[\s]*\([a-z]\)\s+', re.MULTILINE),  # (a) (b) (c)
            re.compile(r'^[\s]*\([ivxIVX]+\)\s+', re.MULTILINE),  # (i) (ii) (iii)
            re.compile(r'^[\s]*[a-z]\)\s+', re.MULTILINE),  # a) b) c)
        ]
        
        # 문장 종결 패턴 (약어 제외)
        self.sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
    def chunk_pages(
        self, 
        pages_std: List[Tuple[int, str]], 
        layout_blocks: Optional[Dict[int, List[Dict]]] = None
    ) -> List[Tuple[str, Dict]]:
        """
        페이지별 영어 기술 문서 청킹
        
        Args:
            pages_std: [(page_no, text), ...]
            layout_blocks: {page_no: [{'text': str, 'bbox': {...}}, ...]}
        
        Returns:
            [(chunk_text, metadata), ...]
        """
        if not pages_std:
            return []
        
        all_chunks = []
        
        # 전체 문서를 하나의 텍스트로 결합 (페이지 경계 정보는 유지)
        full_text = ""
        page_boundaries = []  # (char_start, char_end, page_no)
        
        for page_no, text in pages_std:
            start_pos = len(full_text)
            full_text += text + "\n\n"
            end_pos = len(full_text)
            page_boundaries.append((start_pos, end_pos, page_no))
        
        # 섹션 단위로 파싱
        sections = self._parse_sections(full_text)
        
        if not sections:
            # 섹션이 없으면 문단 기반 청킹
            return self._paragraph_based_chunking(pages_std, layout_blocks)
        
        # 각 섹션을 청킹
        for section in sections:
            section_chunks = self._chunk_section(
                section, 
                page_boundaries,
                layout_blocks
            )
            all_chunks.extend(section_chunks)
        
        return self._finalize_chunks(all_chunks)
    
    def _parse_sections(self, text: str) -> List[SectionInfo]:
        """문서에서 섹션 구조 파싱"""
        sections = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 섹션 번호 + 제목 매칭
            match = self.section_pattern.match(line)
            if match:
                section_num = match.group(1)
                section_title = match.group(2).strip()
                level = section_num.count('.') + 1
                
                # 다음 섹션까지의 내용 수집
                content_lines = [line]
                i += 1
                
                while i < len(lines):
                    next_line = lines[i]
                    # 다음 섹션 시작이면 중단
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
        """섹션을 의미 단위로 청킹"""
        chunks = []
        
        # 섹션 헤더
        section_header = f"{section.number} {section.title}"
        
        # 섹션 내용 토큰 수 확인
        section_tokens = self._count_tokens(section.content)
        
        # 섹션이 목표 크기 이하면 통째로 청크
        if section_tokens <= self.max_chunk_tokens:
            # 페이지 정보 추출
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
                "token_count": section_tokens,
            }
            
            chunks.append((section.content, metadata))
            return chunks
        
        # 섹션이 크면 하위 단위로 분할
        # 1. 하위 섹션 확인
        subsections = self._find_subsections(section.content, section.number)
        
        if subsections:
            # 하위 섹션이 있으면 그것을 기준으로 분할
            for subsec in subsections:
                subsec_chunks = self._chunk_subsection(
                    subsec,
                    section.start_line,
                    page_boundaries,
                    layout_blocks
                )
                chunks.extend(subsec_chunks)
        else:
            # 하위 섹션이 없으면 문단 기반 분할
            para_chunks = self._chunk_section_by_paragraphs(
                section,
                page_boundaries,
                layout_blocks
            )
            chunks.extend(para_chunks)
        
        return chunks
    
    def _find_subsections(self, text: str, parent_num: str) -> List[SectionInfo]:
        """텍스트 내에서 하위 섹션 찾기"""
        subsections = []
        lines = text.split('\n')
        
        # 예상되는 하위 섹션 패턴 (예: 3.1 내의 3.1.1, 3.1.2)
        parent_level = parent_num.count('.') + 1
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = self.section_pattern.match(line)
            
            if match:
                section_num = match.group(1)
                # 직계 하위 섹션인지 확인
                if section_num.startswith(parent_num + '.'):
                    section_title = match.group(2).strip()
                    level = section_num.count('.') + 1
                    
                    # 내용 수집
                    content_lines = [line]
                    i += 1
                    
                    while i < len(lines):
                        next_line = lines[i]
                        next_match = self.section_pattern.match(next_line)
                        
                        if next_match:
                            next_num = next_match.group(1)
                            # 같은 레벨 또는 상위 레벨 섹션이면 중단
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
        content_tokens = self._count_tokens(subsection.content)
        
        if content_tokens <= self.max_chunk_tokens:
            # 작으면 통째로
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
        """섹션을 문단 단위로 청킹 (불릿 포인트 그룹 보존)"""
        chunks = []
        
        # 문단 분리 (불릿 포인트 그룹은 하나의 문단으로)
        paragraphs = self._split_into_semantic_paragraphs(section.content)
        
        current_chunk = ""
        current_tokens = 0
        chunk_start_line = 0
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            # 문단 자체가 너무 크면 분할
            if para_tokens > self.max_chunk_tokens:
                # 현재 청크 저장
                if current_chunk:
                    pages = self._get_pages_for_text(
                        current_chunk,
                        base_start + section.start_line + chunk_start_line,
                        page_boundaries
                    )
                    
                    metadata = {
                        "type": "section_part",
                        "section": f"{section.number} {section.title}",
                        "section_number": section.number,
                        "page": pages[0] if pages else 1,
                        "pages": pages,
                        "token_count": current_tokens,
                    }
                    chunks.append((current_chunk, metadata))
                    current_chunk = ""
                    current_tokens = 0
                
                # 큰 문단을 문장 단위로 분할
                sent_chunks = self._split_large_paragraph(para, section)
                chunks.extend(sent_chunks)
                
            elif current_tokens + para_tokens <= self.target_tokens:
                # 현재 청크에 추가 가능
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    chunk_start_line = section.content.find(para)
                current_tokens += para_tokens
                
            else:
                # 현재 청크 저장하고 새 청크 시작
                if current_chunk:
                    pages = self._get_pages_for_text(
                        current_chunk,
                        base_start + section.start_line + chunk_start_line,
                        page_boundaries
                    )
                    
                    metadata = {
                        "type": "section_part",
                        "section": f"{section.number} {section.title}",
                        "section_number": section.number,
                        "page": pages[0] if pages else 1,
                        "pages": pages,
                        "token_count": current_tokens,
                    }
                    chunks.append((current_chunk, metadata))
                
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
            
            metadata = {
                "type": "section_part",
                "section": f"{section.number} {section.title}",
                "section_number": section.number,
                "page": pages[0] if pages else 1,
                "pages": pages,
                "token_count": current_tokens,
            }
            chunks.append((current_chunk, metadata))
        
        return chunks
    
    def _split_into_semantic_paragraphs(self, text: str) -> List[str]:
        """
        의미론적 문단 분리 (불릿 포인트 그룹 보존)
        """
        paragraphs = []
        lines = text.split('\n')
        
        current_para = []
        in_bullet_group = False
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                # 빈 줄: 문단 구분자
                if current_para:
                    if in_bullet_group:
                        # 불릿 그룹은 계속 유지
                        current_para.append(line)
                    else:
                        # 일반 문단 종료
                        paragraphs.append('\n'.join(current_para))
                        current_para = []
                continue
            
            # 불릿 포인트 체크
            is_bullet = any(p.match(line) for p in self.bullet_patterns)
            
            if is_bullet:
                in_bullet_group = True
                current_para.append(line)
            elif in_bullet_group:
                # 불릿 그룹 내의 연속 텍스트
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
        """큰 문단을 문장 단위로 분할"""
        chunks = []
        
        # 문장 분리
        sentences = self.sentence_end.split(paragraph)
        
        current_chunk = ""
        current_tokens = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_tokens = self._count_tokens(sent)
            
            if current_tokens + sent_tokens <= self.target_tokens:
                if current_chunk:
                    current_chunk += " " + sent
                else:
                    current_chunk = sent
                current_tokens += sent_tokens
            else:
                if current_chunk:
                    metadata = {
                        "type": "section_fragment",
                        "section": f"{section.number} {section.title}",
                        "section_number": section.number,
                        "page": 1,  # 페이지 정보는 상위에서 처리
                        "pages": [1],
                        "token_count": current_tokens,
                    }
                    chunks.append((current_chunk, metadata))
                
                current_chunk = sent
                current_tokens = sent_tokens
        
        if current_chunk:
            metadata = {
                "type": "section_fragment",
                "section": f"{section.number} {section.title}",
                "section_number": section.number,
                "page": 1,
                "pages": [1],
                "token_count": current_tokens,
            }
            chunks.append((current_chunk, metadata))
        
        return chunks
    
    def _paragraph_based_chunking(
        self,
        pages_std: List[Tuple[int, str]],
        layout_blocks: Optional[Dict[int, List[Dict]]]
    ) -> List[Tuple[str, Dict]]:
        """섹션 구조가 없는 경우 문단 기반 청킹"""
        chunks = []
        
        for page_no, text in pages_std:
            paragraphs = self._split_into_semantic_paragraphs(text)
            
            current_chunk = ""
            current_tokens = 0
            
            for para in paragraphs:
                para_tokens = self._count_tokens(para)
                
                if current_tokens + para_tokens <= self.target_tokens:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                    current_tokens += para_tokens
                else:
                    if current_chunk:
                        metadata = {
                            "type": "paragraph_group",
                            "page": page_no,
                            "pages": [page_no],
                            "token_count": current_tokens,
                        }
                        chunks.append((current_chunk, metadata))
                    
                    current_chunk = para
                    current_tokens = para_tokens
            
            if current_chunk:
                metadata = {
                    "type": "paragraph_group",
                    "page": page_no,
                    "pages": [page_no],
                    "token_count": current_tokens,
                }
                chunks.append((current_chunk, metadata))
        
        return chunks
    
    def _get_pages_for_text(
        self,
        text: str,
        text_start: int,
        page_boundaries: List[Tuple[int, int, int]]
    ) -> List[int]:
        """텍스트가 걸쳐있는 페이지 번호 리스트 반환"""
        text_end = text_start + len(text)
        pages = []
        
        for start, end, page_no in page_boundaries:
            # 텍스트가 이 페이지 범위와 겹치는지 확인
            if not (text_end <= start or text_start >= end):
                pages.append(page_no)
        
        return sorted(set(pages)) if pages else [1]
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        try:
            return len(self.encoder(text))
        except:
            # 폴백: 단어 수 기반 추정
            return len(text.split()) * 1.3
    
    def _finalize_chunks(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """최종 청크 정리"""
        finalized = []
        
        for i, (text, meta) in enumerate(chunks):
            if not text.strip():
                continue
            
            # 메타데이터 정규화
            meta["chunk_index"] = i
            meta.setdefault("type", "unknown")
            meta.setdefault("page", 1)
            meta.setdefault("pages", [1])
            meta.setdefault("token_count", self._count_tokens(text))
            
            finalized.append((text, meta))
        
        return finalized


# ========== 외부 인터페이스 함수 ==========

def english_technical_chunk_pages(
    pages_std: List[Tuple[int, str]], 
    encoder_fn: Callable,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
    layout_blocks: Optional[Dict[int, List[Dict]]] = None
) -> List[Tuple[str, Dict]]:
    """
    영어 기술 문서 전용 청킹 함수
    
    Args:
        pages_std: [(page_no, text), ...]
        encoder_fn: 토큰 인코더 함수
        target_tokens: 목표 토큰 수 (권장: 600-1000)
        overlap_tokens: 오버랩 토큰 수
        layout_blocks: 레이아웃 정보
    
    Returns:
        [(chunk_text, metadata), ...]
    """
    if not pages_std:
        return []
    
    chunker = EnglishTechnicalChunker(encoder_fn, target_tokens, overlap_tokens)
    return chunker.chunk_pages(pages_std, layout_blocks)