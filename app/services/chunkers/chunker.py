# app/services/chunkers/chunker.py
"""
스마트 청킹 - 통합 개선 버전
- SmartChunker와 SmartChunkerPlus 통합
- 레이아웃 정보 선택적 활용
- 코드 중복 제거
- 기존 인터페이스 유지
"""
from __future__ import annotations
import json
import re
from typing import List, Tuple, Dict, Optional, Callable
from app.services.enhanced_table_detector import EnhancedTableDetector


class SmartChunker:
    """통합 스마트 청커 - 레이아웃 정보 선택적 활용"""
    
    def __init__(self, encoder_fn: Callable, target_tokens: int = 400, overlap_tokens: int = 100):
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = 50
        self.max_chunk_tokens = target_tokens * 2
        self.table_detector = EnhancedTableDetector()
        
        # 구조 패턴
        self.header_pattern = re.compile(
            r'^(?:제\s*\d+\s*[조절항장편]|[A-Z0-9]+\.\s|\d+\.\d+\s)', 
            re.MULTILINE
        )
        self.list_pattern = re.compile(
            r'^[\s]*(?:\d+\.|[가나다라]\.|\([가나다라]\)|\d+\))', 
            re.MULTILINE
        )
        
    def chunk_pages(
        self, 
        pages_std: List[Tuple[int, str]], 
        layout_blocks: Optional[Dict[int, List[Dict]]] = None
    ) -> List[Tuple[str, Dict]]:
        """
        페이지별 스마트 청킹 - 레이아웃 정보 선택적 활용
        
        Args:
            pages_std: [(page_no, text), ...] 형태의 페이지 데이터
            layout_blocks: 레이아웃 정보 (선택사항)
        
        Returns:
            [(chunk_text, metadata), ...] 형태의 청크 리스트
        """
        
        if not pages_std:
            return []
        
        all_chunks = []
        
        for page_no, page_text in pages_std:
            if not page_text or not page_text.strip():
                continue
            
            # 문서 구조 분석
            structure_type = self._analyze_structure(page_text)
            
            # 레이아웃 정보 확인
            page_layout = layout_blocks.get(page_no) if layout_blocks else None
            
            # 구조 타입에 따른 청킹
            if structure_type == 'structured' and page_layout:
                # 구조화된 문서 + 레이아웃 정보
                page_chunks = self._structured_layout_chunking(
                    page_text, page_no, page_layout
                )
            elif structure_type == 'structured':
                # 구조화된 문서 (레이아웃 정보 없음)
                page_chunks = self._structured_chunking(page_text, page_no)
            else:
                # 일반 문서
                page_chunks = self._paragraph_chunking(page_text, page_no)
            
            all_chunks.extend(page_chunks)
        
        return self._finalize_chunks(all_chunks)
    
    def _analyze_structure(self, text: str) -> str:
        """문서 구조 분석"""
        
        lines = [l for l in text.split('\n') if l.strip()]
        if not lines:
            return 'unstructured'
        
        # 헤더 패턴 비율
        header_count = len(self.header_pattern.findall(text))
        list_count = len(self.list_pattern.findall(text))
        
        structure_ratio = (header_count + list_count) / len(lines)
        
        # 구조화된 문서: 헤더/리스트 패턴이 20% 이상
        if structure_ratio > 0.2:
            return 'structured'
        
        return 'unstructured'
    
    def _structured_layout_chunking(
        self, 
        text: str, 
        page_no: int, 
        layout: List[Dict]
    ) -> List[Tuple[str, Dict]]:
        """구조화된 문서 + 레이아웃 정보 활용"""
        
        chunks = []
        
        # 레이아웃 블록을 Y 좌표로 정렬
        sorted_blocks = sorted(
            layout,
            key=lambda b: b.get('bbox', {}).get('y0', 0)
        )
        
        current_section = []
        current_tokens = 0
        current_header = ""
        
        for block in sorted_blocks:
            block_text = block.get('text', '').strip()
            if not block_text:
                continue
            
            # 헤더 감지
            is_header = bool(self.header_pattern.match(block_text))
            block_tokens = self._count_tokens(block_text)
            
            if is_header:
                # 새 섹션 시작
                if current_section:
                    chunk_text = '\n\n'.join(current_section)
                    chunks.append(
                        self._create_chunk(chunk_text, page_no, 'structured', current_header)
                    )
                
                current_section = [block_text]
                current_tokens = block_tokens
                current_header = block_text[:50]  # 헤더 저장
                
            else:
                # 섹션 내용 누적
                if current_tokens + block_tokens <= self.target_tokens:
                    current_section.append(block_text)
                    current_tokens += block_tokens
                else:
                    # 현재 섹션 청크화
                    if current_section:
                        chunk_text = '\n\n'.join(current_section)
                        chunks.append(
                            self._create_chunk(chunk_text, page_no, 'structured', current_header)
                        )
                    
                    current_section = [block_text]
                    current_tokens = block_tokens
        
        # 마지막 섹션
        if current_section:
            chunk_text = '\n\n'.join(current_section)
            chunks.append(
                self._create_chunk(chunk_text, page_no, 'structured', current_header)
            )
        
        return chunks
    
    def _structured_chunking(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """구조화된 문서 청킹 (레이아웃 정보 없음)"""
        
        chunks = []
        paragraphs = self._split_paragraphs(text)
        
        current_chunk = []
        current_tokens = 0
        current_header = ""
        
        for para in paragraphs:
            is_header = bool(self.header_pattern.match(para))
            para_tokens = self._count_tokens(para)
            
            if is_header:
                # 새 섹션
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(
                        self._create_chunk(chunk_text, page_no, 'structured', current_header)
                    )
                
                current_chunk = [para]
                current_tokens = para_tokens
                current_header = para[:50]
                
            else:
                # 누적
                if current_tokens + para_tokens <= self.target_tokens:
                    current_chunk.append(para)
                    current_tokens += para_tokens
                else:
                    if current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunks.append(
                            self._create_chunk(chunk_text, page_no, 'structured', current_header)
                        )
                    
                    if para_tokens > self.max_chunk_tokens:
                        sub_chunks = self._split_large_para(para, page_no)
                        chunks.extend(sub_chunks)
                        current_chunk = []
                        current_tokens = 0
                    else:
                        current_chunk = [para]
                        current_tokens = para_tokens
        
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(
                self._create_chunk(chunk_text, page_no, 'structured', current_header)
            )
        
        return chunks
    
    def _paragraph_chunking(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """일반 문단 기반 청킹"""
        
        chunks = []
        paragraphs = self._split_paragraphs(text)
        
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            if current_tokens + para_tokens <= self.target_tokens:
                current_chunk.append(para)
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, page_no, 'paragraph', ''))
                
                if para_tokens > self.max_chunk_tokens:
                    sub_chunks = self._split_large_para(para, page_no)
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_tokens = 0
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
        
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, page_no, 'paragraph', ''))
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """문단 분할"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_large_para(self, para: str, page_no: int) -> List[Tuple[str, Dict]]:
        """큰 문단 분할"""
        
        chunks = []
        sentences = self._split_sentences(para)
        
        current_chunk = []
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            
            if current_tokens + sent_tokens <= self.target_tokens:
                current_chunk.append(sent)
                current_tokens += sent_tokens
            else:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, page_no, 'paragraph', ''))
                
                current_chunk = [sent]
                current_tokens = sent_tokens
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, page_no, 'paragraph', ''))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 분할"""
        sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z가-힣])')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(
        self, 
        text: str, 
        page_no: int, 
        chunk_type: str,
        section: str
    ) -> Tuple[str, Dict]:
        """청크 생성"""
        
        clean_text = self._clean_text(text)
        
        meta = {
            'page': page_no,
            'pages': [page_no],
            'type': chunk_type,
            'section': section,
            'token_count': self._count_tokens(clean_text),
        }
        
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        final_text = meta_line + "\n" + clean_text
        
        return (final_text, meta)
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        text = re.sub(r'\b인접행\s*묶음\b', '', text)
        text = re.sub(r'\b[가-힣]*\s*묶음\b', '', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    
    def _finalize_chunks(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """최종 청크 검증"""
        
        finalized = []
        
        for text, meta in chunks:
            if meta.get('token_count', 0) >= self.min_chunk_tokens:
                finalized.append((text, meta))
        
        return finalized
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        if not text:
            return 0
        try:
            return len(self.encoder(text))
        except:
            return int(len(text.split()) * 1.3)


# SmartChunkerPlus는 SmartChunker의 별칭으로 유지 (하위 호환성)
class SmartChunkerPlus(SmartChunker):
    """
    SmartChunker의 별칭 (하위 호환성)
    이제 SmartChunker가 모든 기능 포함
    """
    pass


# 외부 인터페이스 함수들 (기존 함수명 유지)
def smart_chunk_pages(
    pages_std: List[Tuple[int, str]], 
    encoder_fn: Callable,
    target_tokens: int = 400,
    overlap_tokens: int = 100,
    layout_blocks: Optional[Dict[int, List[Dict]]] = None
) -> List[Tuple[str, Dict]]:
    """
    스마트 청킹 함수 (기본 버전)
    
    Args:
        pages_std: [(page_no, text), ...] 형태의 페이지 데이터
        encoder_fn: 토큰 인코딩 함수
        target_tokens: 목표 토큰 수
        overlap_tokens: 오버랩 토큰 수
        layout_blocks: 레이아웃 블록 정보 (선택사항)
    
    Returns:
        [(chunk_text, metadata), ...] 형태의 청크 리스트
    """
    
    if not pages_std:
        return []
    
    chunker = SmartChunker(encoder_fn, target_tokens, overlap_tokens)
    return chunker.chunk_pages(pages_std, layout_blocks)


def smart_chunk_pages_plus(
    pages_std: List[Tuple[int, str]], 
    encoder_fn: Callable,
    target_tokens: int = 400,
    overlap_tokens: int = 100,
    layout_blocks: Optional[Dict[int, List[Dict]]] = None
) -> List[Tuple[str, Dict]]:
    """
    스마트 청킹 함수 (플러스 버전) - 이제 기본 버전과 동일
    하위 호환성을 위해 유지
    
    Args:
        pages_std: [(page_no, text), ...] 형태의 페이지 데이터
        encoder_fn: 토큰 인코딩 함수
        target_tokens: 목표 토큰 수
        overlap_tokens: 오버랩 토큰 수
        layout_blocks: 레이아웃 블록 정보 (선택사항)
    
    Returns:
        [(chunk_text, metadata), ...] 형태의 청크 리스트
    """
    
    # 이제 기본 smart_chunk_pages와 동일
    return smart_chunk_pages(pages_std, encoder_fn, target_tokens, overlap_tokens, layout_blocks)