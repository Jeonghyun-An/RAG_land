# app/services/layout_chunker.py
"""
레이아웃 인지 청킹 - 개선 버전
- bbox 정보를 보조로만 사용
- 텍스트 흐름 우선
- 하드코딩 제거
- 기존 인터페이스 유지
"""
from __future__ import annotations
import json
import re
from typing import List, Tuple, Dict, Optional, Callable
from app.services.enhanced_table_detector import EnhancedTableDetector


class LayoutAwareChunker:
    """레이아웃 인지 청커 - 안정성과 단순성 중심"""
    
    def __init__(self, encoder_fn: Callable, target_tokens: int = 400, overlap_tokens: int = 100):
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = 50
        self.max_chunk_tokens = target_tokens * 2
        self.table_detector = EnhancedTableDetector()
        
    def chunk_pages(
        self, 
        pages_std: List[Tuple[int, str]], 
        layout_blocks: Dict[int, List[Dict]],
        slide_rows: int = 4
    ) -> List[Tuple[str, Dict]]:
        """레이아웃 정보를 활용한 페이지별 청킹"""
        
        if not pages_std:
            return []
        
        all_chunks = []
        
        for page_no, page_text in pages_std:
            if not page_text or not page_text.strip():
                continue
            
            # 레이아웃 블록 가져오기
            blocks_data = layout_blocks.get(page_no, [])
            
            if not blocks_data:
                # 레이아웃 정보 없으면 텍스트 기반 청킹
                page_chunks = self._text_based_chunking(page_text, page_no)
            else:
                # 레이아웃 정보 활용
                page_chunks = self._layout_guided_chunking(page_text, page_no, blocks_data)
            
            all_chunks.extend(page_chunks)
        
        # 최종 정리
        return self._finalize_chunks(all_chunks)
    
    def _layout_guided_chunking(
        self, 
        text: str, 
        page_no: int, 
        blocks: List[Dict]
    ) -> List[Tuple[str, Dict]]:
        """
        레이아웃 정보를 가이드로 사용한 청킹
        - bbox는 보조 정보로만 활용
        - 텍스트 흐름 우선
        """
        chunks = []
        
        # 블록을 Y 좌표 기준으로 정렬 (위→아래)
        sorted_blocks = sorted(
            blocks, 
            key=lambda b: b.get('bbox', {}).get('y0', 0)
        )
        
        current_group = []
        current_tokens = 0
        
        for block in sorted_blocks:
            block_text = block.get('text', '').strip()
            if not block_text:
                continue
            
            block_tokens = self._count_tokens(block_text)
            
            # 표 영역 감지
            is_table = self._is_table_block(block_text)
            
            if is_table:
                # 표는 단독 청크로 처리
                if current_group:
                    chunk_text = '\n\n'.join(current_group)
                    chunks.append(self._create_chunk(chunk_text, page_no, 'text'))
                    current_group = []
                    current_tokens = 0
                
                # 표 청크 생성
                chunks.append(self._create_chunk(block_text, page_no, 'table'))
                continue
            
            # 일반 텍스트 누적
            if current_tokens + block_tokens <= self.target_tokens:
                current_group.append(block_text)
                current_tokens += block_tokens
            else:
                # 현재 그룹 청크화
                if current_group:
                    chunk_text = '\n\n'.join(current_group)
                    chunks.append(self._create_chunk(chunk_text, page_no, 'text'))
                
                # 블록이 너무 크면 분할
                if block_tokens > self.max_chunk_tokens:
                    sub_chunks = self._split_large_text(block_text, page_no)
                    chunks.extend(sub_chunks)
                    current_group = []
                    current_tokens = 0
                else:
                    current_group = [block_text]
                    current_tokens = block_tokens
        
        # 마지막 그룹 처리
        if current_group:
            chunk_text = '\n\n'.join(current_group)
            chunks.append(self._create_chunk(chunk_text, page_no, 'text'))
        
        return chunks
    
    def _text_based_chunking(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """레이아웃 정보 없을 때의 폴백 - 문단 기반 청킹"""
        
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
                # 현재 청크 완료
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, page_no, 'text'))
                
                # 문단이 너무 크면 분할
                if para_tokens > self.max_chunk_tokens:
                    sub_chunks = self._split_large_text(para, page_no)
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_tokens = 0
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
        
        # 마지막 청크
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, page_no, 'text'))
        
        return chunks
    
    def _is_table_block(self, text: str) -> bool:
        """표 블록 감지"""
        # 표 패턴
        table_patterns = [
            r'[\|\+\-]{3,}',  # ASCII 테이블
            r'[┌┐└┘├┤┬┴┼─│]{3,}',  # 박스 문자
            r'\t.*\t',  # 탭 구분
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """텍스트를 문단으로 분할"""
        # 이중 개행 기준
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_large_text(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """큰 텍스트를 문장 단위로 분할"""
        
        chunks = []
        sentences = self._split_sentences(text)
        
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
                    chunks.append(self._create_chunk(chunk_text, page_no, 'text'))
                
                current_chunk = [sent]
                current_tokens = sent_tokens
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, page_no, 'text'))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 분할"""
        # 한국어/영어 문장 종결
        sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z가-힣])')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(self, text: str, page_no: int, chunk_type: str) -> Tuple[str, Dict]:
        """청크 생성"""
        
        # 텍스트 정리
        clean_text = self._clean_chunk_text(text)
        
        # 메타데이터
        meta = {
            'page': page_no,
            'pages': [page_no],
            'type': chunk_type,
            'token_count': self._count_tokens(clean_text),
            'section': '',
        }
        
        # META 라인 추가
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        final_text = meta_line + "\n" + clean_text
        
        return (final_text, meta)
    
    def _clean_chunk_text(self, text: str) -> str:
        """청크 텍스트 정리"""
        # 불필요한 라벨 제거
        text = re.sub(r'\b인접행\s*묶음\b', '', text)
        text = re.sub(r'\b[가-힣]*\s*묶음\b', '', text)
        
        # 과도한 공백 정리
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _strip_meta_line(self, text: str) -> str:
        """META 라인 제거"""
        if text.startswith("META:"):
            nl_pos = text.find("\n")
            return text[nl_pos + 1:] if nl_pos != -1 else ""
        return text
    
    def _finalize_chunks(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """최종 청크 검증 및 정리"""
        
        finalized = []
        
        for text, meta in chunks:
            # 최소 토큰 확인
            if meta.get('token_count', 0) < self.min_chunk_tokens:
                continue
            
            finalized.append((text, meta))
        
        return finalized
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        if not text:
            return 0
        try:
            return len(self.encoder(text))
        except:
            # 폴백: 대략 추정
            return int(len(text.split()) * 1.3)


def layout_aware_chunks(pages_std: List[Tuple[int, str]], 
                       encoder_fn: Callable,
                       target_tokens: int,
                       overlap_tokens: int,
                       slide_rows: int = 4,
                       layout_blocks: Optional[Dict[int, List[Dict]]] = None) -> List[Tuple[str, Dict]]:
    """
    레이아웃 인지 청킹 함수 (기존 인터페이스 호환)
    
    Args:
        pages_std: [(page_no, text), ...] 형태의 페이지 데이터
        encoder_fn: 토큰 인코딩 함수
        target_tokens: 목표 토큰 수
        overlap_tokens: 오버랩 토큰 수 (현재 미사용)
        slide_rows: 슬라이딩 행 수 (현재 미사용)
        layout_blocks: 레이아웃 블록 정보
    
    Returns:
        [(chunk_text, metadata), ...] 형태의 청크 리스트
    """
    
    if not pages_std:
        return []
    
    # 레이아웃 블록 정보가 없으면 빈 결과 반환 (다른 청커로 폴백)
    if not layout_blocks:
        return []
    
    chunker = LayoutAwareChunker(encoder_fn, target_tokens, overlap_tokens)
    return chunker.chunk_pages(pages_std, layout_blocks, slide_rows)