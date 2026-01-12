# app/services/chunkers/simple_proofreading_chunker.py
"""
교정·교열 전용 단순 청킹 모듈
- 복잡한 레이아웃 분석 없이 텍스트만 순서대로 추출
- 페이지 정보 보존
- 고정 토큰 크기로 분할
- 사용자 수정 → 재청킹이 쉬운 구조
"""
from __future__ import annotations
import json
from typing import List, Tuple, Dict, Callable, Optional


class SimpleProofreadingChunker:
    """교정·교열 전용 단순 청킹 클래스"""
    
    def __init__(self, encoder_fn: Callable, target_tokens: int = 400, overlap_tokens: int = 50):
        """
        Args:
            encoder_fn: 토큰 인코딩 함수
            target_tokens: 목표 토큰 수 (기본 400)
            overlap_tokens: 청크 간 오버랩 (기본 50)
        """
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
    
    def chunk_by_page(self, pages: List[Tuple[int, str]]) -> List[Tuple[str, Dict]]:
        """
        페이지별 청킹 (가장 단순)
        - 각 페이지를 하나의 청크로
        - 페이지가 너무 크면 토큰 기준으로 분할
        
        Args:
            pages: [(page_no, text), ...] 형태의 페이지 데이터
        
        Returns:
            [(chunk_text, metadata), ...] 형태의 청크 리스트
        """
        chunks = []
        
        for page_no, text in pages:
            if not text or not text.strip():
                continue
            
            page_tokens = self._count_tokens(text)
            
            if page_tokens <= self.target_tokens:
                # 페이지 전체가 한 청크
                chunks.append(self._create_chunk(text, page_no, [page_no]))
            else:
                # 페이지가 너무 크면 분할
                page_chunks = self._split_large_page(text, page_no)
                chunks.extend(page_chunks)
        
        return chunks
    
    def chunk_by_fixed_tokens(self, pages: List[Tuple[int, str]]) -> List[Tuple[str, Dict]]:
        """
        고정 토큰 크기로 청킹 (슬라이딩 윈도우)
        - 전체 텍스트를 하나로 합친 후 토큰 기준으로 분할
        - 페이지 정보는 메타데이터에 보존
        - 오버랩 적용
        
        Args:
            pages: [(page_no, text), ...] 형태의 페이지 데이터
        
        Returns:
            [(chunk_text, metadata), ...] 형태의 청크 리스트
        """
        if not pages:
            return []
        
        # 전체 텍스트 + 페이지 매핑 구축
        full_text_parts = []
        char_to_page = []  # 각 문자가 어느 페이지에서 왔는지
        
        for page_no, text in pages:
            if not text or not text.strip():
                continue
            
            start_idx = len(''.join(full_text_parts))
            full_text_parts.append(text)
            
            # 이 페이지 텍스트의 문자들에 페이지 번호 매핑
            for _ in range(len(text)):
                char_to_page.append(page_no)
            
            # 페이지 간 구분자 (선택사항)
            full_text_parts.append('\n\n')
            char_to_page.extend([page_no, page_no])
        
        full_text = ''.join(full_text_parts)
        
        # 슬라이딩 윈도우로 청킹
        chunks = []
        start_char = 0
        
        while start_char < len(full_text):
            # 목표 토큰 크기만큼 추출
            end_char = self._find_chunk_end(
                full_text, start_char, self.target_tokens
            )
            
            chunk_text = full_text[start_char:end_char].strip()
            
            if chunk_text:
                # 이 청크가 걸쳐 있는 페이지들 찾기
                chunk_pages = self._get_pages_for_range(
                    char_to_page, start_char, end_char
                )
                
                chunks.append(self._create_chunk(
                    chunk_text, 
                    chunk_pages[0] if chunk_pages else 1,
                    chunk_pages
                ))
            
            # 다음 청크 시작 위치 (오버랩 적용)
            overlap_chars = self._tokens_to_chars(
                full_text, end_char, self.overlap_tokens, reverse=True
            )
            start_char = end_char - overlap_chars
            
            # 무한루프 방지
            if start_char >= end_char - 10:
                start_char = end_char
        
        return chunks
    
    def chunk_by_paragraph(self, pages: List[Tuple[int, str]]) -> List[Tuple[str, Dict]]:
        """
        문단 기준 청킹
        - 빈 줄(\\n\\n)을 기준으로 문단 구분
        - 문단들을 토큰 제한 내에서 묶음
        - 가장 자연스러운 청킹
        
        Args:
            pages: [(page_no, text), ...] 형태의 페이지 데이터
        
        Returns:
            [(chunk_text, metadata), ...] 형태의 청크 리스트
        """
        chunks = []
        
        for page_no, text in pages:
            if not text or not text.strip():
                continue
            
            # 문단 단위로 분리
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            current_chunk_paras = []
            current_tokens = 0
            
            for para in paragraphs:
                para_tokens = self._count_tokens(para)
                
                # 단일 문단이 너무 크면 강제 분할
                if para_tokens > self.target_tokens:
                    # 현재까지 모은 문단들 저장
                    if current_chunk_paras:
                        chunk_text = '\n\n'.join(current_chunk_paras)
                        chunks.append(self._create_chunk(chunk_text, page_no, [page_no]))
                        current_chunk_paras = []
                        current_tokens = 0
                    
                    # 큰 문단 분할
                    para_chunks = self._split_large_paragraph(para, page_no)
                    chunks.extend(para_chunks)
                    continue
                
                # 토큰 제한 내에 들어가면 추가
                if current_tokens + para_tokens <= self.target_tokens:
                    current_chunk_paras.append(para)
                    current_tokens += para_tokens
                else:
                    # 현재 청크 완성
                    if current_chunk_paras:
                        chunk_text = '\n\n'.join(current_chunk_paras)
                        chunks.append(self._create_chunk(chunk_text, page_no, [page_no]))
                    
                    # 새 청크 시작
                    current_chunk_paras = [para]
                    current_tokens = para_tokens
            
            # 마지막 청크
            if current_chunk_paras:
                chunk_text = '\n\n'.join(current_chunk_paras)
                chunks.append(self._create_chunk(chunk_text, page_no, [page_no]))
        
        return chunks
    
    def _split_large_page(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """큰 페이지를 여러 청크로 분할"""
        chunks = []
        words = text.split()
        
        current_words = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self._count_tokens(word)
            
            if current_tokens + word_tokens <= self.target_tokens:
                current_words.append(word)
                current_tokens += word_tokens
            else:
                if current_words:
                    chunk_text = ' '.join(current_words)
                    chunks.append(self._create_chunk(chunk_text, page_no, [page_no]))
                
                current_words = [word]
                current_tokens = word_tokens
        
        if current_words:
            chunk_text = ' '.join(current_words)
            chunks.append(self._create_chunk(chunk_text, page_no, [page_no]))
        
        return chunks
    
    def _split_large_paragraph(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """큰 문단을 문장 단위로 분할"""
        import re
        
        # 문장 분리 (한국어/영어)
        sentences = re.split(r'(?<=[.!?。])\s+', text)
        
        chunks = []
        current_sents = []
        current_tokens = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_tokens = self._count_tokens(sent)
            
            if current_tokens + sent_tokens <= self.target_tokens:
                current_sents.append(sent)
                current_tokens += sent_tokens
            else:
                if current_sents:
                    chunk_text = ' '.join(current_sents)
                    chunks.append(self._create_chunk(chunk_text, page_no, [page_no]))
                
                current_sents = [sent]
                current_tokens = sent_tokens
        
        if current_sents:
            chunk_text = ' '.join(current_sents)
            chunks.append(self._create_chunk(chunk_text, page_no, [page_no]))
        
        return chunks
    
    def _find_chunk_end(self, text: str, start: int, target_tokens: int) -> int:
        """목표 토큰 수에 해당하는 종료 위치 찾기"""
        # 대략적인 추정으로 시작
        estimated_chars = int(target_tokens * 4)  # 한글 기준 대략 4자/토큰
        end = min(start + estimated_chars, len(text))
        
        # 실제 토큰 수 확인하며 조정
        chunk_text = text[start:end]
        actual_tokens = self._count_tokens(chunk_text)
        
        # 토큰 수가 목표보다 많으면 줄이기
        while actual_tokens > target_tokens and end > start + 10:
            end -= max(10, (actual_tokens - target_tokens) * 2)
            chunk_text = text[start:end]
            actual_tokens = self._count_tokens(chunk_text)
        
        # 토큰 수가 목표보다 적으면 늘리기
        while actual_tokens < target_tokens * 0.9 and end < len(text):
            end += max(10, (target_tokens - actual_tokens) * 2)
            chunk_text = text[start:end]
            actual_tokens = self._count_tokens(chunk_text)
            
            if actual_tokens > target_tokens:
                # 넘어가면 이전 위치로 복구
                end -= max(10, (actual_tokens - target_tokens) * 2)
                break
        
        return min(end, len(text))
    
    def _tokens_to_chars(self, text: str, pos: int, tokens: int, reverse: bool = False) -> int:
        """토큰 수에 해당하는 문자 수 추정"""
        # 간단한 추정
        estimated_chars = int(tokens * 4)
        
        if reverse:
            # 뒤로 이동
            start = max(0, pos - estimated_chars)
            chunk = text[start:pos]
        else:
            # 앞으로 이동
            end = min(len(text), pos + estimated_chars)
            chunk = text[pos:end]
        
        actual_tokens = self._count_tokens(chunk)
        
        # 토큰 수에 맞게 조정
        if actual_tokens > 0:
            char_per_token = len(chunk) / actual_tokens
            return int(tokens * char_per_token)
        
        return estimated_chars
    
    def _get_pages_for_range(self, char_to_page: List[int], start: int, end: int) -> List[int]:
        """문자 범위가 걸쳐 있는 페이지 목록"""
        if not char_to_page:
            return [1]
        
        start = max(0, min(start, len(char_to_page) - 1))
        end = max(0, min(end, len(char_to_page)))
        
        pages = set()
        for i in range(start, end):
            if i < len(char_to_page):
                pages.add(char_to_page[i])
        
        return sorted(list(pages)) if pages else [1]
    
    def _create_chunk(self, text: str, page_no: int, pages: List[int]) -> Tuple[str, Dict]:
        """청크 생성"""
        meta = {
            "type": "proofreading_chunk",
            "page": page_no,
            "pages": pages,
            "token_count": self._count_tokens(text),
            "char_count": len(text)
        }
        
        # META 라인 추가
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        final_text = meta_line + "\n" + text
        
        return (final_text, meta)
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        if not text:
            return 0
        try:
            return len(self.encoder(text))
        except:
            # 폴백: 단어 수 기반 추정
            return int(len(text.split()) * 1.3)


# ========== 외부 인터페이스 함수 ==========

def simple_chunk_by_page(
    pages: List[Tuple[int, str]], 
    encoder_fn: Callable,
    target_tokens: int = 400
) -> List[Tuple[str, Dict]]:
    """페이지별 청킹 (가장 단순)"""
    chunker = SimpleProofreadingChunker(encoder_fn, target_tokens)
    return chunker.chunk_by_page(pages)


def simple_chunk_by_tokens(
    pages: List[Tuple[int, str]], 
    encoder_fn: Callable,
    target_tokens: int = 400,
    overlap_tokens: int = 50
) -> List[Tuple[str, Dict]]:
    """고정 토큰 크기 청킹 (슬라이딩 윈도우)"""
    chunker = SimpleProofreadingChunker(encoder_fn, target_tokens, overlap_tokens)
    return chunker.chunk_by_fixed_tokens(pages)


def simple_chunk_by_paragraph(
    pages: List[Tuple[int, str]], 
    encoder_fn: Callable,
    target_tokens: int = 400
) -> List[Tuple[str, Dict]]:
    """문단 기준 청킹 (추천)"""
    chunker = SimpleProofreadingChunker(encoder_fn, target_tokens)
    return chunker.chunk_by_paragraph(pages)