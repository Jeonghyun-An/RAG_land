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
    
    def _fast_paragraph_chunking(
        self,
        pages_std: List[Tuple[int, str]]
    ) -> List[Tuple[str, Dict]]:
        """초고속 문단 기반 청킹"""
        chunks = []
        
        for page_no, text in pages_std:
            if not text or not text.strip():
                continue
            
            # 문단 분리 (빈 줄 기준)
            paragraphs = self._split_paragraphs(text)
            
            current_chunk = ""
            current_tokens = 0
            
            for para in paragraphs:
                # ⭐ 빠른 토큰 추정 (인코딩 안 함!)
                para_tokens = len(para.split()) * 1.3
                
                if para_tokens > self.max_chunk_tokens:
                    # 현재 청크 저장
                    if current_chunk:
                        chunks.append(self._create_chunk(current_chunk, page_no))
                        current_chunk = ""
                        current_tokens = 0
                    
                    # 큰 문단은 문장으로 분할
                    sent_chunks = self._split_large_paragraph(para, page_no)
                    chunks.extend(sent_chunks)
                    
                elif current_tokens + para_tokens <= self.target_tokens:
                    # 현재 청크에 추가
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                    current_tokens += para_tokens
                    
                else:
                    # 현재 청크 완료
                    if current_chunk:
                        chunks.append(self._create_chunk(current_chunk, page_no))
                    
                    current_chunk = para
                    current_tokens = para_tokens
            
            # 마지막 청크
            if current_chunk:
                chunks.append(self._create_chunk(current_chunk, page_no))
        
        return self._finalize_chunks(chunks)
    
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