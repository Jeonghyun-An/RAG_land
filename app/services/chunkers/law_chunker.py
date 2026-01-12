# app/services/chunkers/law_chunker.py
"""
원자력 법령/매뉴얼 전용 고도화 청킹 모듈 (v2.0 - 완전 개선)
- 문단간 유기성 보존 (크로스 페이지 포함)
- 조항/절/항 구조 완전 인식 (제목 포함)
- IAEA 규정, 원자력안전법 등 특화 처리
- 의미론적 연속성 보장
- 표/박스 통합 처리
"""
from __future__ import annotations
import re
import json
from typing import List, Tuple, Dict, Optional, Callable, Any


# ==================== 법령/규정 패턴 정의 ====================

LEGAL_PATTERNS = {
    # 조항 번호 패턴 (제목 포함)
    'article_enhanced': re.compile(
        r'제\s*(\d+)\s*조(?:\s*\(([가-힣\s]+)\)|\s+([가-힣\s]+))?', 
        re.IGNORECASE
    ),
    'article': re.compile(r'제\s*(\d+)\s*조(?:\s*[가-힣\s]*)?', re.IGNORECASE),
    'section': re.compile(r'제\s*(\d+)\s*절(?:\s*[가-힣\s]*)?', re.IGNORECASE),
    'paragraph': re.compile(r'제\s*(\d+)\s*항(?:\s*[가-힣\s]*)?', re.IGNORECASE),
    'clause': re.compile(r'제\s*(\d+)\s*호(?:\s*[가-힣\s]*)?', re.IGNORECASE),
    
    # 항 기호
    'paragraph_symbol': re.compile(r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]'),
    
    # IAEA 규정 패턴
    'infcirc': re.compile(r'INFCIRC[/\-](\d+)(?:\s*\([^)]*\))?', re.IGNORECASE),
    'iaea_requirement': re.compile(r'Requirement\s+(\d+)(?:\.(\d+))?:', re.IGNORECASE),
    'iaea_section': re.compile(r'(\d+)\.(\d+)(?:\.(\d+))?\s*[A-Za-z\s]*', re.IGNORECASE),
    
    # 기술 표준 패턴
    'technical_code': re.compile(r'[A-Z]{2,}-\d+(?:\.\d+)*', re.IGNORECASE),
    
    # 목록/항목 패턴
    'list_item': re.compile(r'^[\s]*(?:\([가-힣]\)|\d+\)|\([ivx]+\)|\d+\.)', re.MULTILINE),
    
    # 표/박스 구조 패턴
    'table_header': re.compile(r'\+[-=]+\+'),
    'box_structure': re.compile(r'[\+\|]{3,}'),
    
    # 각주 패턴
    'footnote': re.compile(r'\[\^(\d+)\]'),
}

# 원자력 분야 전문용어 그룹
NUCLEAR_KEYWORDS = {
    'safety': ['안전', '보안', '방호', '방사선', '오염', '누설', '사고'],
    'materials': ['핵물질', '우라늄', '플루토늄', '토륨', '핵연료', '방사성물질'],
    'facilities': ['원자로', '핵시설', '저장시설', '처리시설', '방사성폐기물'],
    'regulations': ['보장조치', '사찰', '신고', '허가', '인가', '승인'],
    'procedures': ['절차', '방법', '기준', '요건', '조건', '규정'],
}


# ==================== 메인 청킹 클래스 ====================

class NuclearLegalChunker:
    """원자력 법령/매뉴얼 전용 청킹 클래스 (v2.0)"""
    
    def __init__(self, encoder_fn: Callable, target_tokens: int = 400, overlap_tokens: int = 100):
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = 50
        self.max_chunk_tokens = target_tokens * 2
        
        # 표 감지는 enhanced_table_detector가 있으면 사용, 없으면 기본 패턴 매칭
        try:
            from app.services.enhanced_table_detector import EnhancedTableDetector
            self.table_detector = EnhancedTableDetector()
        except ImportError:
            self.table_detector = None
            print("[LAW-CHUNKER] enhanced_table_detector not available, using pattern matching")
        
        # 페이지별 최대 청크 크기 제한
        self.max_tokens_per_page = target_tokens * 3
        
    def chunk_pages(self, pages_std: List[Tuple[int, str]], 
                   layout_blocks: Optional[Dict[int, List[Dict]]] = None,
                   min_chunk_tokens: int = 100) -> List[Tuple[str, Dict]]:
        """페이지별 텍스트를 법령/매뉴얼 특화 청킹"""
        if not pages_std:
            return []
            
        self.min_chunk_tokens = min_chunk_tokens
        all_chunks = []
        
        for page_no, text in pages_std:
            if not text or not text.strip():
                continue
                
            # 페이지별 레이아웃 블록 정보
            page_blocks = layout_blocks.get(page_no, []) if layout_blocks else []
            
            # 문서 유형 자동 감지
            doc_type = self._detect_document_type(text)
            
            # 페이지별로 독립 청킹 수행
            page_chunks = self._chunk_single_page(text, page_no, page_blocks, doc_type)
            
            all_chunks.extend(page_chunks)
            
        # 크로스 페이지 연결성 처리
        connected_chunks = self._process_cross_page_continuity(all_chunks)
        
        # 최종 검증 및 정리
        final_chunks = self._validate_and_clean_chunks(connected_chunks)
        
        return final_chunks
    
    def _chunk_single_page(self, text: str, page_no: int, 
                          page_blocks: List[Dict], doc_type: str) -> List[Tuple[str, Dict]]:
        """단일 페이지를 독립적으로 청킹"""
        # 문서 유형에 따른 청킹
        if doc_type == 'iaea_guide':
            page_chunks = self._chunk_iaea_guide(text, page_no, page_blocks)
        elif doc_type == 'korean_law':
            page_chunks = self._chunk_korean_law(text, page_no, page_blocks)
        elif doc_type == 'technical_manual':
            page_chunks = self._chunk_technical_manual(text, page_no, page_blocks)
        else:
            page_chunks = self._chunk_structured_text(text, page_no, page_blocks)
        
        # 페이지별 토큰 제한 확인 및 강제 분할
        page_chunks = self._enforce_page_token_limit(page_chunks, page_no)
        
        return page_chunks
    
    def _enforce_page_token_limit(self, chunks: List[Tuple[str, Dict]], 
                                  page_no: int) -> List[Tuple[str, Dict]]:
        """페이지별 토큰 제한 강제 - 한 페이지 내용이 모두 하나의 청크로 들어가는 것 방지"""
        result = []
        
        for chunk_text, chunk_meta in chunks:
            tokens = self._count_tokens(chunk_text)
            
            if tokens <= self.max_tokens_per_page:
                result.append((chunk_text, chunk_meta))
            else:
                # 강제 분할 필요
                print(f"[LAW-CHUNKER] Page {page_no} chunk too large ({tokens} tokens), forcing split")
                split_chunks = self._force_split_large_chunk(chunk_text, chunk_meta)
                result.extend(split_chunks)
        
        return result
    
    def _force_split_large_chunk(self, text: str, meta: Dict) -> List[Tuple[str, Dict]]:
        """너무 큰 청크를 강제로 분할"""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= self.target_tokens:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
            else:
                if current_chunk.strip():
                    new_meta = meta.copy()
                    new_meta['token_count'] = current_tokens
                    chunks.append((current_chunk.strip(), new_meta))
                
                current_chunk = sentence
                current_tokens = sentence_tokens
        
        if current_chunk.strip():
            new_meta = meta.copy()
            new_meta['token_count'] = current_tokens
            chunks.append((current_chunk.strip(), new_meta))
        
        return chunks
    
    # ==================== 문서 유형별 청킹 ====================
    
    def _detect_document_type(self, text: str) -> str:
        """문서 유형 자동 감지"""
        text_lower = text.lower()
        
        # IAEA 문서 판별
        if any(pattern in text_lower for pattern in ['iaea', 'infcirc', 'safeguards', 'requirement']):
            return 'iaea_guide'
            
        # 한국 법령 판별    
        if re.search(r'제\s*\d+\s*조', text) and any(word in text for word in ['법', '시행령', '시행규칙']):
            return 'korean_law'
            
        # 기술 매뉴얼 판별
        if any(pattern in text_lower for pattern in ['manual', 'procedure', 'standard', '매뉴얼', '절차서']):
            return 'technical_manual'
            
        return 'general'
    
    def _chunk_iaea_guide(self, text: str, page_no: int, blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """IAEA 가이드라인 특화 청킹"""
        detected_tables = []
        if self.table_detector:
            detected_tables = self.table_detector.detect_tables(text, page_no, blocks)
        
        if detected_tables:
            return self._chunk_with_tables_iaea(text, page_no, detected_tables, blocks)
   
        chunks = []
        
        # Requirement 단위로 분할 (최우선)
        req_pattern = LEGAL_PATTERNS['iaea_requirement']
        requirements = list(req_pattern.finditer(text))
        
        if requirements:
            # Requirement가 있으면 Requirement 단위로 청킹
            for i, req_match in enumerate(requirements):
                req_number = req_match.group(1)
                req_start = req_match.start()
                
                # Requirement 끝: 다음 Requirement 시작 or 텍스트 끝
                if i < len(requirements) - 1:
                    req_end = requirements[i + 1].start()
                else:
                    req_end = len(text)
                
                req_text = text[req_start:req_end].strip()
                
                # Requirement는 완전히 보존
                chunks.append(self._create_chunk(req_text, page_no, f"Requirement {req_number}"))
        else:
            # Requirement 없으면 섹션 단위로 분할
            sections = self._split_by_iaea_sections(text)
            
            for section_info in sections:
                section_text = section_info['text']
                section_id = section_info['id']
                
                # 박스/표 구조 보존
                if self._has_structured_content(section_text):
                    chunks.extend(self._preserve_structured_content(section_text, page_no, section_id))
                else:
                    chunks.extend(self._semantic_chunking(section_text, page_no, section_id))
                
        return chunks
    
    def _chunk_korean_law(self, text: str, page_no: int, blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """한국 법령 특화 청킹"""
        detected_tables = []
        if self.table_detector:
            detected_tables = self.table_detector.detect_tables(text, page_no, blocks)
        
        if detected_tables:
            return self._chunk_with_tables_law(text, page_no, detected_tables, blocks)
        
        chunks = []
        
        # 조항별 분할 (개선 버전 - 제목 포함)
        articles = self._split_by_articles(text)
        
        for article in articles:
            article_text = article['text']
            article_num = article['number']
            article_title = article.get('title', '')
            
            section_id = f"제{article_num}조"
            if article_title:
                section_id += f"({article_title})"
            
            # 조항이 길면 항/호로 세분화
            if self._count_tokens(article_text) > self.target_tokens:
                sub_chunks = self._split_article_by_paragraphs(article_text, page_no, section_id)
                chunks.extend(sub_chunks)
            else:
                # 조항 전체를 하나의 청크로
                chunks.append(self._create_chunk(article_text, page_no, section_id))
                
        return chunks
    
    def _chunk_technical_manual(self, text: str, page_no: int, blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """기술 매뉴얼 특화 청킹"""
        detected_tables = []
        if self.table_detector:
            detected_tables = self.table_detector.detect_tables(text, page_no, blocks)
    
        if detected_tables:
            return self._chunk_with_tables_manual(text, page_no, detected_tables, blocks)
    
        chunks = []
        
        # 절차 단계별 분할
        procedures = self._split_by_procedures(text)
        
        for proc in procedures:
            proc_text = proc['text']
            proc_id = proc['id']
            
            # 절차가 복잡하면 세부 단계로 분할
            if self._is_complex_procedure(proc_text):
                sub_chunks = self._split_complex_procedure(proc_text, page_no, proc_id)
                chunks.extend(sub_chunks)
            else:
                chunks.append(self._create_chunk(proc_text, page_no, proc_id))
                
        return chunks
    
    def _chunk_structured_text(self, text: str, page_no: int, blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """일반 구조화 텍스트 청킹"""
        # 문단별 분할
        paragraphs = self._split_by_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            if current_tokens + para_tokens <= self.target_tokens:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens
            else:
                if current_chunk.strip():
                    chunks.append(self._create_chunk(current_chunk, page_no))
                
                current_chunk = para
                current_tokens = para_tokens
        
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk, page_no))
        
        return chunks
    
    # ==================== 구조 분할 헬퍼 ====================
    
    def _split_by_articles(self, text: str) -> List[Dict]:
        """
        텍스트를 조항별로 분할 (개선 버전)
        - 조항 제목까지 포함하여 완전한 단위로 분할
        - 패턴: "제 N 조 (조항명)" 형태 지원
        """
        articles = []
        
        # 조항 패턴 (제목 포함): 제 N 조 (선택적 조항명)
        # 예: "제1조(목적)", "제2조 목적" 등
        article_pattern_enhanced = re.compile(
            r'(제\s*(\d+)\s*조(?:\s*\(([가-힣\s]+)\)|\s+([가-힣\s]+))?)', 
            re.IGNORECASE
        )
        
        # 조항 위치 찾기
        matches = list(article_pattern_enhanced.finditer(text))
        
        if not matches:
            # 조항이 없으면 전체를 하나로
            if text.strip():
                articles.append({"number": "전문", "text": text.strip(), "title": ""})
            return articles
        
        # 첫 번째 조항 이전 (서문/헤더)
        if matches[0].start() > 0:
            header_text = text[:matches[0].start()].strip()
            if header_text:
                articles.append({"number": "서문", "text": header_text, "title": ""})
        
        # 조항별 분할
        for i, match in enumerate(matches):
            full_article_header = match.group(1)  # "제1조(목적)" 전체
            article_number = match.group(2)       # "1"
            
            # 조항 제목 추출 (괄호 안 or 공백 뒤)
            article_title = match.group(3) or match.group(4) or ""
            article_title = article_title.strip() if article_title else ""
            
            # 조항 내용 시작
            start_pos = match.end()
            
            # 조항 내용 끝 (다음 조항 시작 or 텍스트 끝)
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            article_content = text[start_pos:end_pos].strip()
            
            # 조항 전체 = 헤더 + 내용
            full_article_text = full_article_header + "\n" + article_content
            
            articles.append({
                "number": article_number,
                "title": article_title,
                "text": full_article_text.strip()
            })
        
        return articles
    
    def _split_by_iaea_sections(self, text: str) -> List[Dict]:
        """IAEA 섹션 번호로 분할 (예: 1.1, 2.3.4)"""
        sections = []
        
        section_pattern = LEGAL_PATTERNS['iaea_section']
        matches = list(section_pattern.finditer(text))
        
        if not matches:
            # 섹션이 없으면 전체를 하나로
            if text.strip():
                sections.append({"id": "intro", "text": text.strip()})
            return sections
        
        # 첫 섹션 이전
        if matches[0].start() > 0:
            intro = text[:matches[0].start()].strip()
            if intro:
                sections.append({"id": "intro", "text": intro})
        
        # 섹션별 분할
        for i, match in enumerate(matches):
            section_id = match.group(0).strip()
            start = match.start()
            
            if i < len(matches) - 1:
                end = matches[i + 1].start()
            else:
                end = len(text)
            
            section_text = text[start:end].strip()
            sections.append({"id": section_id, "text": section_text})
        
        return sections
    
    def _split_by_procedures(self, text: str) -> List[Dict]:
        """절차 단계별 분할"""
        procedures = []
        
        step_pattern = re.compile(
            r'(Step\s+\d+|단계\s*\d+|Step\s*\d+|\d+\))', 
            re.MULTILINE | re.IGNORECASE
        )
        
        parts = step_pattern.split(text)
        matches = step_pattern.findall(text)
        
        # 첫 번째 부분 (헤더)
        if parts and parts[0].strip():
            procedures.append({"id": "header", "text": parts[0].strip()})
        
        # 나머지 부분들
        for i in range(1, len(parts)):
            if parts[i].strip():
                proc_id = matches[i-1] if i-1 < len(matches) else f"step_{i}"
                procedures.append({"id": proc_id, "text": parts[i].strip()})
                
        return procedures
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """문단별 분할 (빈 줄 기준)"""
        paragraphs = []
        current_para = ""
        
        lines = text.split('\n')
        
        for line in lines:
            if line.strip():
                if current_para:
                    current_para += "\n" + line
                else:
                    current_para = line
            else:
                if current_para.strip():
                    paragraphs.append(current_para.strip())
                    current_para = ""
        
        # 마지막 문단 처리
        if current_para.strip():
            paragraphs.append(current_para.strip())
            
        return paragraphs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장으로 분할"""
        # 한국어/영어 혼재 문장 분리
        sentence_endings = re.compile(r'(?<=[.!?다요]\s)|\n+')
        sentences = sentence_endings.split(text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_article_by_paragraphs(self, article_text: str, page_no: int, section_id: str) -> List[Tuple[str, Dict]]:
        """조항을 항/호로 세분화"""
        chunks = []
        
        # 항 패턴 분할 (①②③ 기호 사용)
        paragraph_pattern = LEGAL_PATTERNS['paragraph_symbol']
        
        if paragraph_pattern.search(article_text):
            # 항 기호로 분할
            parts = paragraph_pattern.split(article_text)
            matches = paragraph_pattern.findall(article_text)
            
            current_text = parts[0].strip() if parts else ""
            
            for i, symbol in enumerate(matches):
                if i+1 < len(parts):
                    current_text += symbol + parts[i+1]
                    
                    # 항 하나가 완성되면 청크로
                    if self._count_tokens(current_text) >= self.target_tokens or i == len(matches) - 1:
                        if current_text.strip():
                            chunks.append(self._create_chunk(current_text, page_no, section_id))
                        current_text = ""
            
            if current_text.strip():
                chunks.append(self._create_chunk(current_text, page_no, section_id))
        else:
            # 항 기호 없으면 문장 단위로
            chunks = self._semantic_chunking(article_text, page_no, section_id)
        
        return chunks
    
    def _is_complex_procedure(self, text: str) -> bool:
        """절차가 복잡한지 판단"""
        sub_steps = len(re.findall(r'(?:하위|세부)\s*단계', text, re.I))
        tokens = self._count_tokens(text)
        
        return sub_steps > 3 or tokens > self.target_tokens * 1.5
    
    def _split_complex_procedure(self, text: str, page_no: int, proc_id: str) -> List[Tuple[str, Dict]]:
        """복잡한 절차를 세부 단계로 분할"""
        return self._semantic_chunking(text, page_no, proc_id)
    
    def _has_structured_content(self, text: str) -> bool:
        """구조화된 내용(표, 박스) 포함 여부 확인"""
        return any(pattern.search(text) for pattern in [
            LEGAL_PATTERNS['table_header'],
            LEGAL_PATTERNS['box_structure']
        ])
    
    def _preserve_structured_content(self, text: str, page_no: int, section_id: str) -> List[Tuple[str, Dict]]:
        """구조화된 내용을 보존하면서 청킹"""
        chunks = []
        
        # 표/박스와 일반 텍스트 분리
        parts = re.split(r'(\+[-=]+\+|[\+\|]{3,})', text)
        
        current_structure = ""
        is_in_structure = False
        
        for part in parts:
            if LEGAL_PATTERNS['table_header'].match(part) or LEGAL_PATTERNS['box_structure'].match(part):
                is_in_structure = True
                current_structure += part
            elif is_in_structure:
                current_structure += part
                if not any(p in part for p in ['+', '|', '─', '═']):
                    # 구조 종료
                    if current_structure.strip():
                        chunks.append(self._create_chunk(current_structure, page_no, section_id + "_table"))
                    current_structure = ""
                    is_in_structure = False
            else:
                # 일반 텍스트
                if part.strip():
                    chunks.extend(self._semantic_chunking(part, page_no, section_id))
        
        # 마지막 구조
        if current_structure.strip():
            chunks.append(self._create_chunk(current_structure, page_no, section_id + "_table"))
        
        return chunks
    
    def _semantic_chunking(self, text: str, page_no: int, section: str = "") -> List[Tuple[str, Dict]]:
        """의미론적 청킹 (문장 단위)"""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= self.target_tokens:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
            else:
                if current_chunk.strip():
                    chunks.append(self._create_chunk(current_chunk, page_no, section))
                current_chunk = sentence
                current_tokens = sentence_tokens
        
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk, page_no, section))
        
        return chunks
    
    # ==================== 표 처리 ====================
    
    def _chunk_with_tables_iaea(self, text: str, page_no: int, tables: List, blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """IAEA 문서에서 표와 텍스트를 함께 청킹"""
        # 표가 있으면 표를 독립 청크로, 나머지는 일반 청킹
        chunks = []
        
        for table in tables:
            table_text = table.get('content', '')
            if table_text.strip():
                chunks.append(self._create_chunk(table_text, page_no, "table"))
        
        # 나머지 텍스트 청킹
        text_only = text
        for table in tables:
            text_only = text_only.replace(table.get('content', ''), '')
        
        if text_only.strip():
            text_chunks = self._chunk_iaea_guide(text_only, page_no, [])
            chunks.extend(text_chunks)
        
        return chunks
    
    def _chunk_with_tables_law(self, text: str, page_no: int, tables: List, blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """한국 법령에서 표와 텍스트를 함께 청킹"""
        return self._chunk_with_tables_iaea(text, page_no, tables, blocks)  # 동일 처리
    
    def _chunk_with_tables_manual(self, text: str, page_no: int, tables: List, blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """기술 매뉴얼에서 표와 텍스트를 함께 청킹"""
        return self._chunk_with_tables_iaea(text, page_no, tables, blocks)  # 동일 처리
    
    # ==================== 크로스 페이지 연결성 ====================
    
    def _process_cross_page_continuity(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """
        페이지 경계를 넘어가는 조항/섹션 연결 처리
        - 페이지 끝에서 문장이 끊긴 경우 다음 페이지 시작과 병합
        """
        if len(chunks) < 2:
            return chunks
        
        connected = []
        i = 0
        
        while i < len(chunks):
            current_text, current_meta = chunks[i]
            
            # 다음 청크가 있고, 페이지가 연속되며, 현재 청크가 미완성 문장으로 끝나는 경우
            if i + 1 < len(chunks):
                next_text, next_meta = chunks[i + 1]
                current_page = current_meta.get('page', 0)
                next_page = next_meta.get('page', 0)
                
                # 페이지가 연속되고, 현재 청크가 미완성으로 끝나는지 확인
                if next_page == current_page + 1 and self._is_incomplete_sentence(current_text):
                    # 병합 가능한지 확인 (토큰 수 제한)
                    combined_text = current_text + " " + next_text
                    combined_tokens = self._count_tokens(combined_text)
                    
                    if combined_tokens <= self.max_chunk_tokens:
                        # 병합
                        merged_meta = current_meta.copy()
                        merged_meta['pages'] = [current_page, next_page]
                        merged_meta['token_count'] = combined_tokens
                        merged_meta['cross_page'] = True
                        
                        connected.append((combined_text, merged_meta))
                        i += 2  # 두 청크를 병합했으므로 2 증가
                        continue
            
            # 병합하지 않는 경우
            connected.append((current_text, current_meta))
            i += 1
        
        return connected
    
    def _is_incomplete_sentence(self, text: str) -> bool:
        """문장이 미완성으로 끝나는지 확인"""
        text = text.strip()
        
        # 완전한 문장 종료 패턴
        complete_endings = ['.', '!', '?', '다.', '요.', '함.', '음.']
        
        # 미완성 가능성이 높은 패턴
        incomplete_endings = [',', ';', ':', '및', '그리고', '또는', 'and', 'or', 'but']
        
        # 완전한 종료로 끝나면 False
        if any(text.endswith(ending) for ending in complete_endings):
            return False
        
        # 미완성 패턴으로 끝나면 True
        if any(text.endswith(ending) for ending in incomplete_endings):
            return True
        
        # 마지막 단어가 조사/접속사로 끝나면 True
        last_word = text.split()[-1] if text.split() else ""
        if last_word in ['은', '는', '이', '가', '을', '를', '에', '의', '와', '과']:
            return True
        
        return False
    
    # ==================== 검증 및 정리 ====================
    
    def _validate_and_clean_chunks(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """최종 청크 검증 및 정리"""
        cleaned = []
        
        for text, meta in chunks:
            # 빈 청크 제거
            if not text.strip():
                continue
            
            # 최소 토큰 수 확인
            token_count = self._count_tokens(text)
            if token_count < self.min_chunk_tokens:
                # 너무 짧은 청크는 건너뛰거나 이전 청크와 병합
                if cleaned and self._count_tokens(cleaned[-1][0]) + token_count <= self.max_chunk_tokens:
                    # 이전 청크와 병합 가능
                    prev_text, prev_meta = cleaned[-1]
                    combined_text = prev_text + "\n\n" + text
                    prev_meta['token_count'] = self._count_tokens(combined_text)
                    cleaned[-1] = (combined_text, prev_meta)
                continue
            
            # 메타데이터 정규화
            cleaned_meta = {
                'page': meta.get('page', 1),
                'pages': meta.get('pages', [meta.get('page', 1)]),
                'section': meta.get('section', ''),
                'token_count': token_count,
                'type': meta.get('type', 'text'),
                'cross_page': meta.get('cross_page', False)
            }
            
            cleaned.append((text, cleaned_meta))
        
        return cleaned
    
    def _create_chunk(self, text: str, page_no: int, section: str = "") -> Tuple[str, Dict]:
        """청크 및 메타데이터 생성"""
        metadata = {
            'page': page_no,
            'pages': [page_no],
            'section': section,
            'token_count': self._count_tokens(text),
            'type': 'table' if 'table' in section.lower() else 'text',
            'cross_page': False
        }
        return (text.strip(), metadata)
    
    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        try:
            tokens = self.encoder(text)
            return len(tokens) if tokens else len(text.split())
        except:
            return len(text.split())

# ==================== 외부 인터페이스 함수 ====================

def law_chunk_pages(pages_std: List[Tuple[int, str]], 
                   encoder_fn: Callable,
                   target_tokens: int = 400,
                   overlap_tokens: int = 100,
                   layout_blocks: Optional[Dict[int, List[Dict]]] = None,
                   min_chunk_tokens: int = 100) -> List[Tuple[str, Dict]]:
    """
    법령/매뉴얼 전용 청킹 함수 (기존 인터페이스 호환)
    
    Args:
        pages_std: [(page_no, text), ...] 형태의 페이지 데이터
        encoder_fn: 토큰 인코딩 함수
        target_tokens: 목표 토큰 수
        overlap_tokens: 오버랩 토큰 수
        layout_blocks: 레이아웃 블록 정보
        min_chunk_tokens: 최소 청크 토큰 수
    
    Returns:
        [(chunk_text, metadata), ...] 형태의 청크 리스트
    """
    
    if not pages_std:
        return []
    
    # 법령/매뉴얼 내용인지 확인
    full_text = " ".join(text for _, text in pages_std)
    
    # 원자력/법령 관련 키워드 확인
    legal_indicators = ['조', '항', '호', '법', 'IAEA', 'INFCIRC', 'Requirement', '보장조치', '원자력']
    if not any(indicator in full_text for indicator in legal_indicators):
        # 법령/매뉴얼이 아니면 빈 결과 반환 (다른 청커로 폴백)
        return []
    
    chunker = NuclearLegalChunker(encoder_fn, target_tokens, overlap_tokens)
    return chunker.chunk_pages(pages_std, layout_blocks, min_chunk_tokens)