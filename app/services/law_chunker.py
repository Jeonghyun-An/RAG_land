# app/services/law_chunker.py
"""
원자력 법령/매뉴얼 전용 고도화 청킹 모듈
- 문단간 유기성 보존
- 조항/절/항 구조 인식
- IAEA 규정, 원자력안전법 등 특화 처리
- 의미론적 연속성 보장
- [수정] 페이지별 분할 강화, 조항 경계 정확도 개선
"""
from __future__ import annotations
import re
import json
from typing import List, Tuple, Dict, Optional, Callable, Any
from app.services.enhanced_table_detector import EnhancedTableDetector, TableRegion

# 법령/규정 패턴 정의
LEGAL_PATTERNS = {
    # 조항 번호 패턴
    'article': re.compile(r'제\s*(\d+)\s*조(?:\s*[가-힣\s]*)?', re.IGNORECASE),
    'section': re.compile(r'제\s*(\d+)\s*절(?:\s*[가-힣\s]*)?', re.IGNORECASE),
    'paragraph': re.compile(r'제\s*(\d+)\s*항(?:\s*[가-힣\s]*)?', re.IGNORECASE),
    'clause': re.compile(r'제\s*(\d+)\s*호(?:\s*[가-힣\s]*)?', re.IGNORECASE),
    
    # IAEA 규정 패턴
    'infcirc': re.compile(r'INFCIRC[/\-](\d+)(?:\s*\([^)]*\))?', re.IGNORECASE),
    'iaea_section': re.compile(r'(\d+)\.(\d+)(?:\.(\d+))?\s*[가-힣\s]*', re.IGNORECASE),
    
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

class NuclearLegalChunker:
    """원자력 법령/매뉴얼 전용 청킹 클래스"""
    
    def __init__(self, encoder_fn: Callable, target_tokens: int = 400, overlap_tokens: int = 100):
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = 50
        self.max_chunk_tokens = target_tokens * 2
        self.table_detector = EnhancedTableDetector()
        
        # [신규] 페이지별 최대 청크 크기 제한
        self.max_tokens_per_page = target_tokens * 3  # 한 페이지에서 최대 3청크 정도
        
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
            
            # [수정] 페이지별로 독립 청킹 수행
            page_chunks = self._chunk_single_page(text, page_no, page_blocks, doc_type)
            
            all_chunks.extend(page_chunks)
            
        # 크로스 페이지 연결성 처리 (조심스럽게)
        connected_chunks = self._process_cross_page_continuity(all_chunks)
        
        # 최종 검증 및 정리
        final_chunks = self._validate_and_clean_chunks(connected_chunks)
        
        return final_chunks
    
    def _chunk_single_page(self, text: str, page_no: int, 
                          page_blocks: List[Dict], doc_type: str) -> List[Tuple[str, Dict]]:
        """
        [신규] 단일 페이지를 독립적으로 청킹
        - 페이지 내용이 너무 크면 강제 분할
        """
        # 문서 유형에 따른 청킹
        if doc_type == 'iaea_guide':
            page_chunks = self._chunk_iaea_guide(text, page_no, page_blocks)
        elif doc_type == 'korean_law':
            page_chunks = self._chunk_korean_law(text, page_no, page_blocks)
        elif doc_type == 'technical_manual':
            page_chunks = self._chunk_technical_manual(text, page_no, page_blocks)
        else:
            page_chunks = self._chunk_structured_text(text, page_no, page_blocks)
        
        # [수정] 페이지별 토큰 제한 확인 및 강제 분할
        page_chunks = self._enforce_page_token_limit(page_chunks, page_no)
        
        return page_chunks
    
    def _enforce_page_token_limit(self, chunks: List[Tuple[str, Dict]], 
                                  page_no: int) -> List[Tuple[str, Dict]]:
        """
        [신규] 페이지별 토큰 제한 강제
        - 한 페이지 내용이 모두 하나의 청크로 들어가는 것 방지
        """
        result = []
        
        for chunk_text, chunk_meta in chunks:
            tokens = self._count_tokens(chunk_text)
            
            # 토큰이 너무 많으면 강제 분할
            if tokens > self.max_tokens_per_page:
                # 문장 단위로 재분할
                sub_chunks = self._force_split_by_sentences(
                    chunk_text, page_no, chunk_meta.get('section', '')
                )
                result.extend(sub_chunks)
            else:
                result.append((chunk_text, chunk_meta))
        
        return result
    
    def _force_split_by_sentences(self, text: str, page_no: int, 
                                  section: str) -> List[Tuple[str, Dict]]:
        """
        [신규] 문장 단위로 강제 분할 (토큰 제한 준수)
        """
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= self.target_tokens:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
            else:
                # 현재 청크 저장
                if current_chunk.strip():
                    chunks.append(self._create_chunk(current_chunk, page_no, section))
                
                # 새 청크 시작
                current_chunk = sentence
                current_tokens = sentence_tokens
        
        # 마지막 청크
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk, page_no, section))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        문장 단위 분할 (한국어/영어 대응)
        """
        # 한국어 문장 종결 + 영어 문장 종결
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[가-힣A-Z])|(?<=[다요]\s)\s*(?=\n|[가-힣])')
        sentences = sentence_endings.split(text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _detect_document_type(self, text: str) -> str:
        """문서 유형 자동 감지"""
        text_lower = text.lower()
        
        # IAEA 문서 판별
        if any(pattern in text_lower for pattern in ['iaea', 'infcirc', 'safeguards', '보장조치']):
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
        detected_tables = self.table_detector.detect_tables(text, page_no, blocks)
        if detected_tables:
            # 표가 있으면 표 중심 청킹
            return self._chunk_with_tables_iaea(text, page_no, detected_tables, blocks)
   
        chunks = []
        
        # 섹션 단위로 분할
        sections = self._split_by_iaea_sections(text)
        
        for section_info in sections:
            section_text = section_info['text']
            section_id = section_info['id']
            
            # 박스/표 구조 보존
            if self._has_structured_content(section_text):
                # 구조화된 내용은 통째로 보존
                chunks.extend(self._preserve_structured_content(section_text, page_no, section_id))
            else:
                # 일반 텍스트는 의미 단위로 청킹
                chunks.extend(self._semantic_chunking(section_text, page_no, section_id))
                
        return chunks
    
    def _chunk_korean_law(self, text: str, page_no: int, blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """한국 법령 특화 청킹"""
        detected_tables = self.table_detector.detect_tables(text, page_no, blocks)
        
        if detected_tables:
            return self._chunk_with_tables_law(text, page_no, detected_tables, blocks)
        
        chunks = []
        
        # 조항별 분할
        articles = self._split_by_articles(text)
        
        for article in articles:
            article_text = article['text']
            article_num = article['number']
            
            # 조항이 길면 항/호로 세분화
            if self._count_tokens(article_text) > self.target_tokens:
                sub_chunks = self._split_article_by_paragraphs(article_text, page_no, article_num)
                chunks.extend(sub_chunks)
            else:
                # 조항 전체를 하나의 청크로
                chunks.append(self._create_chunk(article_text, page_no, f"제{article_num}조"))
                
        return chunks
    
    def _chunk_technical_manual(self, text: str, page_no: int, blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """기술 매뉴얼 특화 청킹"""
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
        """일반 구조적 텍스트 청킹"""
        chunks = []
        
        # 문단 단위 분할
        paragraphs = self._split_by_paragraphs(text)
        
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            # 현재 청크에 추가할 수 있는지 확인
            if current_tokens + para_tokens <= self.target_tokens:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens
            else:
                # 현재 청크 완료
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, page_no))
                
                # 새 청크 시작
                if para_tokens > self.max_chunk_tokens:
                    # 너무 긴 문단은 문장 단위로 재분할
                    sub_chunks = self._split_long_paragraph(para, page_no)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = para
                    current_tokens = para_tokens
        
        # 마지막 청크 처리
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, page_no))
            
        return chunks
    
    def _split_by_iaea_sections(self, text: str) -> List[Dict]:
        """
        IAEA 섹션별 분할
        [수정] 새 섹션 감지 시 이전 섹션을 먼저 저장
        """
        sections = []
        current_section = {"id": "", "text": ""}
        
        lines = text.split('\n')
        
        for line in lines:
            # 섹션 헤더 검출
            section_match = LEGAL_PATTERNS['iaea_section'].match(line.strip())
            if section_match:
                # [수정] 이전 섹션 저장 (새 라인 추가 전에)
                if current_section["text"].strip():
                    sections.append(current_section)
                
                # 새 섹션 시작
                current_section = {
                    "id": section_match.group(0),
                    "text": line  # 새 섹션 헤더만 포함
                }
            else:
                # 현재 섹션에 라인 추가
                if current_section["text"]:
                    current_section["text"] += "\n" + line
                else:
                    current_section["text"] = line
        
        # 마지막 섹션 처리
        if current_section["text"].strip():
            sections.append(current_section)
            
        return sections
    
    def _split_by_articles(self, text: str) -> List[Dict]:
        """
        조항별 분할
        [수정] 새 조항 감지 시 이전 조항을 먼저 저장
        """
        articles = []
        current_article = {"number": "", "text": ""}
        
        lines = text.split('\n')
        
        for line in lines:
            article_match = LEGAL_PATTERNS['article'].search(line)
            if article_match:
                # [수정] 이전 조항 저장 (새 라인 추가 전에)
                if current_article["text"].strip():
                    articles.append(current_article)
                
                # 새 조항 시작
                current_article = {
                    "number": article_match.group(1),
                    "text": line  # 새 조항 헤더만 포함
                }
            else:
                # 현재 조항에 라인 추가
                if current_article["text"]:
                    current_article["text"] += "\n" + line
                else:
                    current_article["text"] = line
        
        # 마지막 조항 처리
        if current_article["text"].strip():
            articles.append(current_article)
            
        return articles
    
    def _split_by_procedures(self, text: str) -> List[Dict]:
        """
        절차별 분할
        [수정] 새 절차 감지 시 이전 절차를 먼저 저장
        """
        procedures = []
        
        # 절차 패턴: "단계 N", "Step N", "N)" 등
        step_pattern = re.compile(r'^(?:단계\s*\d+|Step\s*\d+|\d+\))', re.MULTILINE | re.IGNORECASE)
        
        # 텍스트를 절차 패턴으로 분할
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
    
    def _split_article_by_paragraphs(self, article_text: str, page_no: int, 
                                    article_num: str) -> List[Tuple[str, Dict]]:
        """조항을 항/호로 세분화"""
        chunks = []
        
        # 항 패턴 분할
        paragraph_pattern = re.compile(r'(①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|\d+\.)')
        parts = paragraph_pattern.split(article_text)
        
        current_text = ""
        for i, part in enumerate(parts):
            if paragraph_pattern.match(part):
                if current_text.strip():
                    chunks.append(self._create_chunk(
                        current_text, page_no, f"제{article_num}조"
                    ))
                current_text = part
            else:
                current_text += part
        
        if current_text.strip():
            chunks.append(self._create_chunk(current_text, page_no, f"제{article_num}조"))
        
        return chunks
    
    def _split_long_paragraph(self, para: str, page_no: int) -> List[Tuple[str, Dict]]:
        """긴 문단을 문장 단위로 분할"""
        return self._semantic_chunking(para, page_no)
    
    def _is_complex_procedure(self, text: str) -> bool:
        """절차가 복잡한지 판단"""
        # 하위 단계가 많거나 토큰이 많으면 복잡
        sub_steps = len(re.findall(r'(?:^|\n)\s*[가-하]\)', text))
        return sub_steps > 3 or self._count_tokens(text) > self.target_tokens
    
    def _split_complex_procedure(self, text: str, page_no: int, 
                                proc_id: str) -> List[Tuple[str, Dict]]:
        """복잡한 절차를 세부 단계로 분할"""
        return self._semantic_chunking(text, page_no, proc_id)
    
    def _chunk_with_tables_iaea(self, text: str, page_no: int, 
                                tables: List[TableRegion], blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """IAEA 문서의 표 포함 청킹"""
        chunks = []
        lines = text.split('\n')
        table_regions = sorted(tables, key=lambda t: t.start_line)
        current_line = 0

        for table in table_regions:
            # 표 이전 텍스트 - 섹션 단위로 처리
            if current_line < table.start_line:
                before_text = '\n'.join(lines[current_line:table.start_line])
                if before_text.strip():
                    sections = self._split_by_iaea_sections(before_text)
                    for sec in sections:
                        chunks.extend(self._semantic_chunking(sec['text'], page_no, sec['id']))

            # 표 처리
            if table.content.strip():
                table_tokens = self._count_tokens(table.content)
                if table_tokens <= self.target_tokens:
                    chunks.append(self._create_chunk(table.content, page_no, "IAEA Table"))
                else:
                    # 큰 표는 행 단위 분할
                    chunks.extend(self._split_large_table(table.content, page_no, "IAEA Table"))

            current_line = table.end_line + 1

        # 마지막 텍스트
        if current_line < len(lines):
            after_text = '\n'.join(lines[current_line:])
            if after_text.strip():
                sections = self._split_by_iaea_sections(after_text)
                for sec in sections:
                    chunks.extend(self._semantic_chunking(sec['text'], page_no, sec['id']))

        return chunks

    def _chunk_with_tables_law(self, text: str, page_no: int, 
                               tables: List[TableRegion], blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """법령 문서의 표 포함 청킹"""
        chunks = []
        lines = text.split('\n')
        table_regions = sorted(tables, key=lambda t: t.start_line)
        current_line = 0

        for table in table_regions:
            # 표 이전 텍스트 - 조항 단위로 처리
            if current_line < table.start_line:
                before_text = '\n'.join(lines[current_line:table.start_line])
                if before_text.strip():
                    articles = self._split_by_articles(before_text)
                    for article in articles:
                        if self._count_tokens(article['text']) > self.target_tokens:
                            chunks.extend(self._split_article_by_paragraphs(
                                article['text'], page_no, article['number']
                            ))
                        else:
                            chunks.append(self._create_chunk(
                                article['text'], page_no, f"제{article['number']}조"
                            ))

            # 표 처리
            if table.content.strip():
                chunks.append(self._create_chunk(
                    table.content, page_no, 
                    f"법령 별표 (page {page_no})"
                ))

            current_line = table.end_line + 1

        # 마지막 텍스트
        if current_line < len(lines):
            after_text = '\n'.join(lines[current_line:])
            if after_text.strip():
                articles = self._split_by_articles(after_text)
                for article in articles:
                    if self._count_tokens(article['text']) > self.target_tokens:
                        chunks.extend(self._split_article_by_paragraphs(
                            article['text'], page_no, article['number']
                        ))
                    else:
                        chunks.append(self._create_chunk(
                            article['text'], page_no, f"제{article['number']}조"
                        ))

        return chunks

    def _chunk_with_tables_manual(self, text: str, page_no: int, 
                                  tables: List[TableRegion], blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """매뉴얼 문서의 표 포함 청킹"""
        chunks = []
        lines = text.split('\n')
        table_regions = sorted(tables, key=lambda t: t.start_line)
        current_line = 0

        for table in table_regions:
            # 표 이전 텍스트 - 절차 단위로 처리
            if current_line < table.start_line:
                before_text = '\n'.join(lines[current_line:table.start_line])
                if before_text.strip():
                    procedures = self._split_by_procedures(before_text)
                    for proc in procedures:
                        if self._is_complex_procedure(proc['text']):
                            chunks.extend(self._split_complex_procedure(
                                proc['text'], page_no, proc['id']
                            ))
                        else:
                            chunks.append(self._create_chunk(
                                proc['text'], page_no, proc['id']
                            ))

            # 표 처리
            if table.content.strip():
                chunks.append(self._create_chunk(
                    table.content, page_no, 
                    f"매뉴얼 표 (page {page_no})"
                ))

            current_line = table.end_line + 1

        # 마지막 텍스트
        if current_line < len(lines):
            after_text = '\n'.join(lines[current_line:])
            if after_text.strip():
                procedures = self._split_by_procedures(after_text)
                for proc in procedures:
                    if self._is_complex_procedure(proc['text']):
                        chunks.extend(self._split_complex_procedure(
                            proc['text'], page_no, proc['id']
                        ))
                    else:
                        chunks.append(self._create_chunk(
                            proc['text'], page_no, proc['id']
                        ))

        return chunks

    def _split_large_table(self, table_text: str, page_no: int, section: str) -> List[Tuple[str, Dict]]:
        """큰 표를 행 단위로 분할"""
        chunks = []
        lines = table_text.split('\n')

        # 헤더 추출
        header_lines = []
        for i, line in enumerate(lines[:3]):
            if any(kw in line for kw in ['구분', '항목', '내용', '번호']):
                header_lines.append(line)
            elif '─' in line or '═' in line or '|' in line:
                header_lines.append(line)

        header = '\n'.join(header_lines) if header_lines else ""
        data_start = len(header_lines)

        # 데이터 행 그룹핑
        current_chunk = header + "\n" if header else ""
        current_tokens = self._count_tokens(header)

        for line in lines[data_start:]:
            line_tokens = self._count_tokens(line)

            if current_tokens + line_tokens <= self.target_tokens:
                current_chunk += line + "\n"
                current_tokens += line_tokens
            else:
                if current_chunk.strip():
                    chunks.append(self._create_chunk(current_chunk, page_no, section))
                current_chunk = (header + "\n" if header else "") + line + "\n"
                current_tokens = self._count_tokens(header) + line_tokens

        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk, page_no, section))

        return chunks
    
    def _process_cross_page_continuity(
        self, 
        chunks: List[Tuple[str, Dict]]
    ) -> List[Tuple[str, Dict]]:
        """
        페이지 간 연속성 처리
        [수정] 페이지 경계를 넘어서는 병합은 매우 조심스럽게 수행
        """
        if len(chunks) < 2:
            return chunks
        
        processed = []
        i = 0
        
        while i < len(chunks):
            current_text, current_meta = chunks[i]
            
            # [수정] 다음 청크와 연결 가능한지 확인 (조건 강화)
            if (i + 1 < len(chunks) and 
                self._should_merge_across_pages(current_meta, chunks[i + 1][1])):
                
                next_text, next_meta = chunks[i + 1]
                
                # 두 청크 병합
                merged_text = self._merge_chunk_texts(current_text, next_text)
                merged_meta = self._merge_chunk_metadata(current_meta, next_meta)
                
                processed.append((merged_text, merged_meta))
                i += 2
            else:
                processed.append((current_text, current_meta))
                i += 1
        
        return processed
    
    def _should_merge_across_pages(self, meta1: Dict, meta2: Dict) -> bool:
        """
        페이지 간 청크 병합 여부 판단
        [수정] 매우 보수적으로 판단
        """
        # 연속 페이지가 아니면 병합하지 않음
        if abs(meta1.get('page', 0) - meta2.get('page', 0)) != 1:
            return False
        
        # 토큰 수 제한 (더 엄격하게)
        total_tokens = meta1.get('token_count', 0) + meta2.get('token_count', 0)
        if total_tokens > self.target_tokens:  # max_chunk_tokens가 아닌 target_tokens
            return False
        
        # 표나 구조화된 내용은 병합하지 않음
        if 'table' in meta1.get('type', '') or 'table' in meta2.get('type', ''):
            return False
        if 'image_page' in (meta1.get('type', ''), meta2.get('type', '')):
            return False
        
        # 같은 섹션/조항이고, 둘 다 매우 작은 청크일 때만 병합
        section1 = meta1.get('section', '')
        section2 = meta2.get('section', '')
        
        if (section1 and section2 and section1 == section2 and
            meta1.get('token_count', 0) < self.min_chunk_tokens and
            meta2.get('token_count', 0) < self.min_chunk_tokens):
            return True
        
        return False
    
    def _merge_chunk_texts(self, text1: str, text2: str) -> str:
        """두 청크 텍스트 병합"""
        return text1.rstrip() + "\n\n" + text2.lstrip()
    
    def _merge_chunk_metadata(self, meta1: Dict, meta2: Dict) -> Dict:
        """두 청크 메타데이터 병합"""
        merged = meta1.copy()
        
        # 페이지 범위 병합
        pages1 = meta1.get('pages', [meta1.get('page')])
        pages2 = meta2.get('pages', [meta2.get('page')])
        merged['pages'] = sorted(set(pages1 + pages2))
        merged['page'] = merged['pages'][0] if merged['pages'] else meta1.get('page', 1)
        
        # 토큰 수 합산
        merged['token_count'] = meta1.get('token_count', 0) + meta2.get('token_count', 0)
        
        # bbox 병합
        bboxes1 = meta1.get('bboxes', {})
        bboxes2 = meta2.get('bboxes', {})
        merged['bboxes'] = {**bboxes1, **bboxes2}
        
        return merged
    
    def _validate_and_clean_chunks(
        self, 
        chunks: List[Tuple[str, Dict]]
    ) -> List[Tuple[str, Dict]]:
        """최종 청크 검증 및 정리"""
        cleaned = []
        
        for text, meta in chunks:
            # 빈 청크 제거
            if not text.strip():
                continue
            
            # 너무 작은 청크는 다음 청크와 병합 시도
            token_count = self._count_tokens(text)
            if token_count < self.min_chunk_tokens and cleaned:
                prev_text, prev_meta = cleaned[-1]
                prev_tokens = prev_meta.get('token_count', 0)
                
                # 이전 청크와 병합 가능한지 확인
                if prev_tokens + token_count <= self.max_chunk_tokens:
                    merged_text = self._merge_chunk_texts(prev_text, text)
                    merged_meta = self._merge_chunk_metadata(prev_meta, meta)
                    cleaned[-1] = (merged_text, merged_meta)
                    continue
            
            # 메타데이터 정리
            cleaned_meta = {
                'page': meta.get('page', 1),
                'pages': meta.get('pages', [meta.get('page', 1)]),
                'section': meta.get('section', ''),
                'token_count': token_count,
                'type': meta.get('type', 'text'),
                'bboxes': meta.get('bboxes', {})
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
            'type': 'table' if 'table' in section.lower() or 'table' in text.lower() else 'text',
            'bboxes': {}
        }
        return (text.strip(), metadata)
    
    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        try:
            tokens = self.encoder(text)
            return len(tokens) if tokens else len(text.split())
        except:
            return len(text.split())


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
    legal_indicators = ['조', '항', '호', '법', 'IAEA', 'INFCIRC', '보장조치', '원자력']
    if not any(indicator in full_text for indicator in legal_indicators):
        # 법령/매뉴얼이 아니면 빈 결과 반환 (다른 청커로 폴백)
        return []
    
    chunker = NuclearLegalChunker(encoder_fn, target_tokens, overlap_tokens)
    return chunker.chunk_pages(pages_std, layout_blocks, min_chunk_tokens)