# app/services/law_chunker.py
"""
원자력 법령/매뉴얼 전용 고도화 청킹 모듈
- 문단간 유기성 보존
- 조항/절/항 구조 인식
- IAEA 규정, 원자력안전법 등 특화 처리
- 의미론적 연속성 보장
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
            
            # 문서 유형에 따른 청킹
            if doc_type == 'iaea_guide':
                page_chunks = self._chunk_iaea_guide(text, page_no, page_blocks)
            elif doc_type == 'korean_law':
                page_chunks = self._chunk_korean_law(text, page_no, page_blocks)
            elif doc_type == 'technical_manual':
                page_chunks = self._chunk_technical_manual(text, page_no, page_blocks)
            else:
                # 일반 구조적 청킹
                page_chunks = self._chunk_structured_text(text, page_no, page_blocks)
                
            all_chunks.extend(page_chunks)
            
        # 크로스 페이지 연결성 처리
        connected_chunks = self._process_cross_page_continuity(all_chunks)
        
        # 최종 검증 및 정리
        final_chunks = self._validate_and_clean_chunks(connected_chunks)
        
        return final_chunks
    
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
        """IAEA 섹션별 분할"""
        sections = []
        current_section = {"id": "", "text": ""}
        
        lines = text.split('\n')
        
        for line in lines:
            # 섹션 헤더 검출
            section_match = LEGAL_PATTERNS['iaea_section'].match(line.strip())
            if section_match:
                # 이전 섹션 저장
                if current_section["text"].strip():
                    sections.append(current_section)
                
                # 새 섹션 시작
                current_section = {
                    "id": section_match.group(0),
                    "text": line
                }
            else:
                current_section["text"] += "\n" + line
        
        # 마지막 섹션 처리
        if current_section["text"].strip():
            sections.append(current_section)
            
        return sections
    
    def _split_by_articles(self, text: str) -> List[Dict]:
        """조항별 분할"""
        articles = []
        current_article = {"number": "", "text": ""}
        
        lines = text.split('\n')
        
        for line in lines:
            article_match = LEGAL_PATTERNS['article'].search(line)
            if article_match:
                # 이전 조항 저장
                if current_article["text"].strip():
                    articles.append(current_article)
                
                # 새 조항 시작
                current_article = {
                    "number": article_match.group(1),
                    "text": line
                }
            else:
                current_article["text"] += "\n" + line
        
        # 마지막 조항 처리
        if current_article["text"].strip():
            articles.append(current_article)
            
        return articles
    
    def _split_by_procedures(self, text: str) -> List[Dict]:
        """절차별 분할"""
        procedures = []
        current_proc = {"id": "", "text": ""}
        
        # 절차 패턴: "단계 N", "Step N", "N)" 등
        step_pattern = re.compile(r'^(?:단계\s*\d+|Step\s*\d+|\d+\))', re.MULTILINE | re.IGNORECASE)
        
        parts = step_pattern.split(text)
        matches = step_pattern.findall(text)
        
        for i, part in enumerate(parts):
            if i == 0 and part.strip():
                # 첫 번째 부분 (헤더)
                procedures.append({"id": "header", "text": part.strip()})
            elif i > 0:
                proc_id = matches[i-1] if i-1 < len(matches) else f"step_{i}"
                procedures.append({"id": proc_id, "text": part.strip()})
                
        return procedures
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """문단별 분할 (빈 줄 기준)"""
        paragraphs = []
        current_para = ""
        
        lines = text.split('\n')
        
        for line in lines:
            if line.strip():
                current_para += "\n" + line if current_para else line
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
        parts = self._separate_structured_content(text)
        
        for part in parts:
            if part['type'] == 'structured':
                # 구조화된 내용은 통째로 유지
                chunks.append(self._create_chunk(part['text'], page_no, section_id))
            else:
                # 일반 텍스트는 토큰 수 확인 후 처리
                if self._count_tokens(part['text']) > self.target_tokens:
                    sub_chunks = self._semantic_chunking(part['text'], page_no, section_id)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(self._create_chunk(part['text'], page_no, section_id))
                    
        return chunks
    
    def _separate_structured_content(self, text: str) -> List[Dict]:
        """구조화된 내용과 일반 텍스트 분리"""
        parts = []
        current_text = ""
        in_structure = False
        
        lines = text.split('\n')
        
        for line in lines:
            if LEGAL_PATTERNS['table_header'].search(line) or LEGAL_PATTERNS['box_structure'].search(line):
                if not in_structure and current_text.strip():
                    parts.append({"type": "text", "text": current_text.strip()})
                    current_text = ""
                in_structure = True
                current_text += "\n" + line
            else:
                if in_structure and not line.strip():
                    # 구조화된 내용 종료
                    if current_text.strip():
                        parts.append({"type": "structured", "text": current_text.strip()})
                        current_text = ""
                    in_structure = False
                else:
                    current_text += "\n" + line if current_text else line
        
        # 마지막 부분 처리
        if current_text.strip():
            part_type = "structured" if in_structure else "text"
            parts.append({"type": part_type, "text": current_text.strip()})
            
        return parts
    
    def _semantic_chunking(self, text: str, page_no: int, section_id: str) -> List[Tuple[str, Dict]]:
        """의미론적 청킹 (문맥 보존)"""
        chunks = []
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)
            
            # 문맥 연결성 확인
            continuity_score = self._calculate_continuity(sentence, sentences[i-1:i])
            
            if (current_tokens + sentence_tokens <= self.target_tokens or 
                continuity_score > 0.7):  # 연속성이 높으면 토큰 제한 완화
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
            else:
                # 현재 청크 완료
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, page_no, section_id))
                
                current_chunk = sentence
                current_tokens = sentence_tokens
        
        # 마지막 청크 처리
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, page_no, section_id))
            
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리 (한국어 특화)"""
        # 한국어 문장 종결 패턴
        sentence_end = re.compile(r'[.!?]+\s*(?=[A-Z가-힣]|$)')
        
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_continuity(self, current: str, previous: List[str]) -> float:
        """문맥 연속성 점수 계산"""
        if not previous:
            return 0.0
        
        score = 0.0
        prev_text = " ".join(previous[-2:])  # 최근 2문장 참조
        
        # 키워드 연속성
        current_keywords = self._extract_nuclear_keywords(current)
        prev_keywords = self._extract_nuclear_keywords(prev_text)
        
        if current_keywords and prev_keywords:
            common_keywords = current_keywords & prev_keywords
            score += len(common_keywords) / max(len(current_keywords), len(prev_keywords))
        
        # 지시어 연결성 ("이는", "따라서", "그러나" 등)
        connective_patterns = ['이는', '따라서', '그러나', '또한', '즉', '예를 들어']
        if any(pattern in current for pattern in connective_patterns):
            score += 0.3
        
        # 번호/순서 연속성
        if self._has_sequential_numbering(current, prev_text):
            score += 0.4
            
        return min(score, 1.0)
    
    def _extract_nuclear_keywords(self, text: str) -> set:
        """원자력 전문용어 추출"""
        keywords = set()
        text_lower = text.lower()
        
        for category, terms in NUCLEAR_KEYWORDS.items():
            for term in terms:
                if term in text_lower:
                    keywords.add(term)
                    
        return keywords
    
    def _has_sequential_numbering(self, current: str, previous: str) -> bool:
        """순차적 번호 매김 확인"""
        # 숫자 패턴 추출
        current_nums = re.findall(r'\d+', current)
        prev_nums = re.findall(r'\d+', previous)
        
        if current_nums and prev_nums:
            try:
                return int(current_nums[0]) == int(prev_nums[-1]) + 1
            except (ValueError, IndexError):
                pass
                
        return False
    
    def _process_cross_page_continuity(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """페이지 간 연속성 처리"""
        if len(chunks) < 2:
            return chunks
        
        processed = []
        i = 0
        
        while i < len(chunks):
            current_chunk, current_meta = chunks[i]
            
            # 다음 청크와의 연결성 확인
            if (i + 1 < len(chunks) and 
                self._should_merge_across_pages(current_meta, chunks[i + 1][1])):
                
                next_chunk, next_meta = chunks[i + 1]
                
                # 두 청크 병합
                merged_text = current_chunk + "\n\n" + next_chunk
                merged_meta = self._merge_chunk_metadata(current_meta, next_meta)
                
                processed.append((merged_text, merged_meta))
                i += 2  # 두 청크 건너뛰기
            else:
                processed.append((current_chunk, current_meta))
                i += 1
                
        return processed
    
    def _should_merge_across_pages(self, meta1: Dict, meta2: Dict) -> bool:
        """페이지 간 병합 필요성 판단"""
        # 연속 페이지 확인
        if abs(meta1.get('page', 0) - meta2.get('page', 0)) != 1:
            return False
        
        # 같은 섹션 확인
        if meta1.get('section') and meta2.get('section'):
            return meta1['section'] == meta2['section']
        
        # 토큰 수 확인 (병합 후에도 적정 크기인지)
        total_tokens = (meta1.get('token_count', 0) + meta2.get('token_count', 0))
        return total_tokens <= self.max_chunk_tokens
    
    def _merge_chunk_metadata(self, meta1: Dict, meta2: Dict) -> Dict:
        """청크 메타데이터 병합"""
        merged = meta1.copy()
        
        # 페이지 범위 확장
        pages1 = meta1.get('pages', [meta1.get('page', 0)])
        pages2 = meta2.get('pages', [meta2.get('page', 0)])
        merged['pages'] = sorted(set(pages1 + pages2))
        merged['page'] = merged['pages'][0]
        
        # 토큰 수 합계
        merged['token_count'] = meta1.get('token_count', 0) + meta2.get('token_count', 0)
        
        return merged
    
    def _validate_and_clean_chunks(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """최종 청크 검증 및 정리"""
        valid_chunks = []
        
        for text, meta in chunks:
            if not text or not text.strip():
                continue
                
            # 최소 토큰 수 확인
            token_count = self._count_tokens(text)
            if token_count < self.min_chunk_tokens:
                continue
            
            # 메타데이터 정규화
            clean_text = self._clean_text(text)
            clean_meta = self._normalize_metadata(meta, token_count)
            
            # META 라인 추가
            meta_line = "META: " + json.dumps(clean_meta, ensure_ascii=False)
            final_text = meta_line + "\n" + clean_text
            
            valid_chunks.append((final_text, clean_meta))
            
        return valid_chunks
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 과도한 공백 제거
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 이상한 라벨 제거 (예: "인접행 묶음")
        text = re.sub(r'\b인접행\s*묶음\b', '', text)
        text = re.sub(r'\b[가-힣]+\s*묶음\b', '', text)
        
        return text.strip()
    
    def _normalize_metadata(self, meta: Dict, token_count: int) -> Dict:
        """메타데이터 정규화"""
        normalized = {
            "type": "law_chunk",
            "section": meta.get('section', ''),
            "page": meta.get('page', 0),
            "pages": meta.get('pages', [meta.get('page', 0)]),
            "token_count": token_count,
            "bboxes": meta.get('bboxes', {}),
        }
        
        # 섹션 길이 제한
        if len(normalized['section']) > 512:
            normalized['section'] = normalized['section'][:512]
            
        return normalized
    
    def _create_chunk(self, text: str, page_no: int, section: str = "") -> Tuple[str, Dict]:
        """청크 생성 헬퍼"""
        meta = {
            "section": section,
            "page": page_no,
            "pages": [page_no],
            "token_count": self._count_tokens(text),
            "bboxes": {},
        }
        
        return (text, meta)
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        if not text:
            return 0
        try:
            return len(self.encoder(text))
        except:
            # 폴백: 대략적 추정
            return len(text.split()) * 1.3
    
    def _split_article_by_paragraphs(self, text: str, page_no: int, article_num: str) -> List[Tuple[str, Dict]]:
        """조항을 항/호별로 세분화"""
        chunks = []
        
        # 항별 분할
        paragraphs = re.split(r'\n(?=\s*①|\s*②|\s*③)', text)
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                section_id = f"제{article_num}조제{i+1}항" if i > 0 else f"제{article_num}조"
                chunks.append(self._create_chunk(para.strip(), page_no, section_id))
                
        return chunks
    
    def _is_complex_procedure(self, text: str) -> bool:
        """복잡한 절차인지 판단"""
        return (self._count_tokens(text) > self.target_tokens or 
                len(re.findall(r'\d+\)', text)) > 5)
    
    def _split_complex_procedure(self, text: str, page_no: int, proc_id: str) -> List[Tuple[str, Dict]]:
        """복잡한 절차를 세부 단계로 분할"""
        chunks = []
        
        # 세부 단계별 분할
        steps = re.split(r'\n(?=\s*\d+\))', text)
        
        for i, step in enumerate(steps):
            if step.strip():
                step_id = f"{proc_id}_step{i+1}"
                chunks.append(self._create_chunk(step.strip(), page_no, step_id))
                
        return chunks
    
    def _split_long_paragraph(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """긴 문단을 문장 단위로 분할"""
        chunks = []
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= self.target_tokens:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, page_no))
                current_chunk = sentence
                current_tokens = sentence_tokens
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, page_no))
            
        return chunks
    
    def _chunk_with_tables_iaea(self, text: str, page_no: int, 
                                tables: List[TableRegion], blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """IAEA 문서의 표 포함 청킹"""
        chunks = []
        lines = text.split('\n')
        table_regions = sorted(tables, key=lambda t: t.start_line)
        current_line = 0

        for table in table_regions:
            # 표 이전 텍스트
            if current_line < table.start_line:
                before_text = '\n'.join(lines[current_line:table.start_line])
                if before_text.strip():
                    # IAEA 섹션 분석 후 청킹
                    sections = self._split_by_iaea_sections(before_text)
                    for sec in sections:
                        chunks.extend(self._semantic_chunking(sec['text'], page_no, sec['id']))

            # 표 처리
            if table.content.strip():
                table_tokens = self._count_tokens(table.content)
                if table_tokens <= self.max_chunk_tokens:
                    chunks.append(self._create_chunk(
                        table.content, page_no, 
                        f"IAEA Table (page {page_no})"
                    ))
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


