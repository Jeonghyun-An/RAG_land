# app/services/chunker.py
"""
스마트 청킹 모듈 - 개선된 버전
- 의미론적 연속성 보장
- 문단간 유기성 보존  
- 다양한 문서 구조 대응
- 토큰 기반 적응형 분할
"""
from __future__ import annotations
import json
import re
from typing import List, Tuple, Dict, Optional, Callable, Any
import math

class SmartChunker:
    """개선된 스마트 청킹 클래스"""
    
    def __init__(self, encoder_fn: Callable, target_tokens: int = 400, overlap_tokens: int = 100):
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = 50
        self.max_chunk_tokens = target_tokens * 2
        
        # 문서 구조 패턴
        self.structure_patterns = {
            'header': re.compile(r'^(?:제\s*\d+\s*[조절항장편]|[A-Z0-9]+\.\s|\d+\.\d+\s)', re.MULTILINE),
            'list_item': re.compile(r'^[\s]*(?:\d+\.|[가나다라마바사아자차카타파하]\.|\([가나다라마바사아자차카타파하]\)|\d+\))', re.MULTILINE),
            'quote': re.compile(r'^[\s]*(?:"|"|'|'|※|□|▪|▫)', re.MULTILINE),
            'table_line': re.compile(r'[\|\+\-]{3,}|[┌┐└┘├┤┬┴┼─│]'),
        }
        
        # 연결어 패턴
        self.connective_words = [
            '따라서', '그러므로', '그런데', '그러나', '하지만', '또한', '또한', '더욱이',
            '즉', '다시 말해서', '예를 들어', '구체적으로', '특히', '반면에',
            '이에 따라', '이와 같이', '이와 더불어', '이와 관련하여'
        ]
        
    def chunk_pages(self, pages_std: List[Tuple[int, str]], 
                   layout_blocks: Optional[Dict[int, List[Dict]]] = None) -> List[Tuple[str, Dict]]:
        """페이지별 스마트 청킹"""
        if not pages_std:
            return []
            
        all_chunks = []
        
        for page_no, text in pages_std:
            if not text or not text.strip():
                continue
                
            # 문서 구조 분석
            structure_info = self._analyze_document_structure(text)
            
            # 구조 기반 청킹
            if structure_info['has_clear_structure']:
                page_chunks = self._structure_based_chunking(text, page_no, structure_info)
            else:
                # 의미론적 청킹
                page_chunks = self._semantic_chunking(text, page_no)
            
            all_chunks.extend(page_chunks)
        
        # 페이지 간 연속성 처리
        connected_chunks = self._process_inter_page_continuity(all_chunks)
        
        # 최종 검증 및 정리
        return self._finalize_chunks(connected_chunks)
    
    def _analyze_document_structure(self, text: str) -> Dict:
        """문서 구조 분석"""
        structure_info = {
            'has_clear_structure': False,
            'structure_type': 'unstructured',
            'sections': [],
            'list_density': 0,
            'table_density': 0
        }
        
        lines = text.split('\n')
        total_lines = len(lines)
        
        if total_lines == 0:
            return structure_info
        
        # 구조적 요소 카운트
        header_count = 0
        list_count = 0
        table_count = 0
        
        for line in lines:
            if self.structure_patterns['header'].match(line.strip()):
                header_count += 1
            elif self.structure_patterns['list_item'].match(line.strip()):
                list_count += 1
            elif self.structure_patterns['table_line'].search(line):
                table_count += 1
        
        # 밀도 계산
        structure_info['list_density'] = list_count / total_lines
        structure_info['table_density'] = table_count / total_lines
        
        # 구조 유형 판단
        if header_count >= 3:  # 헤더가 3개 이상
            structure_info['has_clear_structure'] = True
            structure_info['structure_type'] = 'hierarchical'
        elif structure_info['list_density'] > 0.3:  # 목록 비율 30% 이상
            structure_info['has_clear_structure'] = True
            structure_info['structure_type'] = 'list_heavy'
        elif structure_info['table_density'] > 0.2:  # 표 비율 20% 이상
            structure_info['has_clear_structure'] = True
            structure_info['structure_type'] = 'table_heavy'
        
        return structure_info
    
    def _structure_based_chunking(self, text: str, page_no: int, structure_info: Dict) -> List[Tuple[str, Dict]]:
        """구조 기반 청킹"""
        chunks = []
        
        if structure_info['structure_type'] == 'hierarchical':
            chunks = self._hierarchical_chunking(text, page_no)
        elif structure_info['structure_type'] == 'list_heavy':
            chunks = self._list_based_chunking(text, page_no)
        elif structure_info['structure_type'] == 'table_heavy':
            chunks = self._table_preserving_chunking(text, page_no)
        else:
            chunks = self._semantic_chunking(text, page_no)
        
        return chunks
    
    def _hierarchical_chunking(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """계층적 구조 기반 청킹"""
        chunks = []
        sections = self._split_by_headers(text)
        
        for section in sections:
            section_text = section['content']
            section_header = section['header']
            
            if self._count_tokens(section_text) <= self.target_tokens:
                # 섹션이 적당한 크기면 그대로 청크로
                chunks.append(self._create_chunk(section_text, page_no, section_header))
            else:
                # 섹션이 크면 문단 단위로 세분화
                sub_chunks = self._subdivide_section(section_text, page_no, section_header)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_by_headers(self, text: str) -> List[Dict]:
        """헤더 기준으로 섹션 분할"""
        sections = []
        lines = text.split('\n')
        current_section = {'header': '', 'content': ''}
        
        for line in lines:
            if self.structure_patterns['header'].match(line.strip()):
                # 이전 섹션 저장
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # 새 섹션 시작
                current_section = {
                    'header': line.strip(),
                    'content': line
                }
            else:
                current_section['content'] += '\n' + line
        
        # 마지막 섹션 처리
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _list_based_chunking(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """목록 구조 기반 청킹"""
        chunks = []
        list_groups = self._group_list_items(text)
        
        for group in list_groups:
            if group['type'] == 'list':
                # 목록 항목들을 적절한 크기로 그룹핑
                list_chunks = self._chunk_list_group(group['items'], page_no)
                chunks.extend(list_chunks)
            else:
                # 일반 텍스트
                if self._count_tokens(group['content']) <= self.target_tokens:
                    chunks.append(self._create_chunk(group['content'], page_no))
                else:
                    sub_chunks = self._split_large_text(group['content'], page_no)
                    chunks.extend(sub_chunks)
        
        return chunks
    
    def _group_list_items(self, text: str) -> List[Dict]:
        """목록 항목들을 그룹핑"""
        groups = []
        lines = text.split('\n')
        current_group = {'type': 'text', 'content': '', 'items': []}
        
        for line in lines:
            if self.structure_patterns['list_item'].match(line.strip()):
                # 목록 항목 발견
                if current_group['type'] == 'text' and current_group['content'].strip():
                    # 이전 텍스트 그룹 저장
                    groups.append(current_group)
                    current_group = {'type': 'list', 'content': '', 'items': []}
                
                current_group['type'] = 'list'
                current_group['items'].append(line.strip())
                current_group['content'] += line + '\n'
            else:
                # 일반 텍스트
                if current_group['type'] == 'list' and current_group['items']:
                    # 이전 목록 그룹 저장
                    groups.append(current_group)
                    current_group = {'type': 'text', 'content': '', 'items': []}
                
                current_group['type'] = 'text'
                current_group['content'] += line + '\n'
        
        # 마지막 그룹 처리
        if current_group['content'].strip() or current_group['items']:
            groups.append(current_group)
        
        return groups
    
    def _chunk_list_group(self, items: List[str], page_no: int) -> List[Tuple[str, Dict]]:
        """목록 그룹을 청킹"""
        chunks = []
        current_items = []
        current_tokens = 0
        
        for item in items:
            item_tokens = self._count_tokens(item)
            
            if current_tokens + item_tokens <= self.target_tokens:
                current_items.append(item)
                current_tokens += item_tokens
            else:
                # 현재 그룹 완료
                if current_items:
                    list_text = '\n'.join(current_items)
                    chunks.append(self._create_chunk(list_text, page_no, "목록"))
                
                current_items = [item]
                current_tokens = item_tokens
        
        # 마지막 그룹 처리
        if current_items:
            list_text = '\n'.join(current_items)
            chunks.append(self._create_chunk(list_text, page_no, "목록"))
        
        return chunks
    
    def _table_preserving_chunking(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """표 구조 보존 청킹"""
        chunks = []
        segments = self._separate_tables_and_text(text)
        
        for segment in segments:
            if segment['type'] == 'table':
                # 표는 가능한 한 통째로 보존
                if self._count_tokens(segment['content']) <= self.max_chunk_tokens:
                    chunks.append(self._create_chunk(segment['content'], page_no, "표"))
                else:
                    # 너무 크면 행 단위로 분할
                    table_chunks = self._split_large_table(segment['content'], page_no)
                    chunks.extend(table_chunks)
            else:
                # 일반 텍스트
                if self._count_tokens(segment['content']) <= self.target_tokens:
                    chunks.append(self._create_chunk(segment['content'], page_no))
                else:
                    sub_chunks = self._semantic_chunking(segment['content'], page_no)
                    chunks.extend(sub_chunks)
        
        return chunks
    
    def _separate_tables_and_text(self, text: str) -> List[Dict]:
        """표와 일반 텍스트 분리"""
        segments = []
        lines = text.split('\n')
        current_segment = {'type': 'text', 'content': ''}
        in_table = False
        
        for line in lines:
            is_table_line = bool(self.structure_patterns['table_line'].search(line))
            
            if is_table_line and not in_table:
                # 표 시작
                if current_segment['content'].strip():
                    segments.append(current_segment)
                current_segment = {'type': 'table', 'content': line + '\n'}
                in_table = True
            elif is_table_line and in_table:
                # 표 계속
                current_segment['content'] += line + '\n'
            elif not is_table_line and in_table:
                # 표 끝? (빈 줄 확인)
                if line.strip():
                    current_segment['content'] += line + '\n'
                else:
                    # 표 종료
                    segments.append(current_segment)
                    current_segment = {'type': 'text', 'content': ''}
                    in_table = False
            else:
                # 일반 텍스트
                if in_table:
                    # 표 종료
                    segments.append(current_segment)
                    current_segment = {'type': 'text', 'content': line + '\n'}
                    in_table = False
                else:
                    current_segment['content'] += line + '\n'
        
        # 마지막 세그먼트 처리
        if current_segment['content'].strip():
            segments.append(current_segment)
        
        return segments
    
    def _split_large_table(self, table_text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """큰 표를 행 단위로 분할"""
        chunks = []
        lines = table_text.split('\n')
        
        # 표 헤더 찾기
        header_lines = []
        data_lines = []
        
        for i, line in enumerate(lines[:5]):  # 처음 5줄에서 헤더 찾기
            if self.structure_patterns['table_line'].search(line) and '+' in line:
                header_lines = lines[:i+3]  # 헤더 라인들
                data_lines = lines[i+3:]
                break
        
        if not header_lines:
            # 헤더 구분이 안되면 임시로 분할
            chunk_size = max(10, len(lines) // 3)
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i+chunk_size]
                chunk_text = '\n'.join(chunk_lines)
                if chunk_text.strip():
                    chunks.append(self._create_chunk(chunk_text, page_no, "표 일부"))
        else:
            # 헤더와 함께 데이터 행들을 적절히 분할
            current_chunk = '\n'.join(header_lines) + '\n'
            current_tokens = self._count_tokens(current_chunk)
            
            for line in data_lines:
                line_tokens = self._count_tokens(line)
                
                if current_tokens + line_tokens <= self.target_tokens:
                    current_chunk += line + '\n'
                    current_tokens += line_tokens
                else:
                    # 현재 청크 완료
                    if current_chunk.strip():
                        chunks.append(self._create_chunk(current_chunk, page_no, "표 일부"))
                    
                    # 새 청크 시작 (헤더 포함)
                    current_chunk = '\n'.join(header_lines) + '\n' + line + '\n'
                    current_tokens = self._count_tokens(current_chunk)
            
            # 마지막 청크 처리
            if current_chunk.strip():
                chunks.append(self._create_chunk(current_chunk, page_no, "표 일부"))
        
        return chunks
    
    def _semantic_chunking(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """의미론적 청킹"""
        chunks = []
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = ""
        current_tokens = 0
        
        for i, paragraph in enumerate(paragraphs):
            para_tokens = self._count_tokens(paragraph)
            
            # 문맥 연결성 점수 계산
            continuity_score = 0
            if i > 0:
                continuity_score = self._calculate_semantic_continuity(paragraph, paragraphs[i-1])
            
            # 연결 조건 확인
            should_connect = (
                current_tokens + para_tokens <= self.target_tokens or
                (continuity_score > 0.7 and current_tokens + para_tokens <= self.max_chunk_tokens)
            )
            
            if should_connect:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                current_tokens += para_tokens
            else:
                # 현재 청크 완료
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, page_no))
                
                # 문단이 너무 크면 문장 단위로 분할
                if para_tokens > self.max_chunk_tokens:
                    sub_chunks = self._split_large_paragraph(paragraph, page_no)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = paragraph
                    current_tokens = para_tokens
        
        # 마지막 청크 처리
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, page_no))
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """텍스트를 문단으로 분할"""
        # 이중 개행을 기준으로 분할
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 빈 문단 제거 및 정리
        clean_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para:
                clean_paragraphs.append(para)
        
        return clean_paragraphs
    
    def _calculate_semantic_continuity(self, current: str, previous: str) -> float:
        """두 문단 간의 의미론적 연속성 점수 계산"""
        score = 0.0
        
        # 1. 연결어 확인
        current_lower = current.lower()
        for connective in self.connective_words:
            if current.startswith(connective) or current_lower.startswith(connective):
                score += 0.4
                break
        
        # 2. 키워드 연속성
        current_words = set(re.findall(r'[가-힣]{2,}|[A-Za-z]{3,}', current))
        previous_words = set(re.findall(r'[가-힣]{2,}|[A-Za-z]{3,}', previous))
        
        if current_words and previous_words:
            common_words = current_words & previous_words
            similarity = len(common_words) / max(len(current_words), len(previous_words))
            score += similarity * 0.3
        
        # 3. 번호/순서 연속성
        current_numbers = re.findall(r'\d+', current)
        previous_numbers = re.findall(r'\d+', previous)
        
        if current_numbers and previous_numbers:
            try:
                if int(current_numbers[0]) == int(previous_numbers[-1]) + 1:
                    score += 0.3
            except (ValueError, IndexError):
                pass
        
        # 4. 구조적 연속성 (같은 들여쓰기, 목록 구조 등)
        current_indent = len(current) - len(current.lstrip())
        previous_indent = len(previous) - len(previous.lstrip())
        
        if abs(current_indent - previous_indent) <= 2:  # 들여쓰기 차이 2 이하
            score += 0.2
        
        return min(score, 1.0)
    
    def _split_large_paragraph(self, paragraph: str, page_no: int) -> List[Tuple[str, Dict]]:
        """큰 문단을 문장 단위로 분할"""
        chunks = []
        sentences = self._split_into_sentences(paragraph)
        
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
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장으로 분할"""
        # 한국어 문장 종결 패턴
        sentence_endings = re.compile(r'[.!?]+\s*(?=[A-Z가-힣]|$)')
        sentences = sentence_endings.split(text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _subdivide_section(self, section_text: str, page_no: int, section_header: str) -> List[Tuple[str, Dict]]:
        """섹션을 세분화"""
        chunks = []
        paragraphs = self._split_into_paragraphs(section_text)
        
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            para_tokens = self._count_tokens(paragraph)
            
            if current_tokens + para_tokens <= self.target_tokens:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, page_no, section_header))
                current_chunk = paragraph
                current_tokens = para_tokens
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, page_no, section_header))
        
        return chunks
    
    def _split_large_text(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """큰 텍스트를 적절히 분할"""
        return self._semantic_chunking(text, page_no)
    
    def _process_inter_page_continuity(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """페이지 간 연속성 처리"""
        if len(chunks) < 2:
            return chunks
        
        processed = []
        i = 0
        
        while i < len(chunks):
            current_text, current_meta = chunks[i]
            
            # 다음 청크와 연결 가능한지 확인
            if (i + 1 < len(chunks) and 
                self._should_merge_chunks(current_meta, chunks[i + 1][1])):
                
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
    
    def _should_merge_chunks(self, meta1: Dict, meta2: Dict) -> bool:
        """두 청크 병합 여부 판단"""
        # 연속 페이지 확인
        if abs(meta1.get('page', 0) - meta2.get('page', 0)) != 1:
            return False
        
        # 토큰 수 제한
        total_tokens = meta1.get('token_count', 0) + meta2.get('token_count', 0)
        if total_tokens > self.max_chunk_tokens:
            return False
        
        # 같은 섹션 확인
        section1 = meta1.get('section', '')
        section2 = meta2.get('section', '')
        
        if section1 and section2 and section1 == section2:
            return True
        
        # 구조적 연속성 확인
        type1 = meta1.get('type', '')
        type2 = meta2.get('type', '')
        
        # 같은 타입의 청크들은 연결 가능성 높음
        if type1 == type2 and 'table' not in type1:  # 표는 제외
            return True
        
        return False
    
    def _merge_chunk_texts(self, text1: str, text2: str) -> str:
        """두 청크 텍스트 병합"""
        clean_text1 = self._strip_meta_line(text1)
        clean_text2 = self._strip_meta_line(text2)
        
        return clean_text1 + "\n\n" + clean_text2
    
    def _merge_chunk_metadata(self, meta1: Dict, meta2: Dict) -> Dict:
        """두 청크 메타데이터 병합"""
        merged = meta1.copy()
        
        # 페이지 범위 확장
        pages1 = meta1.get('pages', [meta1.get('page', 0)])
        pages2 = meta2.get('pages', [meta2.get('page', 0)])
        merged['pages'] = sorted(set(pages1 + pages2))
        merged['page'] = merged['pages'][0]
        
        # 토큰 수 합계
        merged['token_count'] = meta1.get('token_count', 0) + meta2.get('token_count', 0)
        
        return merged
    
    def _finalize_chunks(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """최종 청크 검증 및 정리"""
        finalized = []
        
        for text, meta in chunks:
            # 최소 토큰 수 확인
            if meta.get('token_count', 0) < self.min_chunk_tokens:
                continue
            
            # 텍스트 정리
            clean_text = self._clean_chunk_text(text)
            if not clean_text.strip():
                continue
            
            # 메타데이터 정규화
            clean_meta = self._normalize_chunk_metadata(meta, clean_text)
            
            # META 라인 추가
            meta_line = "META: " + json.dumps(clean_meta, ensure_ascii=False)
            final_text = meta_line + "\n" + self._strip_meta_line(clean_text)
            
            finalized.append((final_text, clean_meta))
        
        return finalized
    
    def _clean_chunk_text(self, text: str) -> str:
        """청크 텍스트 정리"""
        # 이상한 라벨 제거 ("인접행 묶음" 등)
        text = re.sub(r'\b인접행\s*묶음\b', '', text)
        text = re.sub(r'\b[가-힣]*\s*묶음\b', '', text)
        text = re.sub(r'\b\w*\s*묶음\b', '', text)
        
        # 과도한 공백/개행 정리
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _normalize_chunk_metadata(self, meta: Dict, text: str) -> Dict:
        """청크 메타데이터 정규화"""
        normalized = {
            "type": meta.get('type', 'smart_chunk'),
            "section": str(meta.get('section', ''))[:512],  # 길이 제한
            "page": meta.get('page', 0),
            "pages": meta.get('pages', [meta.get('page', 0)]),
            "token_count": self._count_tokens(text),
            "bboxes": meta.get('bboxes', {}),
        }
        
        return normalized
    
    def _create_chunk(self, text: str, page_no: int, section: str = "") -> Tuple[str, Dict]:
        """청크 생성 헬퍼 함수"""
        meta = {
            "type": "smart_chunk",
            "section": section,
            "page": page_no,
            "pages": [page_no],
            "token_count": self._count_tokens(text),
            "bboxes": {},
        }
        
        return (text, meta)
    
    def _strip_meta_line(self, text: str) -> str:
        """META 라인 제거"""
        if text.startswith("META:"):
            nl_pos = text.find("\n")
            return text[nl_pos + 1:] if nl_pos != -1 else ""
        return text
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        if not text:
            return 0
        try:
            return len(self.encoder(text))
        except:
            # 폴백: 대략적 추정 (한국어 특성 반영)
            korean_chars = len(re.findall(r'[가-힣]', text))
            english_words = len(re.findall(r'[A-Za-z]+', text))
            numbers = len(re.findall(r'\d+', text))
            
            return int(korean_chars * 0.8 + english_words * 1.2 + numbers * 0.5)


class SmartChunkerPlus(SmartChunker):
    """스마트 청커 플러스 버전 - 레이아웃 정보 활용 기능 추가"""
    
    def __init__(self, encoder_fn: Callable, target_tokens: int = 400, overlap_tokens: int = 100):
        super().__init__(encoder_fn, target_tokens, overlap_tokens)
        
    def chunk_pages_plus(self, pages_std: List[Tuple[int, str]], 
                        layout_blocks: Optional[Dict[int, List[Dict]]] = None) -> List[Tuple[str, Dict]]:
        """레이아웃 정보를 활용한 스마트 청킹"""
        if not pages_std:
            return []
        
        if not layout_blocks:
            # 레이아웃 정보 없으면 기본 스마트 청킹
            return self.chunk_pages(pages_std)
        
        all_chunks = []
        
        for page_no, text in pages_std:
            if not text or not text.strip():
                continue
            
            # 레이아웃 정보 활용
            page_blocks = layout_blocks.get(page_no, [])
            if page_blocks:
                page_chunks = self._layout_enhanced_chunking(text, page_no, page_blocks)
            else:
                page_chunks = self._semantic_chunking(text, page_no)
            
            all_chunks.extend(page_chunks)
        
        return self._finalize_chunks(all_chunks)
    
    def _layout_enhanced_chunking(self, text: str, page_no: int, blocks: List[Dict]) -> List[Tuple[str, Dict]]:
        """레이아웃 정보를 활용한 향상된 청킹"""
        chunks = []
        
        # 블록 정보에서 텍스트와 위치 추출
        text_blocks = []
        for block in blocks:
            block_text = block.get('text', '').strip()
            if block_text:
                text_blocks.append({
                    'text': block_text,
                    'bbox': block.get('bbox', {}),
                    'y': block.get('bbox', {}).get('y0', 0)
                })
        
        # Y 좌표 기준 정렬 (위에서 아래로)
        text_blocks.sort(key=lambda b: b['y'])
        
        # 블록들을 의미론적으로 그룹핑
        semantic_groups = self._group_blocks_semantically(text_blocks)
        
        # 그룹별 청킹
        for group in semantic_groups:
            group_text = '\n\n'.join(block['text'] for block in group['blocks'])
            
            if self._count_tokens(group_text) <= self.target_tokens:
                chunks.append(self._create_chunk(group_text, page_no, group.get('section', '')))
            else:
                # 큰 그룹은 세분화
                sub_chunks = self._subdivide_block_group(group, page_no)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _group_blocks_semantically(self, blocks: List[Dict]) -> List[Dict]:
        """블록들을 의미론적으로 그룹핑"""
        if not blocks:
            return []
        
        groups = []
        current_group = {'blocks': [blocks[0]], 'section': ''}
        
        for i in range(1, len(blocks)):
            current_block = blocks[i]
            prev_block = blocks[i-1]
            
            # 수직 거리 계산
            y_distance = current_block['y'] - prev_block.get('bbox', {}).get('y1', prev_block['y'])
            
            # 그룹 연속성 판단
            should_continue_group = (
                y_distance < 30 and  # 30pt 미만의 간격
                self._blocks_are_semantically_related(current_block, prev_block)
            )
            
            if should_continue_group:
                current_group['blocks'].append(current_block)
            else:
                # 새 그룹 시작
                groups.append(current_group)
                current_group = {'blocks': [current_block], 'section': ''}
        
        # 마지막 그룹 추가
        groups.append(current_group)
        
        return groups
    
    def _blocks_are_semantically_related(self, block1: Dict, block2: Dict) -> bool:
        """두 블록이 의미론적으로 연관되어 있는지 확인"""
        text1 = block1['text'].lower()
        text2 = block2['text'].lower()
        
        # 키워드 유사성
        words1 = set(re.findall(r'[가-힣]{2,}|[A-Za-z]{3,}', text1))
        words2 = set(re.findall(r'[가-힣]{2,}|[A-Za-z]{3,}', text2))
        
        if words1 and words2:
            overlap = len(words1 & words2)
            similarity = overlap / min(len(words1), len(words2))
            if similarity > 0.3:
                return True
        
        # 구조적 연속성
        if any(connector in text1 for connector in self.connective_words):
            return True
        
        return False
    
    def _subdivide_block_group(self, group: Dict, page_no: int) -> List[Tuple[str, Dict]]:
        """큰 블록 그룹을 세분화"""
        chunks = []
        blocks = group['blocks']
        
        current_chunk_blocks = []
        current_tokens = 0
        
        for block in blocks:
            block_tokens = self._count_tokens(block['text'])
            
            if current_tokens + block_tokens <= self.target_tokens:
                current_chunk_blocks.append(block)
                current_tokens += block_tokens
            else:
                # 현재 청크 완료
                if current_chunk_blocks:
                    chunk_text = '\n\n'.join(b['text'] for b in current_chunk_blocks)
                    chunks.append(self._create_chunk(chunk_text, page_no, group.get('section', '')))
                
                current_chunk_blocks = [block]
                current_tokens = block_tokens
        
        # 마지막 청크 처리
        if current_chunk_blocks:
            chunk_text = '\n\n'.join(b['text'] for b in current_chunk_blocks)
            chunks.append(self._create_chunk(chunk_text, page_no, group.get('section', '')))
        
        return chunks


# 기존 인터페이스 호환 함수들
def smart_chunk_pages(pages_std: List[Tuple[int, str]], 
                     encoder_fn: Callable,
                     target_tokens: int = 400,
                     overlap_tokens: int = 100) -> List[Tuple[str, Dict]]:
    """
    스마트 청킹 함수 (기본 버전)
    
    Args:
        pages_std: [(page_no, text), ...] 형태의 페이지 데이터
        encoder_fn: 토큰 인코딩 함수
        target_tokens: 목표 토큰 수
        overlap_tokens: 오버랩 토큰 수
    
    Returns:
        [(chunk_text, metadata), ...] 형태의 청크 리스트
    """
    if not pages_std:
        return []
    
    chunker = SmartChunker(encoder_fn, target_tokens, overlap_tokens)
    return chunker.chunk_pages(pages_std)


def smart_chunk_pages_plus(pages_std: List[Tuple[int, str]], 
                          encoder_fn: Callable,
                          target_tokens: int = 400,
                          overlap_tokens: int = 100,
                          layout_blocks: Optional[Dict[int, List[Dict]]] = None) -> List[Tuple[str, Dict]]:
    """
    스마트 청킹 함수 (플러스 버전 - 레이아웃 정보 활용)
    
    Args:
        pages_std: [(page_no, text), ...] 형태의 페이지 데이터
        encoder_fn: 토큰 인코딩 함수
        target_tokens: 목표 토큰 수
        overlap_tokens: 오버랩 토큰 수
        layout_blocks: 레이아웃 블록 정보
    
    Returns:
        [(chunk_text, metadata), ...] 형태의 청크 리스트
    """
    if not pages_std:
        return []
    
    chunker = SmartChunkerPlus(encoder_fn, target_tokens, overlap_tokens)
    return chunker.chunk_pages_plus(pages_std, layout_blocks)