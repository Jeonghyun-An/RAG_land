# app/services/layout_chunker.py
"""
레이아웃 인지 청킹 모듈 - 고도화 버전
- PDF 레이아웃 정보를 활용한 의미론적 청킹
- 표, 그림, 박스 구조 보존
- 문단 간 시각적 연관성 분석
- 원자력 문서의 구조적 특성 반영
"""
from __future__ import annotations
import json
import re
from typing import List, Tuple, Dict, Optional, Callable, Set
from dataclasses import dataclass
import math
from app.services.enhanced_table_detector import EnhancedTableDetector, TableRegion

@dataclass
class BBox:
    """바운딩 박스 정보"""
    x0: float
    y0: float
    x1: float
    y1: float
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2
    
    def distance_to(self, other: 'BBox') -> float:
        """다른 박스와의 중심점 거리"""
        dx = self.center_x - other.center_x
        dy = self.center_y - other.center_y
        return math.sqrt(dx*dx + dy*dy)
    
    def vertical_distance_to(self, other: 'BBox') -> float:
        """세로 방향 거리 (위아래 관계)"""
        if self.y1 <= other.y0:  # self가 위에
            return other.y0 - self.y1
        elif other.y1 <= self.y0:  # other가 위에
            return self.y0 - other.y1
        else:  # 겹침
            return 0.0
    
    def horizontal_overlap(self, other: 'BBox') -> float:
        """가로 겹침 비율 (0~1)"""
        overlap = min(self.x1, other.x1) - max(self.x0, other.x0)
        if overlap <= 0:
            return 0.0
        return overlap / min(self.width, other.width)

class LayoutBlock:
    """레이아웃 블록 (텍스트 + 위치 정보)"""
    
    def __init__(self, text: str, bbox_dict: Dict):
        self.text = text.strip()
        self.bbox = BBox(
            bbox_dict.get('x0', 0),
            bbox_dict.get('y0', 0),
            bbox_dict.get('x1', 0),
            bbox_dict.get('y1', 0)
        )
        self.block_type = self._classify_block_type()
        self.table_detector = EnhancedTableDetector()
        
    def _classify_block_type(self) -> str:
        """블록 유형 분류"""
        text = self.text.lower()
        
        # 제목/헤더
        if (len(self.text) < 100 and 
            any(pattern in text for pattern in ['제', '조', '항', '절', '장', '편'])):
            return 'header'
        
        # 표 구조
        if any(char in self.text for char in ['|', '+', '─', '┌', '┐', '└', '┘']):
            return 'table'
        
        # 목록
        if re.match(r'^\s*(?:\d+\.|[가나다라마바사]\.|\([가나다라마바사]\))', self.text):
            return 'list_item'
        
        # 각주
        if re.match(r'^\[\^\d+\]', self.text) or text.startswith('주)'):
            return 'footnote'
        
        # 인용/박스
        if self.text.startswith('"') or '※' in self.text or '□' in self.text:
            return 'quote_box'
        
        return 'paragraph'

class LayoutAwareChunker:
    """레이아웃 인지 청킹 클래스"""
    
    def __init__(self, encoder_fn: Callable, target_tokens: int = 400, overlap_tokens: int = 100):
        self.encoder = encoder_fn
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = 50
        self.slide_rows = 4
        self.table_detector = EnhancedTableDetector()
        
    def chunk_pages(self, pages_std: List[Tuple[int, str]], 
                   layout_blocks: Dict[int, List[Dict]],
                   slide_rows: int = 4) -> List[Tuple[str, Dict]]:
        """레이아웃 정보를 활용한 페이지별 청킹"""
        if not pages_std or not layout_blocks:
            return []
            
        self.slide_rows = slide_rows
        all_chunks = []
        
        for page_no, page_text in pages_std:
            if not page_text or not page_text.strip():
                continue
                
            # 페이지의 레이아웃 블록 가져오기
            blocks_data = layout_blocks.get(page_no, [])
            if not blocks_data:
                # 레이아웃 정보 없으면 기본 텍스트 청킹
                fallback_chunks = self._fallback_chunking(page_text, page_no)
                all_chunks.extend(fallback_chunks)
                continue
            
            # 레이아웃 블록 생성 및 정렬
            blocks = [LayoutBlock(block.get('text', ''), block.get('bbox', {})) 
                     for block in blocks_data if block.get('text', '').strip()]
            
            if not blocks:
                continue
                
            # 블록들을 읽기 순서로 정렬
            sorted_blocks = self._sort_blocks_by_reading_order(blocks)
            
            # 의미론적 그룹핑
            semantic_groups = self._group_blocks_semantically(sorted_blocks)
            
            # 그룹별 청킹
            page_chunks = self._chunk_semantic_groups(semantic_groups, page_no)
            
            all_chunks.extend(page_chunks)
        
        # 페이지 간 연결 처리
        connected_chunks = self._handle_cross_page_connections(all_chunks)
        
        return self._finalize_chunks(connected_chunks)
    
    def _sort_blocks_by_reading_order(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """블록들을 읽기 순서로 정렬 (좌→우, 위→아래)"""
        # Y 좌표 기준으로 행 그룹 생성
        blocks_by_row = {}
        row_tolerance = 10  # 같은 행으로 간주할 Y 좌표 차이
        
        for block in blocks:
            row_key = round(block.bbox.center_y / row_tolerance) * row_tolerance
            if row_key not in blocks_by_row:
                blocks_by_row[row_key] = []
            blocks_by_row[row_key].append(block)
        
        # 각 행 내에서 X 좌표로 정렬
        sorted_blocks = []
        for row_y in sorted(blocks_by_row.keys(),reverse=True):
            row_blocks = sorted(blocks_by_row[row_y], key=lambda b: b.bbox.x0)
            sorted_blocks.extend(row_blocks)
        
        return sorted_blocks
    
    def _group_blocks_semantically(self, blocks: List[LayoutBlock]) -> List[Dict]:
        """의미론적으로 연관된 블록들을 그룹핑"""
        groups = []
        current_group = {
            'type': 'paragraph',
            'blocks': [],
            'bbox': None
        }
        
        for i, block in enumerate(blocks):
            # 그룹 시작 조건
            if self._should_start_new_group(block, current_group, blocks[i-1:i]):
                # 현재 그룹 완료
                if current_group['blocks']:
                    groups.append(current_group)
                
                # 새 그룹 시작
                current_group = {
                    'type': block.block_type,
                    'blocks': [block],
                    'bbox': block.bbox
                }
            else:
                # 현재 그룹에 추가
                current_group['blocks'].append(block)
                current_group['bbox'] = self._merge_bboxes([current_group['bbox'], block.bbox])
                
                # 그룹 타입 업데이트 (우선순위: table > quote_box > header > list_item > paragraph)
                type_priority = {'table': 5, 'quote_box': 4, 'header': 3, 'list_item': 2, 'paragraph': 1}
                if type_priority.get(block.block_type, 0) > type_priority.get(current_group['type'], 0):
                    current_group['type'] = block.block_type
        
        # 마지막 그룹 처리
        if current_group['blocks']:
            groups.append(current_group)
        
        return groups
    
    def _should_start_new_group(self, block: LayoutBlock, current_group: Dict, prev_blocks: List[LayoutBlock]) -> bool:
        """새 그룹을 시작할지 판단"""
        if not current_group['blocks']:
            return True
        
        # 블록 타입 변화
        if block.block_type != current_group['type']:
            # 특별한 경우: paragraph와 list_item은 연속성 확인
            if not (block.block_type in ['paragraph', 'list_item'] and 
                   current_group['type'] in ['paragraph', 'list_item']):
                return True
        
        # 큰 수직 간격 (줄바꿈)
        if prev_blocks:
            prev_block = prev_blocks[-1]
            vertical_gap = block.bbox.vertical_distance_to(prev_block.bbox)
            avg_height = (block.bbox.height + prev_block.bbox.height) / 2
            
            if vertical_gap > avg_height * 1.5:  # 평균 높이의 1.5배 이상 간격
                return True
        
        # 가로 정렬 변화 (들여쓰기 등)
        if current_group['bbox']:
            x_diff = abs(block.bbox.x0 - current_group['bbox'].x0)
            if x_diff > 20:  # 20pt 이상 차이
                return True
        
        # 텍스트 크기 기준 그룹 분할
        current_text_length = sum(len(b.text) for b in current_group['blocks'])
        if current_text_length > self.target_tokens * 1.5:
            return True
        
        return False
    
    def _merge_bboxes(self, bboxes: List[BBox]) -> BBox:
        """여러 bbox를 병합"""
        if not bboxes:
            return BBox(0, 0, 0, 0)
        
        valid_bboxes = [b for b in bboxes if b is not None]
        if not valid_bboxes:
            return BBox(0, 0, 0, 0)
        
        min_x0 = min(b.x0 for b in valid_bboxes)
        min_y0 = min(b.y0 for b in valid_bboxes)
        max_x1 = max(b.x1 for b in valid_bboxes)
        max_y1 = max(b.y1 for b in valid_bboxes)
        
        return BBox(min_x0, min_y0, max_x1, max_y1)
    
    def _chunk_semantic_groups(self, groups: List[Dict], page_no: int) -> List[Tuple[str, Dict]]:
        """의미론적 그룹들을 청킹"""
        chunks = []
        
        for group in groups:
            group_text = self._extract_group_text(group)
            if not group_text.strip():
                continue
            
            group_tokens = self._count_tokens(group_text)
            
            if group_tokens <= self.target_tokens:
                # 적정 크기면 그대로 청크 생성
                chunk = self._create_chunk_from_group(group, page_no)
                chunks.append(chunk)
            else:
                # 크기가 크면 세분화
                sub_chunks = self._subdivide_large_group(group, page_no)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _extract_group_text(self, group: Dict) -> str:
        """그룹에서 텍스트 추출"""
        blocks = group['blocks']
        
        if group['type'] == 'table':
            # 표 구조는 원형 보존
            return '\n'.join(block.text for block in blocks)
        elif group['type'] == 'list_item':
            # 목록은 적절한 들여쓰기
            return '\n'.join(f"  {block.text}" if not block.text.startswith((' ', '\t')) else block.text 
                           for block in blocks)
        else:
            # 일반 텍스트는 문단 단위로 결합
            return '\n\n'.join(block.text for block in blocks)
    
    def _create_chunk_from_group(self, group: Dict, page_no: int) -> Tuple[str, Dict]:
        """그룹으로부터 청크 생성"""
        text = self._extract_group_text(group)
        
        # 섹션 정보 추출
        section = self._extract_section_info(group)
        
        # 메타데이터 구성
        meta = {
            "type": f"layout_{group['type']}",
            "section": section,
            "page": page_no,
            "pages": [page_no],
            "bboxes": self._serialize_bbox(group['bbox']),
            "block_count": len(group['blocks']),
            "token_count": self._count_tokens(text)
        }
        
        # META 라인 추가
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        final_text = meta_line + "\n" + text
        
        return (final_text, meta)
    
    def _extract_section_info(self, group: Dict) -> str:
        """그룹에서 섹션 정보 추출"""
        blocks = group['blocks']
        
        # 헤더 타입 블록에서 우선 추출
        for block in blocks:
            if block.block_type == 'header':
                return block.text[:100]  # 최대 100자
        
        # 첫 번째 블록에서 섹션 패턴 찾기
        first_text = blocks[0].text if blocks else ""
        
        # 조항 패턴
        article_match = re.search(r'제\s*\d+\s*조[가-힣\s]*', first_text)
        if article_match:
            return article_match.group(0)
        
        # 절/항 패턴
        section_match = re.search(r'제\s*\d+\s*[절항][가-힣\s]*', first_text)
        if section_match:
            return section_match.group(0)
        
        # IAEA 섹션 패턴
        iaea_match = re.search(r'\d+\.\d+(?:\.\d+)?\s*[가-힣\s]*', first_text)
        if iaea_match:
            return iaea_match.group(0)
        
        # 일반적인 제목 패턴
        if len(first_text) < 100 and not first_text.endswith('.'):
            return first_text
        
        return ""
    
    def _serialize_bbox(self, bbox: BBox) -> Dict:
        """BBox를 직렬화"""
        if not bbox:
            return {}
        
        return {
            "x0": bbox.x0,
            "y0": bbox.y0,
            "x1": bbox.x1,
            "y1": bbox.y1
        }
    
    def _subdivide_large_group(self, group: Dict, page_no: int) -> List[Tuple[str, Dict]]:
        """큰 그룹을 세분화"""
        blocks = group['blocks']
        chunks = []
        
        if group['type'] == 'table':
            # 표는 행 단위로 분할
            chunks.extend(self._split_table_by_rows(blocks, page_no))
        elif group['type'] == 'list_item':
            # 목록은 항목별로 분할 후 재그룹
            chunks.extend(self._split_list_items(blocks, page_no))
        else:
            # 일반 텍스트는 토큰 수 기준으로 분할
            chunks.extend(self._split_blocks_by_tokens(blocks, page_no, group['type']))
        
        return chunks
    
    def _split_table_by_rows(self, blocks: List[LayoutBlock], page_no: int) -> List[Tuple[str, Dict]]:
        """표를 행 단위로 분할"""
        chunks = []
        current_rows = []
        current_tokens = 0
        
        for block in blocks:
            block_tokens = self._count_tokens(block.text)
            
            if current_tokens + block_tokens <= self.target_tokens:
                current_rows.append(block)
                current_tokens += block_tokens
            else:
                if current_rows:
                    chunks.append(self._create_table_chunk(current_rows, page_no))
                current_rows = [block]
                current_tokens = block_tokens
        
        if current_rows:
            chunks.append(self._create_table_chunk(current_rows, page_no))
        
        return chunks
    
    def _create_table_chunk(self, blocks: List[LayoutBlock], page_no: int) -> Tuple[str, Dict]:
        """표 청크 생성"""
        text = '\n'.join(block.text for block in blocks)
        bbox = self._merge_bboxes([block.bbox for block in blocks])
        
        meta = {
            "type": "layout_table_section",
            "section": "표 구조",
            "page": page_no,
            "pages": [page_no],
            "bboxes": self._serialize_bbox(bbox),
            "block_count": len(blocks),
            "token_count": self._count_tokens(text)
        }
        
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        final_text = meta_line + "\n" + text
        
        return (final_text, meta)
    
    def _split_list_items(self, blocks: List[LayoutBlock], page_no: int) -> List[Tuple[str, Dict]]:
        """목록 항목들을 적절히 그룹화"""
        chunks = []
        current_items = []
        current_tokens = 0
        
        for block in blocks:
            block_tokens = self._count_tokens(block.text)
            
            if current_tokens + block_tokens <= self.target_tokens:
                current_items.append(block)
                current_tokens += block_tokens
            else:
                if current_items:
                    chunks.append(self._create_list_chunk(current_items, page_no))
                current_items = [block]
                current_tokens = block_tokens
        
        if current_items:
            chunks.append(self._create_list_chunk(current_items, page_no))
        
        return chunks
    
    def _create_list_chunk(self, blocks: List[LayoutBlock], page_no: int) -> Tuple[str, Dict]:
        """목록 청크 생성"""
        text = '\n'.join(f"  {block.text}" if not block.text.startswith((' ', '\t')) else block.text 
                        for block in blocks)
        bbox = self._merge_bboxes([block.bbox for block in blocks])
        
        meta = {
            "type": "layout_list_section",
            "section": "목록",
            "page": page_no,
            "pages": [page_no],
            "bboxes": self._serialize_bbox(bbox),
            "block_count": len(blocks),
            "token_count": self._count_tokens(text)
        }
        
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        final_text = meta_line + "\n" + text
        
        return (final_text, meta)
    
    def _split_blocks_by_tokens(self, blocks: List[LayoutBlock], page_no: int, group_type: str) -> List[Tuple[str, Dict]]:
        """블록들을 토큰 수 기준으로 분할"""
        chunks = []
        current_blocks = []
        current_tokens = 0
        
        for block in blocks:
            block_tokens = self._count_tokens(block.text)
            
            if current_tokens + block_tokens <= self.target_tokens:
                current_blocks.append(block)
                current_tokens += block_tokens
            else:
                if current_blocks:
                    chunks.append(self._create_blocks_chunk(current_blocks, page_no, group_type))
                
                # 단일 블록이 너무 크면 문장 단위로 분할
                if block_tokens > self.target_tokens:
                    sub_chunks = self._split_large_block(block, page_no, group_type)
                    chunks.extend(sub_chunks)
                    current_blocks = []
                    current_tokens = 0
                else:
                    current_blocks = [block]
                    current_tokens = block_tokens
        
        if current_blocks:
            chunks.append(self._create_blocks_chunk(current_blocks, page_no, group_type))
        
        return chunks
    
    def _create_blocks_chunk(self, blocks: List[LayoutBlock], page_no: int, group_type: str) -> Tuple[str, Dict]:
        """블록들로부터 청크 생성"""
        text = '\n\n'.join(block.text for block in blocks)
        bbox = self._merge_bboxes([block.bbox for block in blocks])
        section = self._extract_section_from_blocks(blocks)
        
        meta = {
            "type": f"layout_{group_type}",
            "section": section,
            "page": page_no,
            "pages": [page_no],
            "bboxes": self._serialize_bbox(bbox),
            "block_count": len(blocks),
            "token_count": self._count_tokens(text)
        }
        
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        final_text = meta_line + "\n" + text
        
        return (final_text, meta)
    
    def _extract_section_from_blocks(self, blocks: List[LayoutBlock]) -> str:
        """블록들에서 섹션 정보 추출"""
        for block in blocks:
            section = self._extract_section_info({'blocks': [block]})
            if section:
                return section
        return ""
    
    def _split_large_block(self, block: LayoutBlock, page_no: int, group_type: str) -> List[Tuple[str, Dict]]:
        """큰 블록을 문장 단위로 분할"""
        chunks = []
        sentences = self._split_sentences(block.text)
        
        current_text = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= self.target_tokens:
                current_text += " " + sentence if current_text else sentence
                current_tokens += sentence_tokens
            else:
                if current_text:
                    chunks.append(self._create_text_chunk(current_text, page_no, group_type, block.bbox))
                current_text = sentence
                current_tokens = sentence_tokens
        
        if current_text:
            chunks.append(self._create_text_chunk(current_text, page_no, group_type, block.bbox))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리"""
        sentence_end = re.compile(r'[.!?]+\s*(?=[A-Z가-힣]|$)')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_text_chunk(self, text: str, page_no: int, group_type: str, bbox: BBox) -> Tuple[str, Dict]:
        """텍스트 청크 생성"""
        meta = {
            "type": f"layout_{group_type}_fragment",
            "section": "",
            "page": page_no,
            "pages": [page_no],
            "bboxes": self._serialize_bbox(bbox),
            "block_count": 1,
            "token_count": self._count_tokens(text)
        }
        
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        final_text = meta_line + "\n" + text
        
        return (final_text, meta)
    
    def _handle_cross_page_connections(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """페이지 간 연결 처리"""
        if len(chunks) < 2:
            return chunks
        
        connected = []
        i = 0
        
        while i < len(chunks):
            current_text, current_meta = chunks[i]
            
            # 다음 청크와 연결 가능한지 확인
            if (i + 1 < len(chunks) and 
                self._should_connect_chunks(current_meta, chunks[i + 1][1])):
                
                next_text, next_meta = chunks[i + 1]
                
                # 연결
                merged_text = self._merge_chunk_texts(current_text, next_text)
                merged_meta = self._merge_chunk_metas(current_meta, next_meta)
                
                connected.append((merged_text, merged_meta))
                i += 2
            else:
                connected.append((current_text, current_meta))
                i += 1
        
        return connected
    
    def _should_connect_chunks(self, meta1: Dict, meta2: Dict) -> bool:
        """두 청크 연결 여부 판단"""
        # 연속 페이지 확인
        if abs(meta1.get('page', 0) - meta2.get('page', 0)) != 1:
            return False
        
        # 같은 타입 확인
        if meta1.get('type') != meta2.get('type'):
            return False
        
        # 토큰 수 제한
        total_tokens = meta1.get('token_count', 0) + meta2.get('token_count', 0)
        if total_tokens > self.target_tokens * 1.8:
            return False
        
        # 레이아웃 연속성 확인 (bbox 위치)
        bbox1 = meta1.get('bboxes', {})
        bbox2 = meta2.get('bboxes', {})
        
        if bbox1 and bbox2:
            # Y 좌표 연속성 (페이지 하단 → 다음 페이지 상단)
            if bbox1.get('y1', 0) > 700 and bbox2.get('y0', 0) < 200:  # 대략적인 페이지 경계
                return True
        
        return False
    
    def _merge_chunk_texts(self, text1: str, text2: str) -> str:
        """두 청크 텍스트 병합"""
        # META 라인 제거
        clean_text1 = self._strip_meta_line(text1)
        clean_text2 = self._strip_meta_line(text2)
        
        return clean_text1 + "\n\n" + clean_text2
    
    def _merge_chunk_metas(self, meta1: Dict, meta2: Dict) -> Dict:
        """두 청크 메타데이터 병합"""
        merged = meta1.copy()
        
        # 페이지 범위 확장
        pages1 = meta1.get('pages', [meta1.get('page', 0)])
        pages2 = meta2.get('pages', [meta2.get('page', 0)])
        merged['pages'] = sorted(set(pages1 + pages2))
        
        # 기타 필드 업데이트
        merged['block_count'] = meta1.get('block_count', 0) + meta2.get('block_count', 0)
        merged['token_count'] = meta1.get('token_count', 0) + meta2.get('token_count', 0)
        
        # 새로운 META 라인 생성
        return merged
    
    def _strip_meta_line(self, text: str) -> str:
        """META 라인 제거"""
        if text.startswith("META:"):
            nl_pos = text.find("\n")
            return text[nl_pos + 1:] if nl_pos != -1 else ""
        return text
    
    def _fallback_chunking(self, text: str, page_no: int) -> List[Tuple[str, Dict]]:
        """레이아웃 정보 없을 때의 폴백 청킹"""
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            para_tokens = self._count_tokens(para)
            
            if current_tokens + para_tokens <= self.target_tokens:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(self._create_fallback_chunk(current_chunk, page_no))
                current_chunk = para
                current_tokens = para_tokens
        
        if current_chunk:
            chunks.append(self._create_fallback_chunk(current_chunk, page_no))
        
        return chunks
    
    def _create_fallback_chunk(self, text: str, page_no: int) -> Tuple[str, Dict]:
        """폴백 청크 생성"""
        meta = {
            "type": "layout_fallback",
            "section": "",
            "page": page_no,
            "pages": [page_no],
            "bboxes": {},
            "block_count": 1,
            "token_count": self._count_tokens(text)
        }
        
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        final_text = meta_line + "\n" + text
        
        return (final_text, meta)
    
    def _finalize_chunks(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """최종 청크 정리"""
        finalized = []
        
        for text, meta in chunks:
            # 최소 토큰 수 확인
            if meta.get('token_count', 0) < self.min_chunk_tokens:
                continue
            
            # 텍스트 정리
            clean_text = self._clean_chunk_text(text)
            if not clean_text.strip():
                continue
            
            # 메타데이터에 새로운 META 라인 적용
            meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
            final_text = meta_line + "\n" + self._strip_meta_line(clean_text)
            
            finalized.append((final_text, meta))
        
        return finalized
    
    def _clean_chunk_text(self, text: str) -> str:
        """청크 텍스트 정리"""
        # 이상한 라벨 제거
        text = re.sub(r'\b인접행\s*묶음\b', '', text)
        text = re.sub(r'\b[가-힣]*\s*묶음\b', '', text)
        
        # 과도한 공백 정리
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        if not text:
            return 0
        try:
            return len(self.encoder(text))
        except:
            return len(text.split()) * 1.3


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