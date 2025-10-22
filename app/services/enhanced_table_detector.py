"""
고도화된 표 감지 및 처리 모듈
app/services/enhanced_table_detector.py
"""
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TableRegion:
    """표 영역 정보"""
    page: int
    start_line: int
    end_line: int
    bbox: Optional[Tuple[float, float, float, float]]  # (x0, y0, x1, y1)
    content: str
    confidence: float  # 0-1
    table_type: str  # 'bordered', 'borderless', 'complex'

class EnhancedTableDetector:
    """레이아웃 정보를 활용한 향상된 표 감지기"""
    
    def __init__(self):
        # 기존 패턴
        self.border_patterns = [
            re.compile(r'[\|\+\-]{3,}'),  # ASCII 테이블
            re.compile(r'[┌┐└┘├┤┬┴┼─│]{3,}'),  # 유니코드 박스
            re.compile(r'═+|║+'),  # 이중선
        ]
        
        # 표 헤더 패턴 (한국어 문서 특화)
        self.header_keywords = [
            '구분', '항목', '내용', '비고', '번호', '명칭', 
            '수량', '금액', '단위', '기준', '조건', '결과'
        ]
        self.row_tolerance = 10
        
    def detect_tables(
        self, 
        text: str, 
        page_no: int,
        layout_blocks: Optional[List[Dict]] = None
    ) -> List[TableRegion]:
        """
        다중 전략으로 표 감지
        1. 레이아웃 bbox 기반 (최우선)
        2. 텍스트 패턴 기반
        3. 구조 분석 기반
        """
        tables = []
        
        # 전략 1: 레이아웃 정보 활용 (가장 정확)
        if layout_blocks:
            layout_tables = self._detect_from_layout(text, page_no, layout_blocks)
            tables.extend(layout_tables)
        
        # 전략 2: 텍스트 패턴 기반
        pattern_tables = self._detect_from_patterns(text, page_no)
        tables.extend(pattern_tables)
        
        # 전략 3: 구조 분석 (탭/공백 정렬)
        structure_tables = self._detect_from_structure(text, page_no)
        tables.extend(structure_tables)
        
        # 중복 제거 및 병합
        return self._merge_overlapping_tables(tables)
    
    def _detect_from_layout(
        self, 
        text: str, 
        page_no: int, 
        layout_blocks: List[Dict]
    ) -> List[TableRegion]:
        """레이아웃 bbox 정보로 표 감지"""
        tables = []
        lines = text.split('\n')
        
        # bbox가 수평/수직 정렬된 블록들을 그룹화
        aligned_groups = self._find_aligned_blocks(layout_blocks)
        
        for group in aligned_groups:
            if self._is_table_like_group(group):
                # 해당 bbox 범위의 텍스트 추출
                table_text = self._extract_text_from_bbox(
                    lines, group['bbox'], group['blocks']
                )
                
                if table_text.strip():
                    tables.append(TableRegion(
                        page=page_no,
                        start_line=group['start_line'],
                        end_line=group['end_line'],
                        bbox=group['bbox'],
                        content=table_text,
                        confidence=0.9,
                        table_type='layout_detected'
                    ))
        
        return tables
    
    def _find_aligned_blocks(self, blocks: List[Dict]) -> List[List[Dict]]:
        """
        Y좌표가 비슷한 블록들을 그룹화 (같은 행)
        [수정] bbox dict 형식 대응
        """
        if not blocks:
            return []

        # [수정] bbox 형식 통일 함수
        def get_y_coord(block):
            bbox = block.get('bbox', {})
            if isinstance(bbox, dict):
                return bbox.get('y0', 0)  # dict 형식
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
                return bbox[1]  # list 형식 [x0, y0, x1, y1]
            else:
                return 0

        # Y 좌표 기준 정렬
        sorted_blocks = sorted(blocks, key=get_y_coord)

        groups = []
        current_group = [sorted_blocks[0]]
        current_y = get_y_coord(sorted_blocks[0])

        for block in sorted_blocks[1:]:
            block_y = get_y_coord(block)

            # Y 좌표 차이가 threshold 이내면 같은 행
            if abs(block_y - current_y) < self.row_tolerance:
                current_group.append(block)
            else:
                if len(current_group) >= 2:  # 최소 2개 이상
                    groups.append(current_group)
                current_group = [block]
                current_y = block_y

        # 마지막 그룹
        if len(current_group) >= 2:
            groups.append(current_group)

        return groups
    
    def _is_table_like_group(self, group: Dict) -> bool:
        """그룹이 표 형태인지 판단"""
        blocks = group['blocks']
        
        # 최소 3개 블록 (제목 + 2개 셀)
        if len(blocks) < 3:
            return False
        
        # X좌표가 규칙적으로 정렬되어 있는지 확인
        x_coords = [b.get('bbox', [0,0,0,0])[0] for b in blocks]
        x_coords.sort()
        
        # 간격이 일정한지 확인
        gaps = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
        if len(set(gaps)) <= 2:  # 최대 2가지 간격 패턴
            return True
        
        # 텍스트가 헤더 키워드를 포함하는지
        text_content = ' '.join(b.get('text', '') for b in blocks)
        if any(kw in text_content for kw in self.header_keywords):
            return True
        
        return False
    
    def _detect_from_patterns(self, text: str, page_no: int) -> List[TableRegion]:
        """텍스트 패턴으로 표 감지 (기존 방식 개선)"""
        tables = []
        lines = text.split('\n')
        
        in_table = False
        table_start = -1
        table_lines = []
        border_count = 0
        
        for i, line in enumerate(lines):
            # 테두리 패턴 검사
            has_border = any(p.search(line) for p in self.border_patterns)
            
            if has_border:
                border_count += 1
                if not in_table:
                    in_table = True
                    table_start = i
                table_lines.append(line)
            elif in_table:
                # 공백 줄이 아니면 표 내용으로 간주
                if line.strip():
                    # 탭이나 여러 공백으로 구분된 경우도 표로 간주
                    if '\t' in line or '  ' in line:
                        table_lines.append(line)
                    elif border_count >= 2:  # 테두리가 충분하면 포함
                        table_lines.append(line)
                    else:
                        # 표 종료
                        if len(table_lines) >= 3:
                            tables.append(self._create_table_region(
                                page_no, table_start, i-1, '\n'.join(table_lines),
                                confidence=0.7, table_type='bordered'
                            ))
                        in_table = False
                        table_lines = []
                        border_count = 0
                else:
                    # 연속 공백 2줄이면 표 종료
                    if i > 0 and not lines[i-1].strip():
                        if len(table_lines) >= 3:
                            tables.append(self._create_table_region(
                                page_no, table_start, i-1, '\n'.join(table_lines),
                                confidence=0.7, table_type='bordered'
                            ))
                        in_table = False
                        table_lines = []
                        border_count = 0
        
        # 마지막 표 처리
        if in_table and len(table_lines) >= 3:
            tables.append(self._create_table_region(
                page_no, table_start, len(lines)-1, '\n'.join(table_lines),
                confidence=0.7, table_type='bordered'
            ))
        
        return tables
    
    def _detect_from_structure(self, text: str, page_no: int) -> List[TableRegion]:
        """구조 분석으로 테두리 없는 표 감지"""
        tables = []
        lines = text.split('\n')
        
        # 탭/공백으로 정렬된 영역 찾기
        aligned_start = -1
        aligned_lines = []
        prev_columns = 0
        
        for i, line in enumerate(lines):
            # 탭이나 연속 공백(2개 이상)으로 분리된 컬럼 수 계산
            if '\t' in line:
                columns = len(line.split('\t'))
            elif '  ' in line:  # 공백 2개 이상
                columns = len(re.split(r'\s{2,}', line.strip()))
            else:
                columns = 0
            
            # 최소 2컬럼 이상
            if columns >= 2:
                if aligned_start == -1:
                    aligned_start = i
                    prev_columns = columns
                    aligned_lines.append(line)
                elif abs(columns - prev_columns) <= 1:  # 컬럼 수 유사
                    aligned_lines.append(line)
                else:
                    # 표 종료
                    if len(aligned_lines) >= 3:
                        tables.append(self._create_table_region(
                            page_no, aligned_start, i-1, '\n'.join(aligned_lines),
                            confidence=0.6, table_type='borderless'
                        ))
                    aligned_start = i
                    aligned_lines = [line]
                    prev_columns = columns
            else:
                # 비정렬 줄
                if aligned_start != -1 and len(aligned_lines) >= 3:
                    tables.append(self._create_table_region(
                        page_no, aligned_start, i-1, '\n'.join(aligned_lines),
                        confidence=0.6, table_type='borderless'
                    ))
                aligned_start = -1
                aligned_lines = []
                prev_columns = 0
        
        # 마지막 표 처리
        if aligned_start != -1 and len(aligned_lines) >= 3:
            tables.append(self._create_table_region(
                page_no, aligned_start, len(lines)-1, '\n'.join(aligned_lines),
                confidence=0.6, table_type='borderless'
            ))
        
        return tables
    
    def _merge_overlapping_tables(self, tables: List[TableRegion]) -> List[TableRegion]:
        """중복/겹치는 표 병합"""
        if not tables:
            return []
        
        # confidence와 라인 범위로 정렬
        tables.sort(key=lambda t: (t.page, t.start_line, -t.confidence))
        
        merged = []
        current = tables[0]
        
        for next_table in tables[1:]:
            # 같은 페이지이고 겹치는 경우
            if (current.page == next_table.page and
                current.start_line <= next_table.start_line <= current.end_line):
                
                # confidence가 더 높은 것 선택
                if next_table.confidence > current.confidence:
                    current = next_table
                # 범위 확장
                current.end_line = max(current.end_line, next_table.end_line)
            else:
                merged.append(current)
                current = next_table
        
        merged.append(current)
        return merged
    
    def _create_table_region(
        self, page: int, start: int, end: int, 
        content: str, confidence: float, table_type: str
    ) -> TableRegion:
        """TableRegion 객체 생성"""
        return TableRegion(
            page=page,
            start_line=start,
            end_line=end,
            bbox=None,
            content=content,
            confidence=confidence,
            table_type=table_type
        )
    
    def _create_group(self, blocks: List[Dict]) -> Dict:
        """블록 그룹 정보 생성"""
        bboxes = [b.get('bbox', [0,0,0,0]) for b in blocks]
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        
        return {
            'blocks': blocks,
            'bbox': (x0, y0, x1, y1),
            'start_line': 0,  # 실제로는 계산 필요
            'end_line': 0
        }
    
    def _extract_text_from_bbox(
        self, 
        lines: List[str], 
        bbox: Tuple, 
        blocks: List[Dict]
    ) -> str:
        """
        bbox 범위의 텍스트 추출
        
        Args:
            lines: 전체 텍스트의 라인 리스트
            bbox: (x0, y0, x1, y1) 좌표
            blocks: bbox에 해당하는 블록들
        
        Returns:
            추출된 텍스트
        """
        # 방법 1: 블록에서 직접 텍스트 추출 (간단하고 정확)
        return '\n'.join(b.get('text', '') for b in blocks)
        
        # 방법 2: bbox 좌표와 라인 매칭 (더 정교하지만 복잡)
        # 이 경우 layout_blocks에 line_index 정보가 필요함
        # 현재는 방법 1 사용 권장