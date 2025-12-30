"""
고도화된 표 감지 및 처리 모듈 (수정 버전)
app/services/enhanced_table_detector.py

수정 사항:
1. _find_aligned_blocks() 반환값 수정 - List[Dict]로 통일
2. bbox 형식 정규화 함수 추가
3. _create_table_group() 실제 사용
4. 표 감지 로직 개선
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
    table_type: str  # 'bordered', 'borderless', 'complex', 'layout_detected'
    
    # 하위 호환성을 위해 추가
    @property
    def text(self) -> str:
        return self.content

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
        
        # 설정값
        self.row_tolerance = 10  # 같은 행으로 판단하는 y좌표 차이
        self.row_gap_threshold = 50  # 다른 표로 판단하는 행 간격
        
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
        
        print(f"[TABLE-DETECTOR] Page {page_no}: detecting tables...")
        
        # 전략 1: 레이아웃 정보 활용 (가장 정확)
        if layout_blocks:
            print(f"[TABLE-DETECTOR] Using layout blocks ({len(layout_blocks)} blocks)")
            layout_tables = self._detect_from_layout(text, page_no, layout_blocks)
            tables.extend(layout_tables)
            print(f"[TABLE-DETECTOR] Found {len(layout_tables)} tables from layout")
        
        # 전략 2: 텍스트 패턴 기반
        pattern_tables = self._detect_from_patterns(text, page_no)
        tables.extend(pattern_tables)
        print(f"[TABLE-DETECTOR] Found {len(pattern_tables)} tables from patterns")
        
        # 전략 3: 구조 분석 (탭/공백 정렬)
        structure_tables = self._detect_from_structure(text, page_no)
        tables.extend(structure_tables)
        print(f"[TABLE-DETECTOR] Found {len(structure_tables)} tables from structure")
        
        # 중복 제거 및 병합
        merged_tables = self._merge_overlapping_tables(tables)
        print(f"[TABLE-DETECTOR] Page {page_no}: {len(merged_tables)} tables after merging")
        
        return merged_tables
    
    # ==================== 유틸리티 함수 ====================
    
    def _normalize_bbox(self, bbox) -> List[float]:
        """bbox를 [x0, y0, x1, y1] 형식으로 정규화"""
        if isinstance(bbox, dict):
            return [
                bbox.get('x0', 0), 
                bbox.get('y0', 0), 
                bbox.get('x1', 0), 
                bbox.get('y1', 0)
            ]
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return list(bbox[:4])
        else:
            return [0, 0, 0, 0]
    
    def _get_y_coord(self, block: Dict) -> float:
        """블록의 y좌표 추출 (형식 무관)"""
        bbox = block.get('bbox', {})
        if isinstance(bbox, dict):
            return bbox.get('y0', 0)
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
            return bbox[1]
        else:
            return 0
    
    def _get_x_coord(self, block: Dict) -> float:
        """블록의 x좌표 추출 (형식 무관)"""
        bbox = block.get('bbox', {})
        if isinstance(bbox, dict):
            return bbox.get('x0', 0)
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 1:
            return bbox[0]
        else:
            return 0
    
    # ==================== 레이아웃 기반 감지 ====================
    
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
        
        print(f"[TABLE-DETECTOR] Found {len(aligned_groups)} aligned groups")
        
        for i, group in enumerate(aligned_groups):
            if self._is_table_like_group(group):
                # 해당 bbox 범위의 텍스트 추출
                table_text = self._extract_text_from_blocks(group['blocks'])
                
                if table_text.strip():
                    tables.append(TableRegion(
                        page=page_no,
                        start_line=group.get('start_line', 0),
                        end_line=group.get('end_line', 0),
                        bbox=group['bbox'],
                        content=table_text,
                        confidence=0.9,
                        table_type='layout_detected'
                    ))
                    print(f"[TABLE-DETECTOR] Table {i+1}: {group['row_count']} rows, "
                          f"{len(group['blocks'])} blocks")
        
        return tables
    
    def _find_aligned_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        Y좌표가 비슷한 블록들을 그룹화하여 표 후보 생성
        
        Returns:
            List[Dict]: 각 그룹은 {'blocks': [...], 'bbox': (...), 'row_count': N}
        """
        if not blocks:
            return []

        # Y 좌표 기준 정렬
        sorted_blocks = sorted(blocks, key=self._get_y_coord)

        # 1단계: Y좌표가 비슷한 블록들을 행으로 그룹화
        row_groups = []
        current_row = [sorted_blocks[0]]
        current_y = self._get_y_coord(sorted_blocks[0])

        for block in sorted_blocks[1:]:
            block_y = self._get_y_coord(block)

            # Y 좌표 차이가 threshold 이내면 같은 행
            if abs(block_y - current_y) < self.row_tolerance:
                current_row.append(block)
            else:
                if len(current_row) >= 2:  # 최소 2개 블록 (2개 셀)
                    row_groups.append(current_row)
                current_row = [block]
                current_y = block_y

        # 마지막 그룹
        if len(current_row) >= 2:
            row_groups.append(current_row)

        # 2단계: 연속된 행들을 하나의 표로 그룹화
        table_groups = []
        
        if not row_groups:
            return []
        
        current_table_rows = [row_groups[0]]
        prev_y = self._get_y_coord(row_groups[0][-1])
        
        for i in range(1, len(row_groups)):
            curr_row = row_groups[i]
            curr_y = self._get_y_coord(curr_row[0])
            
            # 행 간격이 너무 크면 다른 표로 판단
            if abs(curr_y - prev_y) > self.row_gap_threshold:
                # 이전 표 완성
                if len(current_table_rows) >= 2:  # 최소 2개 행
                    table_groups.append(self._create_table_group(current_table_rows))
                current_table_rows = [curr_row]
            else:
                current_table_rows.append(curr_row)
            
            prev_y = self._get_y_coord(curr_row[-1])
        
        # 마지막 표
        if len(current_table_rows) >= 2:
            table_groups.append(self._create_table_group(current_table_rows))
        
        return table_groups
    
    def _create_table_group(self, rows: List[List[Dict]]) -> Dict:
        """
        여러 행을 하나의 표 그룹으로 변환
        
        Args:
            rows: List[List[Dict]] - 각 행은 블록 리스트
        
        Returns:
            Dict: {'blocks': [...], 'bbox': (x0, y0, x1, y1), 'row_count': N}
        """
        all_blocks = []
        for row in rows:
            all_blocks.extend(row)
        
        # 전체 bbox 계산
        bboxes = [self._normalize_bbox(b.get('bbox', [0,0,0,0])) for b in all_blocks]
        
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        
        return {
            'blocks': all_blocks,
            'bbox': (x0, y0, x1, y1),
            'start_line': 0,  # TODO: 실제 라인 번호 계산
            'end_line': len(rows) - 1,
            'row_count': len(rows),
            'col_count': len(rows[0]) if rows else 0,
        }
    
    def _is_table_like_group(self, group: Dict) -> bool:
        """그룹이 표 형태인지 판단"""
        blocks = group['blocks']
        row_count = group.get('row_count', 0)
        
        # 최소 2개 행
        if row_count < 2:
            return False
        
        # 최소 4개 블록 (2x2)
        if len(blocks) < 4:
            return False
        
        # X좌표가 규칙적으로 정렬되어 있는지 확인
        x_coords = sorted(set(self._get_x_coord(b) for b in blocks))
        
        # 최소 2개 열
        if len(x_coords) < 2:
            return False
        
        # 간격이 일정한지 확인
        if len(x_coords) >= 3:
            gaps = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
            # 간격의 표준편차가 작으면 정렬된 것
            avg_gap = sum(gaps) / len(gaps)
            variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
            if variance > avg_gap * 0.5:  # 변동이 크면 비정렬
                # 그래도 헤더 키워드 확인
                pass
        
        # 텍스트가 헤더 키워드를 포함하는지
        text_content = ' '.join(b.get('text', '') for b in blocks)
        if any(kw in text_content for kw in self.header_keywords):
            return True
        
        # 블록 밀도 확인 (표는 셀이 규칙적으로 채워짐)
        expected_cells = row_count * group.get('col_count', 2)
        actual_cells = len(blocks)
        density = actual_cells / expected_cells if expected_cells > 0 else 0
        
        # 70% 이상 채워져 있으면 표로 판단
        return density >= 0.7
    
    def _extract_text_from_blocks(self, blocks: List[Dict]) -> str:
        """블록들에서 텍스트 추출 (y좌표 기준 정렬)"""
        # y좌표 기준 정렬
        sorted_blocks = sorted(blocks, key=self._get_y_coord)
        
        # 같은 행의 블록들을 그룹화
        rows = []
        current_row = [sorted_blocks[0]]
        current_y = self._get_y_coord(sorted_blocks[0])
        
        for block in sorted_blocks[1:]:
            block_y = self._get_y_coord(block)
            
            if abs(block_y - current_y) < self.row_tolerance:
                current_row.append(block)
            else:
                # x좌표 기준 정렬 후 텍스트 결합
                current_row.sort(key=self._get_x_coord)
                row_text = ' '.join(b.get('text', '').strip() for b in current_row if b.get('text', '').strip())
                if row_text:
                    rows.append(row_text)
                current_row = [block]
                current_y = block_y
        
        # 마지막 행
        if current_row:
            current_row.sort(key=self._get_x_coord)
            row_text = ' '.join(b.get('text', '').strip() for b in current_row if b.get('text', '').strip())
            if row_text:
                rows.append(row_text)
        
        return '\n'.join(rows)
    
    # ==================== 텍스트 패턴 기반 감지 ====================
    
    def _detect_from_patterns(self, text: str, page_no: int) -> List[TableRegion]:
        """텍스트 패턴으로 표 감지 (테두리 기반)"""
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
    
    # ==================== 구조 분석 기반 감지 ====================
    
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
    
    # ==================== 중복 제거 ====================
    
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
                else:
                    # 범위 확장
                    current = TableRegion(
                        page=current.page,
                        start_line=current.start_line,
                        end_line=max(current.end_line, next_table.end_line),
                        bbox=current.bbox,
                        content=current.content,
                        confidence=current.confidence,
                        table_type=current.table_type
                    )
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