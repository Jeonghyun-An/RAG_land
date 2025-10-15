"""
스마트 OCR 라우팅 로직
app/services/smart_ocr_router.py
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PageQuality:
    """페이지 품질 평가 결과"""
    page_no: int
    text_density: float  # 0-1
    layout_quality: float  # 0-1
    has_tables: bool
    has_images: bool
    readable_ratio: float  # 0-1
    needs_ocr: bool
    confidence: float

class SmartOCRRouter:
    """레이아웃 품질 기반 지능형 OCR 라우터"""
    
    def __init__(self):
        self.min_text_density = 0.001  # 페이지당 최소 텍스트 밀도
        self.min_readable_ratio = 0.5  # 최소 읽기 가능한 문자 비율
        self.table_ocr_threshold = 0.3  # 표 비율이 이 이상이면 OCR 고려
    
    def should_use_ocr(
        self,
        pages: List[Tuple[int, str]],
        layout_blocks: Optional[Dict[int, List[Dict]]] = None
    ) -> Tuple[bool, List[int]]:
        """
        OCR 사용 여부 및 대상 페이지 결정
        
        Returns:
            (전체_OCR_필요, OCR_필요_페이지_리스트)
        """
        if not pages:
            return False, []
        
        page_qualities = []
        for page_no, text in pages:
            quality = self._evaluate_page_quality(
                page_no, text,
                layout_blocks.get(page_no) if layout_blocks else None
            )
            page_qualities.append(quality)
        
        # 전체 OCR 필요 판단
        needs_full_ocr = self._needs_full_ocr(page_qualities)
        
        # 부분 OCR 필요 페이지
        partial_ocr_pages = [
            q.page_no for q in page_qualities 
            if q.needs_ocr and not needs_full_ocr
        ]
        
        return needs_full_ocr, partial_ocr_pages
    
    def _evaluate_page_quality(
        self,
        page_no: int,
        text: str,
        layout_blocks: Optional[List[Dict]] = None
    ) -> PageQuality:
        """개별 페이지 품질 평가"""
        
        # 1. 텍스트 밀도 계산
        text_density = self._calculate_text_density(text)
        
        # 2. 레이아웃 품질 평가
        layout_quality = self._evaluate_layout_quality(text, layout_blocks)
        
        # 3. 특수 요소 감지
        has_tables = self._has_table_elements(text, layout_blocks)
        has_images = self._has_image_elements(layout_blocks)
        
        # 4. 읽기 가능한 문자 비율
        readable_ratio = self._calculate_readable_ratio(text)
        
        # 5. OCR 필요 여부 종합 판단
        needs_ocr = self._judge_ocr_need(
            text_density, layout_quality, has_tables, 
            has_images, readable_ratio
        )
        
        # 6. 신뢰도 계산
        confidence = self._calculate_confidence(
            text_density, layout_quality, readable_ratio
        )
        
        return PageQuality(
            page_no=page_no,
            text_density=text_density,
            layout_quality=layout_quality,
            has_tables=has_tables,
            has_images=has_images,
            readable_ratio=readable_ratio,
            needs_ocr=needs_ocr,
            confidence=confidence
        )
    
    def _calculate_text_density(self, text: str) -> float:
        """텍스트 밀도 계산 (0-1)"""
        if not text:
            return 0.0
        
        # 공백 제외 문자 수
        non_space_chars = len(text.replace(' ', '').replace('\n', ''))
        
        # 의미있는 텍스트 추정 (평균 페이지당 1000자 기준)
        expected_chars = 1000
        density = min(non_space_chars / expected_chars, 1.0)
        
        return density
    
    def _evaluate_layout_quality(
        self, 
        text: str, 
        layout_blocks: Optional[List[Dict]]
    ) -> float:
        """레이아웃 품질 평가 (0-1)"""
        if not layout_blocks:
            # 레이아웃 정보 없으면 텍스트 기반 추정
            return self._estimate_quality_from_text(text)
        
        quality_score = 0.0
        total_checks = 0
        
        # 1. bbox 정보 완전성
        blocks_with_bbox = sum(
            1 for b in layout_blocks 
            if b.get('bbox') and len(b['bbox']) == 4
        )
        if layout_blocks:
            quality_score += blocks_with_bbox / len(layout_blocks)
            total_checks += 1
        
        # 2. 텍스트-bbox 일치도
        blocks_with_text = sum(
            1 for b in layout_blocks 
            if b.get('text') and b['text'].strip()
        )
        if layout_blocks:
            quality_score += blocks_with_text / len(layout_blocks)
            total_checks += 1
        
        # 3. 정렬 품질 (y좌표 기준)
        if len(layout_blocks) >= 2:
            y_coords = sorted([
                b.get('bbox', [0,0,0,0])[1] 
                for b in layout_blocks
            ])
            # 줄 간격의 일관성
            gaps = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                variance = sum((g - avg_gap)**2 for g in gaps) / len(gaps)
                consistency = max(0, 1 - (variance / (avg_gap**2 + 1)))
                quality_score += consistency
                total_checks += 1
        
        return quality_score / max(total_checks, 1)
    
    def _estimate_quality_from_text(self, text: str) -> float:
        """텍스트만으로 품질 추정"""
        if not text:
            return 0.0
        
        lines = text.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        if not non_empty_lines:
            return 0.0
        
        # 깨진 문자 패턴
        broken_pattern_count = sum(
            1 for line in non_empty_lines
            if any(c in line for c in ['�', '□', '◇', '○'])
        )
        
        # 의미없는 짧은 줄
        too_short_count = sum(
            1 for line in non_empty_lines
            if len(line.strip()) < 3
        )
        
        quality = 1.0 - (broken_pattern_count + too_short_count) / len(non_empty_lines)
        return max(0.0, quality)
    
    def _has_table_elements(
        self, 
        text: str, 
        layout_blocks: Optional[List[Dict]]
    ) -> bool:
        """표 요소 존재 여부"""
        # 텍스트 패턴 검사
        table_indicators = [
            r'[\|\+\-]{3,}',  # ASCII 테이블
            r'[┌┐└┘├┤┬┴┼─│]{3,}',  # 박스 문자
            r'\t.*\t',  # 탭으로 구분된 열
        ]
        
        import re
        for pattern in table_indicators:
            if re.search(pattern, text):
                return True
        
        # 레이아웃 기반 검사
        if layout_blocks and len(layout_blocks) >= 4:
            # 격자 패턴 검사
            x_coords = set()
            y_coords = set()
            for block in layout_blocks:
                bbox = block.get('bbox', [0,0,0,0])
                x_coords.add(round(bbox[0], 1))
                y_coords.add(round(bbox[1], 1))
            
            # X, Y 좌표가 각각 3개 이상의 고유값 = 격자 구조
            if len(x_coords) >= 3 and len(y_coords) >= 3:
                return True
        
        return False
    
    def _has_image_elements(self, layout_blocks: Optional[List[Dict]]) -> bool:
        """이미지 요소 존재 여부"""
        if not layout_blocks:
            return False
        
        # 'image' 타입 블록 또는 텍스트 없는 큰 블록
        for block in layout_blocks:
            if block.get('type') == 'image':
                return True
            
            # 텍스트 없고 bbox가 큰 경우
            bbox = block.get('bbox', [0,0,0,0])
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            text = block.get('text', '').strip()
            
            if not text and width * height > 10000:  # 100x100pt 이상
                return True
        
        return False
    
    def _calculate_readable_ratio(self, text: str) -> float:
        """읽기 가능한 문자 비율"""
        if not text:
            return 0.0
        
        # 의미있는 문자 (한글, 영문, 숫자, 일반 기호)
        import re
        readable = re.findall(r'[가-힣a-zA-Z0-9\s,.\-:()]', text)
        total_chars = len(text)
        
        if total_chars == 0:
            return 0.0
        
        return len(readable) / total_chars
    
    def _judge_ocr_need(
        self,
        text_density: float,
        layout_quality: float,
        has_tables: bool,
        has_images: bool,
        readable_ratio: float
    ) -> bool:
        """OCR 필요 여부 종합 판단"""
        
        # 케이스 1: 텍스트가 거의 없음
        if text_density < self.min_text_density:
            return True
        
        # 케이스 2: 읽을 수 없는 문자가 많음
        if readable_ratio < self.min_readable_ratio:
            return True
        
        # 케이스 3: 표가 있고 레이아웃 품질이 낮음
        if has_tables and layout_quality < 0.6:
            return True
        
        # 케이스 4: 이미지가 있고 텍스트 밀도가 낮음
        if has_images and text_density < 0.3:
            return True
        
        # 케이스 5: 전반적인 품질이 매우 낮음
        overall_quality = (
            text_density * 0.3 +
            layout_quality * 0.3 +
            readable_ratio * 0.4
        )
        if overall_quality < 0.4:
            return True
        
        return False
    
    def _calculate_confidence(
        self,
        text_density: float,
        layout_quality: float,
        readable_ratio: float
    ) -> float:
        """판단 신뢰도 계산"""
        # 가중 평균
        confidence = (
            text_density * 0.3 +
            layout_quality * 0.3 +
            readable_ratio * 0.4
        )
        return min(max(confidence, 0.0), 1.0)
    
    def _needs_full_ocr(self, page_qualities: List[PageQuality]) -> bool:
        """전체 문서 OCR 필요 여부"""
        if not page_qualities:
            return False
        
        # 50% 이상 페이지가 OCR 필요
        ocr_needed_count = sum(1 for q in page_qualities if q.needs_ocr)
        ratio = ocr_needed_count / len(page_qualities)
        
        return ratio >= 0.5
    
    def get_quality_report(
        self,
        pages: List[Tuple[int, str]],
        layout_blocks: Optional[Dict[int, List[Dict]]] = None
    ) -> Dict:
        """페이지 품질 리포트 생성 (디버깅용)"""
        qualities = []
        for page_no, text in pages:
            quality = self._evaluate_page_quality(
                page_no, text,
                layout_blocks.get(page_no) if layout_blocks else None
            )
            qualities.append({
                'page': quality.page_no,
                'text_density': f"{quality.text_density:.2f}",
                'layout_quality': f"{quality.layout_quality:.2f}",
                'readable_ratio': f"{quality.readable_ratio:.2f}",
                'has_tables': quality.has_tables,
                'has_images': quality.has_images,
                'needs_ocr': quality.needs_ocr,
                'confidence': f"{quality.confidence:.2f}"
            })
        
        needs_full, partial_pages = self.should_use_ocr(pages, layout_blocks)
        
        return {
            'total_pages': len(pages),
            'needs_full_ocr': needs_full,
            'partial_ocr_pages': partial_pages,
            'page_details': qualities
        }