# app/services/smart_ocr_router.py
"""
스마트 OCR 라우팅 로직 - 고도화 버전
원자력 안전 문서 특화 OCR 품질 평가 및 라우팅

개선 사항:
1. 원자력 규제 문서 특성 반영 (조항, 표, 기술 용어)
2. 다층 평가 시스템 (텍스트/레이아웃/의미론적 품질)
3. 페이지 타입별 OCR 전략
4. 동적 임계값 조정
5. 상세한 품질 리포트
"""
import os
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PageType(Enum):
    """페이지 타입 분류"""
    LEGAL_TEXT = "legal"  # 법률 조항
    TECHNICAL_MANUAL = "manual"  # 기술 매뉴얼
    TABLE_HEAVY = "table"  # 표 중심
    IMAGE_HEAVY = "image"  # 이미지 중심
    MIXED = "mixed"  # 혼합
    UNKNOWN = "unknown"


@dataclass
class PageQuality:
    """페이지 품질 평가 결과"""
    page_no: int
    page_type: PageType
    text_density: float  # 0-1
    layout_quality: float  # 0-1
    semantic_quality: float  # 0-1 (의미론적 품질)
    has_tables: bool
    has_images: bool
    has_legal_markers: bool  # 법률 조항 마커
    has_technical_terms: bool  # 기술 용어
    readable_ratio: float  # 0-1
    cid_ratio: float  # CID 코드 비율
    rotation_detected: bool  # 회전 감지
    needs_ocr: bool
    confidence: float
    ocr_priority: int  # 1(highest) ~ 5(lowest)


class SmartOCRRouter:
    """레이아웃 품질 기반 지능형 OCR 라우터 - 고도화"""
    
    def __init__(self):
        # ========== 기본 임계값 (환경변수로 오버라이드 가능) ==========
        self.min_text_density = float(os.getenv("OCR_MIN_TEXT_DENSITY", "0.01"))
        self.min_readable_ratio = float(os.getenv("OCR_MIN_READABLE_RATIO", "0.3"))
        self.min_semantic_quality = float(os.getenv("OCR_MIN_SEMANTIC_QUALITY", "0.4"))
        self.cid_threshold = float(os.getenv("OCR_CID_THRESHOLD", "0.15"))
        self.table_ocr_threshold = float(os.getenv("OCR_TABLE_THRESHOLD", "0.25"))
        
        # ========== 원자력 문서 특화 ==========
        # 법률 조항 패턴
        self.legal_patterns = [
            r'제\s*\d+\s*조',  # 제1조, 제 2 조
            r'제\s*\d+\s*항',  # 제1항
            r'제\s*\d+\s*호',  # 제1호
            r'별표\s*\d+',     # 별표 1
            r'별지\s*제\s*\d+\s*호',  # 별지 제1호
            r'附則',           # 부칙
            r'Article\s+\d+',  # 영문 조항
        ]
        
        # 기술 용어 (원자력 안전 분야)
        self.technical_keywords = [
            '방사선', '방사능', '원자로', '핵연료', '선량', 'Sv', 'Gy', 'Bq',
            '안전성', '허가', '검사', '규제', '기준', '제한', '측정',
            'IAEA', 'KINAC', '원안위', '원자력', '핵물질', '보안', '보호',
            'reactor', 'radiation', 'dose', 'nuclear', 'safety', 'regulation',
            'license', 'inspection', 'safeguards', 'security',
        ]
        
        # ========== 페이지 타입별 OCR 전략 ==========
        self.type_ocr_strategies = {
            PageType.LEGAL_TEXT: {
                'min_quality': 0.7,  # 법률 문서는 높은 정확도 요구
                'allow_partial_ocr': False,  # 전체 또는 없음
                'priority': 1,
            },
            PageType.TECHNICAL_MANUAL: {
                'min_quality': 0.6,
                'allow_partial_ocr': True,
                'priority': 2,
            },
            PageType.TABLE_HEAVY: {
                'min_quality': 0.5,  # 표는 OCR이 더 효과적일 수 있음
                'allow_partial_ocr': True,
                'priority': 2,
            },
            PageType.IMAGE_HEAVY: {
                'min_quality': 0.3,  # 이미지는 무조건 OCR
                'allow_partial_ocr': False,
                'priority': 1,
            },
            PageType.MIXED: {
                'min_quality': 0.5,
                'allow_partial_ocr': True,
                'priority': 3,
            },
            PageType.UNKNOWN: {
                'min_quality': 0.5,
                'allow_partial_ocr': True,
                'priority': 4,
            },
        }
    
    def should_use_ocr(
        self,
        pages: List[Tuple[int, str]],
        layout_blocks: Optional[Dict[int, List[Dict]]] = None
    ) -> Tuple[bool, List[int], Dict]:
        """
        OCR 사용 여부 및 대상 페이지 결정 (고도화)
        
        Returns:
            (전체_OCR_필요, OCR_필요_페이지_리스트, 상세_분석)
        """
        if not pages:
            return False, [], {}
        
        page_qualities = []
        for page_no, text in pages:
            quality = self._evaluate_page_quality(
                page_no, text,
                layout_blocks.get(page_no) if layout_blocks else None
            )
            page_qualities.append(quality)
        
        # 전체 OCR 필요 판단 (개선된 로직)
        needs_full_ocr = self._needs_full_ocr(page_qualities)
        
        # 부분 OCR 필요 페이지 (우선순위 정렬)
        partial_ocr_pages = sorted(
            [q.page_no for q in page_qualities if q.needs_ocr and not needs_full_ocr],
            key=lambda pn: next(q.ocr_priority for q in page_qualities if q.page_no == pn)
        )
        
        # 상세 분석
        analysis = {
            'total_pages': len(pages),
            'ocr_needed_pages': len([q for q in page_qualities if q.needs_ocr]),
            'high_priority_pages': len([q for q in page_qualities if q.ocr_priority <= 2]),
            'page_types': {pt.value: len([q for q in page_qualities if q.page_type == pt]) 
                          for pt in PageType},
            'avg_confidence': sum(q.confidence for q in page_qualities) / len(page_qualities),
            'problematic_pages': [q.page_no for q in page_qualities 
                                 if q.cid_ratio > self.cid_threshold or q.rotation_detected],
        }
        
        return needs_full_ocr, partial_ocr_pages, analysis
    
    def _evaluate_page_quality(
        self,
        page_no: int,
        text: str,
        layout_blocks: Optional[List[Dict]] = None
    ) -> PageQuality:
        """개별 페이지 품질 평가 (고도화)"""
        
        # 1. 페이지 타입 분류
        page_type = self._classify_page_type(text, layout_blocks)
        
        # 2. 기본 품질 지표
        text_density = self._calculate_text_density(text)
        layout_quality = self._evaluate_layout_quality(text, layout_blocks)
        readable_ratio = self._calculate_readable_ratio(text)
        
        # 3. 의미론적 품질 (신규)
        semantic_quality = self._evaluate_semantic_quality(text)
        
        # 4. 특수 요소 감지
        has_tables = self._has_table_elements(text, layout_blocks)
        has_images = self._has_image_elements(layout_blocks)
        has_legal_markers = self._has_legal_markers(text)
        has_technical_terms = self._has_technical_terms(text)
        
        # 5. 문제 패턴 감지
        cid_ratio = self._calculate_cid_ratio(text)
        rotation_detected = self._detect_rotation(layout_blocks)
        
        # 6. OCR 필요 여부 종합 판단 (페이지 타입별 전략)
        needs_ocr, ocr_priority = self._judge_ocr_need(
            page_type, text_density, layout_quality, semantic_quality,
            has_tables, has_images, readable_ratio, cid_ratio, rotation_detected
        )
        
        # 7. 신뢰도 계산
        confidence = self._calculate_confidence(
            text_density, layout_quality, semantic_quality, readable_ratio, page_type
        )
        
        return PageQuality(
            page_no=page_no,
            page_type=page_type,
            text_density=text_density,
            layout_quality=layout_quality,
            semantic_quality=semantic_quality,
            has_tables=has_tables,
            has_images=has_images,
            has_legal_markers=has_legal_markers,
            has_technical_terms=has_technical_terms,
            readable_ratio=readable_ratio,
            cid_ratio=cid_ratio,
            rotation_detected=rotation_detected,
            needs_ocr=needs_ocr,
            confidence=confidence,
            ocr_priority=ocr_priority
        )
    
    def _classify_page_type(
        self, 
        text: str, 
        layout_blocks: Optional[List[Dict]]
    ) -> PageType:
        """페이지 타입 분류 (원자력 문서 특화)"""
        if not text:
            return PageType.UNKNOWN
        
        # 법률 문서 체크
        legal_score = sum(1 for pattern in self.legal_patterns 
                         if re.search(pattern, text)) / len(self.legal_patterns)
        
        # 기술 용어 체크
        tech_score = sum(1 for keyword in self.technical_keywords 
                        if keyword.lower() in text.lower()) / len(self.technical_keywords)
        
        # 표 밀도
        table_density = self._calculate_table_density(text, layout_blocks)
        
        # 이미지 밀도
        image_density = self._calculate_image_density(layout_blocks)
        
        # 분류 로직
        if legal_score > 0.3:
            return PageType.LEGAL_TEXT
        elif tech_score > 0.15 and table_density < 0.3:
            return PageType.TECHNICAL_MANUAL
        elif table_density > 0.4:
            return PageType.TABLE_HEAVY
        elif image_density > 0.3:
            return PageType.IMAGE_HEAVY
        elif table_density > 0.2 or image_density > 0.15:
            return PageType.MIXED
        else:
            return PageType.UNKNOWN
    
    def _calculate_text_density(self, text: str) -> float:
        """텍스트 밀도 계산 (0-1)"""
        if not text:
            return 0.0
        
        # 공백 제외 문자 수
        non_space_chars = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        # 원자력 안전 문서는 평균 페이지당 1500자 내외 (기준 조정)
        expected_chars = int(os.getenv("OCR_EXPECTED_CHARS_PER_PAGE", "1500"))
        density = min(non_space_chars / expected_chars, 1.0)
        
        return density
    
    def _evaluate_layout_quality(
        self, 
        text: str, 
        layout_blocks: Optional[List[Dict]]
    ) -> float:
        """레이아웃 품질 평가 (0-1) - 개선"""
        if not layout_blocks:
            return self._estimate_quality_from_text(text)
        
        quality_score = 0.0
        total_checks = 0
        
        # 1. bbox 정보 완전성 (가중치 0.3)
        blocks_with_valid_bbox = sum(
            1 for b in layout_blocks 
            if self._is_valid_bbox(b.get('bbox'))
        )
        if layout_blocks:
            quality_score += (blocks_with_valid_bbox / len(layout_blocks)) * 0.3
            total_checks += 0.3
        
        # 2. 텍스트-bbox 일치도 (가중치 0.25)
        blocks_with_text = sum(
            1 for b in layout_blocks 
            if b.get('text') and len(b['text'].strip()) >= 2
        )
        if layout_blocks:
            quality_score += (blocks_with_text / len(layout_blocks)) * 0.25
            total_checks += 0.25
        
        # 3. 정렬 품질 - 수평/수직 정렬 (가중치 0.25)
        alignment_score = self._calculate_alignment_quality(layout_blocks)
        quality_score += alignment_score * 0.25
        total_checks += 0.25
        
        # 4. 블록 간 중복/겹침 체크 (가중치 0.2)
        overlap_penalty = self._calculate_overlap_penalty(layout_blocks)
        quality_score += (1.0 - overlap_penalty) * 0.2
        total_checks += 0.2
        
        return quality_score / max(total_checks, 1)
    
    def _evaluate_semantic_quality(self, text: str) -> float:
        """
        의미론적 품질 평가 (신규)
        - 문장 구조 완전성
        - 단어 완전성
        - 컨텍스트 연속성
        """
        if not text:
            return 0.0
        
        quality_score = 0.0
        total_checks = 0
        
        # 1. 문장 종결 비율 (완전한 문장)
        sentences = re.split(r'[.!?。]\s+', text)
        complete_sentences = sum(
            1 for s in sentences 
            if len(s.strip()) > 10 and re.search(r'[가-힣a-zA-Z]{3,}', s)
        )
        if sentences:
            quality_score += (complete_sentences / len(sentences)) * 0.4
            total_checks += 0.4
        
        # 2. 단어 완전성 (공백으로 분리된 단어)
        words = text.split()
        complete_words = sum(
            1 for w in words 
            if len(w) >= 2 and re.search(r'[가-힣a-zA-Z]{2,}', w)
        )
        if words:
            quality_score += (complete_words / len(words)) * 0.3
            total_checks += 0.3
        
        # 3. 깨진 문자 패턴 체크
        broken_chars = len(re.findall(r'[�□◇○●■▲▼]', text))
        if len(text) > 0:
            broken_ratio = broken_chars / len(text)
            quality_score += max(0, (1.0 - broken_ratio * 10)) * 0.3
            total_checks += 0.3
        
        return quality_score / max(total_checks, 1)
    
    def _calculate_readable_ratio(self, text: str) -> float:
        """읽기 가능한 문자 비율 - 개선"""
        if not text:
            return 0.0
        
        # 의미있는 문자 (한글, 영문, 숫자, 일반 기호, 한자)
        readable = re.findall(r'[가-힣a-zA-Z0-9\s,.\-:()%/一-龥]', text)
        total_chars = len(text)
        
        if total_chars == 0:
            return 0.0
        
        return len(readable) / total_chars
    
    def _calculate_cid_ratio(self, text: str) -> float:
        """CID 코드 비율 계산"""
        if not text:
            return 0.0
        
        cid_pattern = re.compile(r'\(cid:\d+\)')
        cid_matches = cid_pattern.findall(text)
        cid_char_count = sum(len(m) for m in cid_matches)
        
        return cid_char_count / len(text) if len(text) > 0 else 0.0
    
    def _detect_rotation(self, layout_blocks: Optional[List[Dict]]) -> bool:
        """페이지 회전 감지 (bbox 패턴 분석)"""
        if not layout_blocks or len(layout_blocks) < 5:
            return False
        
        # Y좌표가 X좌표보다 변화가 적으면 회전 가능성
        x_variance = self._calculate_coordinate_variance([
            b.get('bbox', {}).get('x0', 0) for b in layout_blocks
        ])
        y_variance = self._calculate_coordinate_variance([
            b.get('bbox', {}).get('y0', 0) for b in layout_blocks
        ])
        
        # 비정상적인 패턴
        if y_variance < x_variance * 0.3:
            return True
        
        return False
    
    def _judge_ocr_need(
        self,
        page_type: PageType,
        text_density: float,
        layout_quality: float,
        semantic_quality: float,
        has_tables: bool,
        has_images: bool,
        readable_ratio: float,
        cid_ratio: float,
        rotation_detected: bool
    ) -> Tuple[bool, int]:
        """
        OCR 필요 여부 종합 판단 (페이지 타입별 전략)
        
        Returns:
            (needs_ocr, priority)
        """
        strategy = self.type_ocr_strategies.get(page_type, self.type_ocr_strategies[PageType.UNKNOWN])
        min_quality = strategy['min_quality']
        priority = strategy['priority']
        
        # ========== 강제 OCR 조건 (최우선) ==========
        # 1. CID 코드 임계값 초과
        if cid_ratio > self.cid_threshold:
            return True, 1
        
        # 2. 회전 감지
        if rotation_detected:
            return True, 1
        
        # 3. 텍스트가 거의 없음
        if text_density < self.min_text_density:
            return True, 2
        
        # 4. 읽을 수 없는 문자가 과도함
        if readable_ratio < self.min_readable_ratio:
            return True, 2
        
        # ========== 품질 기반 판단 ==========
        # 종합 품질 점수 (가중 평균)
        overall_quality = (
            text_density * 0.25 +
            layout_quality * 0.25 +
            semantic_quality * 0.3 +
            readable_ratio * 0.2
        )
        
        # 페이지 타입별 최소 품질 기준
        if overall_quality < min_quality:
            return True, priority
        
        # ========== 특수 케이스 ==========
        # 법률 문서 + 낮은 의미론적 품질
        if page_type == PageType.LEGAL_TEXT and semantic_quality < 0.7:
            return True, 1
        
        # 표가 있고 레이아웃 품질이 낮음
        if has_tables and layout_quality < 0.6:
            return True, priority
        
        # 이미지가 있고 텍스트 밀도가 낮음
        if has_images and text_density < 0.3:
            return True, priority
        
        return False, 5
    
    def _calculate_confidence(
        self,
        text_density: float,
        layout_quality: float,
        semantic_quality: float,
        readable_ratio: float,
        page_type: PageType
    ) -> float:
        """판단 신뢰도 계산 - 개선"""
        # 기본 신뢰도 (가중 평균)
        base_confidence = (
            text_density * 0.2 +
            layout_quality * 0.25 +
            semantic_quality * 0.3 +
            readable_ratio * 0.25
        )
        
        # 페이지 타입별 신뢰도 보정
        type_confidence_boost = {
            PageType.LEGAL_TEXT: 0.1,  # 법률 문서는 더 높은 신뢰도
            PageType.TECHNICAL_MANUAL: 0.05,
            PageType.TABLE_HEAVY: -0.05,  # 표는 판단이 어려움
            PageType.IMAGE_HEAVY: -0.1,
            PageType.MIXED: 0.0,
            PageType.UNKNOWN: -0.05,
        }
        
        boost = type_confidence_boost.get(page_type, 0.0)
        final_confidence = base_confidence + boost
        
        return min(max(final_confidence, 0.0), 1.0)
    
    def _needs_full_ocr(self, page_qualities: List[PageQuality]) -> bool:
        """전체 문서 OCR 필요 여부 - 개선"""
        if not page_qualities:
            return False
        
        # 고우선순위 페이지가 50% 이상
        high_priority_count = sum(1 for q in page_qualities if q.needs_ocr and q.ocr_priority <= 2)
        if high_priority_count / len(page_qualities) >= 0.5:
            return True
        
        # 전체 OCR 필요 페이지가 60% 이상
        ocr_needed_count = sum(1 for q in page_qualities if q.needs_ocr)
        if ocr_needed_count / len(page_qualities) >= 0.6:
            return True
        
        # 법률 문서가 대부분이고 OCR 필요 페이지가 30% 이상
        legal_pages = sum(1 for q in page_qualities if q.page_type == PageType.LEGAL_TEXT)
        if legal_pages / len(page_qualities) >= 0.7 and ocr_needed_count / len(page_qualities) >= 0.3:
            return True
        
        return False
    
    # ========== 유틸리티 메서드 ==========
    
    def _is_valid_bbox(self, bbox) -> bool:
        """bbox 유효성 검사"""
        if not bbox:
            return False
        
        if isinstance(bbox, dict):
            x0, y0, x1, y1 = bbox.get('x0', 0), bbox.get('y0', 0), bbox.get('x1', 0), bbox.get('y1', 0)
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        else:
            return False
        
        return x1 > x0 and y1 > y0 and (x1 - x0) * (y1 - y0) > 1
    
    def _calculate_alignment_quality(self, layout_blocks: List[Dict]) -> float:
        """블록 정렬 품질 계산"""
        if len(layout_blocks) < 2:
            return 1.0
        
        # Y좌표 기준 정렬 품질
        y_coords = []
        for b in layout_blocks:
            bbox = b.get('bbox', {})
            if isinstance(bbox, dict):
                y_coords.append(bbox.get('y0', 0))
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                y_coords.append(bbox[1])
        
        if len(y_coords) < 2:
            return 0.5
        
        y_coords.sort()
        gaps = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
        
        if not gaps:
            return 0.5
        
        avg_gap = sum(gaps) / len(gaps)
        if avg_gap == 0:
            return 0.5
        
        variance = sum((g - avg_gap)**2 for g in gaps) / len(gaps)
        consistency = max(0, 1 - (variance / (avg_gap**2 + 1)))
        
        return consistency
    
    def _calculate_overlap_penalty(self, layout_blocks: List[Dict]) -> float:
        """블록 간 중복/겹침 페널티 계산"""
        if len(layout_blocks) < 2:
            return 0.0
        
        overlap_count = 0
        total_pairs = 0
        
        for i in range(len(layout_blocks)):
            for j in range(i + 1, len(layout_blocks)):
                total_pairs += 1
                if self._blocks_overlap(layout_blocks[i], layout_blocks[j]):
                    overlap_count += 1
        
        return overlap_count / max(total_pairs, 1)
    
    def _blocks_overlap(self, block1: Dict, block2: Dict) -> bool:
        """두 블록이 겹치는지 확인"""
        bbox1 = block1.get('bbox', {})
        bbox2 = block2.get('bbox', {})
        
        if isinstance(bbox1, dict):
            x1_0, y1_0, x1_1, y1_1 = bbox1.get('x0', 0), bbox1.get('y0', 0), bbox1.get('x1', 0), bbox1.get('y1', 0)
        else:
            return False
        
        if isinstance(bbox2, dict):
            x2_0, y2_0, x2_1, y2_1 = bbox2.get('x0', 0), bbox2.get('y0', 0), bbox2.get('x1', 0), bbox2.get('y1', 0)
        else:
            return False
        
        # AABB 충돌 검사
        return not (x1_1 <= x2_0 or x2_1 <= x1_0 or y1_1 <= y2_0 or y2_1 <= y1_0)
    
    def _calculate_coordinate_variance(self, coords: List[float]) -> float:
        """좌표 분산 계산"""
        if not coords:
            return 0.0
        
        mean = sum(coords) / len(coords)
        variance = sum((c - mean)**2 for c in coords) / len(coords)
        return variance
    
    def _estimate_quality_from_text(self, text: str) -> float:
        """텍스트만으로 품질 추정"""
        if not text:
            return 0.0
        
        lines = text.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        if not non_empty_lines:
            return 0.0
        
        quality_score = 1.0
        
        # 깨진 문자 패턴
        broken_pattern_count = sum(
            1 for line in non_empty_lines
            if any(c in line for c in ['�', '□', '◇', '○'])
        )
        quality_score -= (broken_pattern_count / len(non_empty_lines)) * 0.5
        
        # 의미없는 짧은 줄
        too_short_count = sum(1 for line in non_empty_lines if len(line.strip()) < 3)
        quality_score -= (too_short_count / len(non_empty_lines)) * 0.3
        
        return max(0.0, quality_score)
    
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
        
        for pattern in table_indicators:
            if re.search(pattern, text):
                return True
        
        # 레이아웃 기반 검사
        if layout_blocks and len(layout_blocks) >= 6:
            # 격자 패턴 검사
            x_coords = set()
            y_coords = set()
            for block in layout_blocks:
                bbox = block.get('bbox', {})
                if isinstance(bbox, dict):
                    x_coords.add(round(bbox.get('x0', 0), -1))
                    y_coords.add(round(bbox.get('y0', 0), -1))
            
            # X, Y 좌표가 각각 3개 이상의 고유값 = 격자 구조
            if len(x_coords) >= 3 and len(y_coords) >= 3:
                return True
        
        return False
    
    def _has_image_elements(self, layout_blocks: Optional[List[Dict]]) -> bool:
        """이미지 요소 존재 여부"""
        if not layout_blocks:
            return False
        
        for block in layout_blocks:
            if block.get('type') == 'image':
                return True
            
            # 텍스트 없고 bbox가 큰 경우
            bbox = block.get('bbox', {})
            if isinstance(bbox, dict):
                width = bbox.get('x1', 0) - bbox.get('x0', 0)
                height = bbox.get('y1', 0) - bbox.get('y0', 0)
            else:
                width = height = 0
            
            text = block.get('text', '').strip()
            
            if not text and width * height > 15000:  # 임계값 상향
                return True
        
        return False
    
    def _has_legal_markers(self, text: str) -> bool:
        """법률 조항 마커 존재 여부"""
        for pattern in self.legal_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _has_technical_terms(self, text: str) -> bool:
        """기술 용어 존재 여부"""
        text_lower = text.lower()
        match_count = sum(1 for keyword in self.technical_keywords 
                         if keyword.lower() in text_lower)
        return match_count >= 2  # 최소 2개 이상
    
    def _calculate_table_density(
        self, 
        text: str, 
        layout_blocks: Optional[List[Dict]]
    ) -> float:
        """표 밀도 계산"""
        if not text:
            return 0.0
        
        # 표 패턴 비율
        table_char_count = 0
        for pattern in [r'[\|\+\-─│]', r'[┌┐└┘├┤┬┴┼]']:
            table_char_count += len(re.findall(pattern, text))
        
        density = table_char_count / len(text) if len(text) > 0 else 0.0
        return min(density * 5, 1.0)  # 스케일 조정
    
    def _calculate_image_density(self, layout_blocks: Optional[List[Dict]]) -> float:
        """이미지 밀도 계산"""
        if not layout_blocks:
            return 0.0
        
        image_blocks = sum(1 for b in layout_blocks 
                          if b.get('type') == 'image' or 
                          (not b.get('text', '').strip() and self._is_valid_bbox(b.get('bbox'))))
        
        return image_blocks / len(layout_blocks)
    
    def get_quality_report(
        self,
        pages: List[Tuple[int, str]],
        layout_blocks: Optional[Dict[int, List[Dict]]] = None
    ) -> Dict:
        """페이지 품질 리포트 생성 (디버깅용) - 고도화"""
        qualities = []
        for page_no, text in pages:
            quality = self._evaluate_page_quality(
                page_no, text,
                layout_blocks.get(page_no) if layout_blocks else None
            )
            qualities.append({
                'page': quality.page_no,
                'type': quality.page_type.value,
                'text_density': f"{quality.text_density:.3f}",
                'layout_quality': f"{quality.layout_quality:.3f}",
                'semantic_quality': f"{quality.semantic_quality:.3f}",
                'readable_ratio': f"{quality.readable_ratio:.3f}",
                'cid_ratio': f"{quality.cid_ratio:.3f}",
                'has_tables': quality.has_tables,
                'has_images': quality.has_images,
                'has_legal_markers': quality.has_legal_markers,
                'has_technical_terms': quality.has_technical_terms,
                'rotation_detected': quality.rotation_detected,
                'needs_ocr': quality.needs_ocr,
                'ocr_priority': quality.ocr_priority,
                'confidence': f"{quality.confidence:.3f}"
            })
        
        needs_full, partial_pages, analysis = self.should_use_ocr(pages, layout_blocks)
        
        return {
            **analysis,
            'needs_full_ocr': needs_full,
            'partial_ocr_pages': partial_pages,
            'page_details': qualities,
            'recommendations': self._generate_recommendations(qualities)
        }
    
    def _generate_recommendations(self, qualities: List[Dict]) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        high_cid_pages = [q for q in qualities if float(q['cid_ratio']) > self.cid_threshold]
        if high_cid_pages:
            recommendations.append(
                f"⚠️  {len(high_cid_pages)}개 페이지에서 CID 코드 감지 → 폰트 임베딩 문제 가능성"
            )
        
        low_semantic_pages = [q for q in qualities if float(q['semantic_quality']) < 0.4]
        if low_semantic_pages:
            recommendations.append(
                f"⚠️  {len(low_semantic_pages)}개 페이지에서 낮은 의미론적 품질 → OCR 품질 검토 필요"
            )
        
        rotated_pages = [q for q in qualities if q['rotation_detected']]
        if rotated_pages:
            recommendations.append(
                f"⚠️  {len(rotated_pages)}개 페이지에서 회전 감지 → 전처리 필요"
            )
        
        legal_pages = [q for q in qualities if q['type'] == 'legal']
        if legal_pages:
            recommendations.append(
                f"ℹ️  {len(legal_pages)}개 법률 문서 페이지 감지 → 높은 정확도 요구"
            )
        
        return recommendations or ["✅ 모든 페이지 품질 양호"]