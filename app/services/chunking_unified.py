# app/services/chunking_unified.py
from __future__ import annotations
import os, re
from typing import Dict, List, Tuple

from app.services.embedding_model import get_embedding_model

def _make_encoder():
    """
    llama_router의 _make_encoder와 동일 동작:
    - sentence-transformers 모델에서 tokenizer/max_len 가져오고
    - add_special_tokens=False로 인코딩
    """
    m = get_embedding_model()
    tok = getattr(m, "tokenizer", None)
    max_len = int(getattr(m, "max_seq_length", 128))
    def enc(s: str):
        if tok is None:
            return []
        return tok.encode(s, add_special_tokens=False) or []
    return enc, max_len

def build_chunks(
    pages_std: List[Tuple[int, str]],
    layout_map: Dict[int, List[Dict]] | None = None,
    *,
    job_id: str | None = None
) -> List[Tuple[str, Dict]]:
    """
    llama_router의 청킹 파이프라인을 공용화.
    - English Technical → Law → SmartChunkerPlus(layout) → SmartChunker → 최후 폴백
    - 환경변수 RAG_TARGET_TOKENS / RAG_OVERLAP_TOKENS / RAG_MIN_CHUNK_TOKENS 등 동일 지원
    - 반환 형식: [(text, meta_dict)]
    """
    layout_map = layout_map or {}

    # 인코더/길이
    enc, max_len = _make_encoder()
    default_target = max(64, max_len - 16)
    default_overlap = min(50, default_target // 4)

    target_tokens = int(os.getenv("RAG_TARGET_TOKENS", str(default_target)))
    overlap_tokens = int(os.getenv("RAG_OVERLAP_TOKENS", str(default_overlap)))
    min_chunk_tokens = int(os.getenv("RAG_MIN_CHUNK_TOKENS", "100"))

    chunks: list[tuple[str, dict]] | None = None

    def _nonempty(chs):
        return bool(chs) and any(isinstance(t, (list, tuple)) and len(t) >= 2 and (t[0] or "").strip() for t in chs)

    # 1) 영어 기술 청커 (우선 시도)
    if os.getenv("RAG_ENABLE_EN_TECH_CHUNKER", "1") == "1":
        try:
            from app.services.english_technical_chunker import english_technical_chunk_pages
            if job_id: print(f"[CHUNK-{job_id}] Trying English technical chunker...")
            en_target_tokens = int(os.getenv("RAG_EN_TARGET_TOKENS", "500"))
            ch = english_technical_chunk_pages(pages_std, enc, en_target_tokens, overlap_tokens, layout_map)
            if _nonempty(ch):
                if job_id: print(f"[CHUNK-{job_id}] English technical chunker -> {len(ch)} chunks")
                return ch
            if job_id: print(f"[CHUNK-{job_id}] English technical chunker empty → fallback")
        except Exception as e:
            if job_id: print(f"[CHUNK-{job_id}] English technical chunker error: {e}")

    # 2) 법령 청커
    if os.getenv("RAG_ENABLE_LAW_CHUNKER", "1") == "1":
        try:
            from app.services.law_chunker import law_chunk_pages
            if job_id: print(f"[CHUNK-{job_id}] Trying NuclearLegalChunker...")
            ch = law_chunk_pages(
                pages_std, enc, target_tokens, overlap_tokens,
                layout_blocks=layout_map, min_chunk_tokens=min_chunk_tokens
            )
            if _nonempty(ch):
                if job_id: print(f"[CHUNK-{job_id}] Law chunker -> {len(ch)} chunks")
                return ch
            if job_id: print(f"[CHUNK-{job_id}] Law chunker empty → fallback")
        except Exception as e:
            if job_id: print(f"[CHUNK-{job_id}] Law chunker error: {e}")

    # 3) SmartChunker Plus (레이아웃)
    if os.getenv("RAG_ENABLE_LAYOUT_CHUNKER", "1") == "1" and layout_map:
        try:
            from app.services.chunker import smart_chunk_pages_plus
            if job_id: print(f"[CHUNK-{job_id}] Using layout-aware chunker (SmartChunkerPlus)...")
            ch = smart_chunk_pages_plus(pages_std, enc, target_tokens, overlap_tokens, layout_map)
            if _nonempty(ch):
                if job_id: print(f"[CHUNK-{job_id}] Layout chunker -> {len(ch)} chunks")
                return ch
            if job_id: print(f"[CHUNK-{job_id}] Layout chunker empty → fallback")
        except Exception as e:
            if job_id: print(f"[CHUNK-{job_id}] Layout chunker error: {e}")

    # 4) 기본 SmartChunker
    try:
        from app.services.chunker import smart_chunk_pages
        if job_id: print(f"[CHUNK-{job_id}] Using basic smart chunker...")
        ch = smart_chunk_pages(pages_std, enc, target_tokens, overlap_tokens, layout_map)
        if _nonempty(ch):
            if job_id: print(f"[CHUNK-{job_id}] Basic chunker -> {len(ch)} chunks")
            return ch
        if job_id: print(f"[CHUNK-{job_id}] Basic chunker empty → final fallback")
    except Exception as e:
        if job_id: print(f"[CHUNK-{job_id}] Basic chunker error: {e}")

    # 5) 최후 폴백: 문서 전체를 하나로
    flat_texts = []
    for _, t in pages_std or []:
        tt = (t or "").strip()
        if tt:
            tt = re.sub(r'\b인접행\s*묶음\b', '', tt)
            tt = re.sub(r'\b[가-힣]*\s*묶음\b', '', tt)
            tt = re.sub(r'[\r\n\s]+', ' ', tt)
            if tt.strip():
                flat_texts.append(tt.strip())
    fallback_text = "\n\n".join(flat_texts).strip()
    if not fallback_text:
        if os.getenv("RAG_ALLOW_EMPTY_FALLBACK", "1") == "1":
            fallback_text = "[Document processed but no readable text content found]"
        else:
            raise RuntimeError("모든 청킹 방법이 실패했습니다.")
    tokens = len(enc(fallback_text))
    if job_id: print(f"[CHUNK-{job_id}] Fallback chunk created: {tokens} tokens")
    return [(fallback_text, {"page": 1, "pages": [1], "section": "", "token_count": tokens, "bboxes": {}})]
