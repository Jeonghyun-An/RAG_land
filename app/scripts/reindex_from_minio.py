#!/usr/bin/env python3
"""
MinIO → Milvus 전체 재인덱싱 스크립트 (독립 실행 버전)

사용법:
    cd /app
    python -m app.scripts.reindex_from_minio [OPTIONS]
    python -m app.scripts.reindex_from_minio --skip-errors

옵션:
    --dry-run                   실제 처리 없이 목록만 출력
    --limit N                   처리할 문서 개수 제한
    --force                     이미 인덱싱된 문서도 강제 재처리
    --doc-id DOC_ID            특정 문서만 재인덱싱
    --skip-errors              에러 발생 시 계속 진행
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import json

# app 모듈 경로 추가
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from app.services.minio_store import MinIOStore
from app.services.milvus_store_v2 import MilvusStoreV2
from app.services.embedding_model import embed, get_sentence_embedding_dimension
from app.services.file_parser import parse_pdf_pages_from_bytes, parse_pdf_blocks_from_bytes
from app.api.java_router import perform_advanced_chunking


def _coerce_chunks_for_milvus(chs):
    """청크 정규화 (프로덕션 로직과 동일)"""
    safe = []
    for t in chs or []:
        if not isinstance(t, (list, tuple)) or len(t) < 2:
            continue
        text, meta = t[0], t[1]
        text = "" if text is None else str(text)
        if not isinstance(meta, dict):
            meta = {}

        section = str(meta.get("section", ""))[:512]
        pages = meta.get("pages")
        if isinstance(pages, (list, tuple)) and len(pages) > 0:
            try:
                page = int(pages[0])
            except Exception:
                page = int(meta.get("page", 0))
        else:
            try:
                page = int(meta.get("page", 0))
            except Exception:
                page = 0

        safe.append((text, {"page": page, "section": section, "pages": pages or [], "bboxes": meta.get("bboxes", {})}))

    out = []
    last = None
    for it in safe:
        if it[0] and it != last:
            out.append(it)
            last = it
    return out


def process_single_document(
    minio: MinIOStore,
    mvs: MilvusStoreV2,
    object_name: str,
    force: bool = False,
    skip_errors: bool = False
) -> Dict:
    """단일 문서 재인덱싱"""
    result = {
        "doc_id": None,
        "object_name": object_name,
        "status": "error",
        "chunks": 0,
        "message": ""
    }
    
    try:
        # 1. doc_id 추출
        if not object_name.endswith(".pdf"):
            result["message"] = "PDF 파일이 아님"
            result["status"] = "skipped"
            return result
        
        doc_id = object_name.replace("uploaded/", "").replace(".pdf", "")
        result["doc_id"] = doc_id
        
        # 2. 이미 인덱싱된 문서 체크
        if not force:
            try:
                existing_count = mvs.count_by_doc(doc_id)
                if existing_count > 0:
                    result["status"] = "skipped"
                    result["chunks"] = existing_count
                    result["message"] = f"이미 {existing_count}개 청크 존재"
                    return result
            except Exception as e:
                print(f"[WARN] count_by_doc 실패: {e}")
        
        # 3. MinIO에서 PDF 다운로드
        print(f"\n{'='*60}")
        print(f"[처리 시작] {doc_id}")
        print(f"{'='*60}")
        
        pdf_bytes = minio.get_bytes(object_name)
        if not pdf_bytes:
            result["message"] = "PDF 다운로드 실패"
            return result
        
        print(f"[다운로드] PDF 크기: {len(pdf_bytes):,} bytes")
        
        # 4. PDF 파싱
        print(f"[파싱] PDF 텍스트 추출 중...")
        pages_raw = parse_pdf_pages_from_bytes(pdf_bytes)
        
        print(f"[파싱] PDF 레이아웃 추출 중...")
        blocks_data = parse_pdf_blocks_from_bytes(pdf_bytes)
        
        if not pages_raw:
            result["message"] = "PDF 파싱 실패"
            return result
        
        print(f"[파싱 완료] 총 {len(pages_raw)} 페이지")
        
        # 5. 페이지 구조 변환
        pages_std = []
        for idx, text in enumerate(pages_raw, start=1):
            pages_std.append((idx, text))
        
        # 레이아웃 매핑
        layout_map = {}
        for page_idx, blocks in blocks_data:
            layout_map[page_idx] = blocks
        
        # 6. 고도화 청킹
        print(f"[청킹] 고도화 청킹 시작...")
        chunks = perform_advanced_chunking(pages_std, layout_map, job_id="")
        
        if not chunks:
            result["message"] = "청킹 결과 없음"
            return result
        
        print(f"[청킹 완료] 총 {len(chunks)} 개 청크 생성")
        
        # 7. 청크 정규화
        chunks = _coerce_chunks_for_milvus(chunks)
        
        if not chunks:
            result["message"] = "정규화 후 청크 없음"
            return result
        
        print(f"[정규화] {len(chunks)} 개 청크")
        
        # 8. 기존 청크 삭제
        print(f"[Milvus] 기존 청크 삭제 중...")
        try:
            deleted = mvs._delete_by_doc_id(doc_id)
            print(f"[Milvus] 기존 청크 {deleted}개 삭제됨")
        except Exception as e:
            print(f"[WARN] 기존 청크 삭제 실패: {e}")
        
        # 9. 임베딩 + Milvus 삽입
        print(f"[Milvus] 임베딩 및 삽입 중...")
        insert_result = mvs.insert(
            doc_id=doc_id,
            chunks=chunks,
            embed_fn=embed
        )
        
        inserted_count = insert_result.get("inserted", 0)
        
        print(f"[완료] {inserted_count}개 청크 인덱싱 완료")
        
        result["status"] = "success"
        result["chunks"] = inserted_count
        result["message"] = f"{inserted_count}개 청크 인덱싱 완료"
        
        return result
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        result["message"] = error_msg
        result["status"] = "error"
        
        print(f"\n[ERROR] {doc_id or object_name}")
        print(f"  에러: {error_msg}")
        
        if not skip_errors:
            print(f"\n상세 트레이스:")
            traceback.print_exc()
        
        return result


def main():
    parser = argparse.ArgumentParser(description="MinIO → Milvus 재인덱싱")
    parser.add_argument("--dry-run", action="store_true", help="목록만 출력")
    parser.add_argument("--limit", type=int, help="처리할 문서 개수 제한")
    parser.add_argument("--force", action="store_true", help="강제 재처리")
    parser.add_argument("--doc-id", help="특정 문서만")
    parser.add_argument("--skip-errors", action="store_true", help="에러 시 계속 진행")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MinIO → Milvus 전체 재인덱싱")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # MinIO 연결
    print("MinIO 연결 중...")
    try:
        minio = MinIOStore()
        if not minio.healthcheck():
            print("MinIO 연결 실패!")
            return 1
        print("MinIO 연결 성공")
    except Exception as e:
        print(f"MinIO 초기화 실패: {e}")
        return 1
    
    # Milvus 연결
    print("Milvus 연결 중...")
    try:
        dim = get_sentence_embedding_dimension()
        mvs = MilvusStoreV2(dim=dim)
        print(f"Milvus 연결 성공 (dim={dim})")
    except Exception as e:
        print(f"Milvus 초기화 실패: {e}")
        return 1
    
    print()
    
    # 문서 목록
    if args.doc_id:
        object_name = f"uploaded/{args.doc_id}.pdf"
        if not minio.exists(object_name):
            print(f"문서를 찾을 수 없습니다: {object_name}")
            return 1
        objects = [object_name]
    else:
        print(f"MinIO에서 파일 목록 가져오는 중...")
        objects = minio.list_files(prefix="uploaded/")
        objects = [obj for obj in objects if obj.endswith(".pdf")]
    
    print(f"발견된 PDF 파일: {len(objects)}개")
    
    if args.limit:
        objects = objects[:args.limit]
        print(f"제한 적용: {len(objects)}개만 처리")
    
    if not objects:
        print("처리할 문서가 없습니다.")
        return 0
    
    # Dry-run
    if args.dry_run:
        print("\n[DRY-RUN 모드] 처리할 문서 목록:")
        print("-" * 80)
        for idx, obj in enumerate(objects, 1):
            doc_id = obj.replace("uploaded/", "").replace(".pdf", "")
            print(f"{idx:3d}. {doc_id:20s} ({obj})")
        print("-" * 80)
        print(f"총 {len(objects)}개 문서")
        return 0
    
    # 실제 처리
    print("\n재인덱싱 시작...")
    print("=" * 80)
    
    results = {"success": [], "skipped": [], "error": []}
    start_time = datetime.now()
    
    for idx, object_name in enumerate(objects, 1):
        print(f"\n[{idx}/{len(objects)}] {object_name}")
        
        result = process_single_document(
            minio=minio,
            mvs=mvs,
            object_name=object_name,
            force=args.force,
            skip_errors=args.skip_errors
        )
        
        status = result["status"]
        results[status].append(result)
        
        if status == "success":
            print(f"  성공: {result['chunks']}개 청크")
        elif status == "skipped":
            print(f"  ⏭스킵: {result['message']}")
        else:
            print(f"  실패: {result['message']}")
            if not args.skip_errors:
                print("\n재인덱싱 중단됨 (--skip-errors 옵션으로 계속 진행 가능)")
                break
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    # 최종 결과
    print("\n" + "=" * 80)
    print("재인덱싱 완료")
    print("=" * 80)
    print(f"총 처리 시간: {elapsed:.1f}초")
    print(f"총 문서 수: {len(objects)}개")
    print(f"  성공: {len(results['success'])}개")
    print(f"  ⏭스킵: {len(results['skipped'])}개")
    print(f"  실패: {len(results['error'])}개")
    
    if results['success']:
        total_chunks = sum(r['chunks'] for r in results['success'])
        avg_chunks = total_chunks / len(results['success'])
        print(f"\n청크 통계:")
        print(f"  총 청크 수: {total_chunks:,}개")
        print(f"  평균 청크/문서: {avg_chunks:.1f}개")
    
    if results['error']:
        print(f"\n실패한 문서 목록:")
        for r in results['error']:
            print(f"  - {r['doc_id'] or r['object_name']}: {r['message']}")
    
    # 로그 저장
    log_file = f"/app/reindex_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "elapsed_seconds": elapsed,
                "total": len(objects),
                "success": len(results['success']),
                "skipped": len(results['skipped']),
                "error": len(results['error']),
                "details": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n상세 로그 저장: {log_file}")
    except Exception as e:
        print(f"\n로그 파일 저장 실패: {e}")
    
    print("=" * 80)
    
    return 0 if not results['error'] else 1


if __name__ == "__main__":
    sys.exit(main())