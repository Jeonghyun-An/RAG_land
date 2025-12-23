#!/usr/bin/env python3
"""
MinIO → Milvus 전체 재인덱싱 스크립트 (독립 실행 버전)

사용법:
    cd /app
    python -m app.scripts.reindex_from_minio [OPTIONS]
    python -m app.scripts.reindex_from_minio --skip-errors
    python -m app.scripts.reindex_from_minio --force --verbose

옵션:
    --dry-run                   실제 처리 없이 목록만 출력
    --limit N                   처리할 문서 개수 제한
    --force                     이미 인덱싱된 문서도 강제 재처리
    --doc-id DOC_ID            특정 문서만 재인덱싱
    --skip-errors              에러 발생 시 계속 진행
    --verbose                  상세 디버깅 로그 출력
"""

import os
import sys
import argparse
import traceback
import tempfile
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
from app.services.file_parser import parse_pdf, parse_pdf_blocks
from app.api.java_router import perform_advanced_chunking, _coerce_chunks_for_milvus, _normalize_pages_for_chunkers


def process_single_document(
    minio: MinIOStore,
    mvs: MilvusStoreV2,
    object_name: str,
    force: bool = False,
    skip_errors: bool = False,
    verbose: bool = False
) -> Dict:
    """단일 문서 재인덱싱 (java_router와 동일한 로직)"""
    result = {
        "doc_id": None,
        "object_name": object_name,
        "status": "error",
        "chunks": 0,
        "message": "",
        "details": {}
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
            result["details"]["step"] = "download"
            return result
        
        print(f"[다운로드] PDF 크기: {len(pdf_bytes):,} bytes")
        result["details"]["pdf_size"] = len(pdf_bytes)
        
        # 4. 임시 파일로 저장 (parse_pdf는 파일 경로 필요)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            
            if verbose:
                print(f"[파싱] 임시 파일 생성: {tmp_path}")
            
            # 5. 텍스트 추출 (OCR 폴백 포함!) - java_router와 동일
            print(f"[파싱] PDF 텍스트 추출 중 (OCR 폴백 지원)...")
            pages = parse_pdf(tmp_path, by_page=True)
            
            if not pages:
                result["message"] = "PDF 파싱 실패 - 텍스트 없음"
                result["details"]["step"] = "parsing"
                result["details"]["pages_raw"] = 0
                return result
            
            print(f"[파싱 완료] 총 {len(pages)} 페이지")
            result["details"]["total_pages"] = len(pages)
            
            # 6. 레이아웃 정보 추출 - java_router와 동일
            blocks_by_page_list = parse_pdf_blocks(tmp_path)
            layout_map = {int(p): blks for p, blks in (blocks_by_page_list or [])}
            print(f"[파싱] 레이아웃 블록: {len(layout_map)} 페이지")
            result["details"]["has_blocks"] = len(layout_map)
            
            # 7. 페이지 정규화 - java_router와 동일
            pages_std = _normalize_pages_for_chunkers(pages)
            
            total_text_len = sum(len(text) for _, text in pages_std)
            print(f"[파싱] 전체 텍스트 길이: {total_text_len:,} 문자")
            result["details"]["total_text_length"] = total_text_len
            
            if verbose:
                print(f"\n[VERBOSE] 페이지별 텍스트 길이:")
                for idx, (page_num, text) in enumerate(pages_std[:5], 1):
                    print(f"  페이지 {page_num}: {len(text):,} 문자")
                    if text:
                        print(f"    샘플: '{text[:100]}...'")
            
            # 8. 고도화된 청킹 - java_router와 동일
            print(f"[청킹] 고도화 청킹 시작...")
            chunks = perform_advanced_chunking(pages_std, layout_map, job_id=f"reindex_{doc_id}")
            
            if not chunks:
                result["message"] = "청킹 결과 없음"
                result["details"]["step"] = "chunking"
                result["details"]["chunks_raw"] = 0
                return result
            
            print(f"[청킹 완료] 총 {len(chunks)} 개 청크 생성")
            result["details"]["chunks_raw"] = len(chunks)
            
            if verbose:
                print(f"\n[VERBOSE] 청킹 결과 샘플 (처음 3개):")
                for idx, chunk in enumerate(chunks[:3]):
                    if isinstance(chunk, (list, tuple)) and len(chunk) >= 2:
                        text, meta = chunk[0], chunk[1]
                        print(f"  청크 #{idx}:")
                        print(f"    텍스트: '{str(text)[:100]}...'")
                        print(f"    메타: {meta}")
            
            # 9. 청크 정규화 - java_router와 동일
            chunks = _coerce_chunks_for_milvus(chunks)
            
            if not chunks:
                result["message"] = "정규화 후 청크 없음"
                result["details"]["step"] = "normalization"
                result["details"]["chunks_normalized"] = 0
                return result
            
            print(f"[정규화] {len(chunks)} 개 청크")
            result["details"]["chunks_normalized"] = len(chunks)
            
        finally:
            # 임시 파일 삭제
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    if verbose:
                        print(f"[파싱] 임시 파일 삭제: {tmp_path}")
                except Exception as e:
                    print(f"[WARN] 임시 파일 삭제 실패: {e}")
        
        # 10. 기존 청크 삭제
        print(f"[Milvus] 기존 청크 삭제 중...")
        try:
            deleted = mvs._delete_by_doc_id(doc_id)
            print(f"[Milvus] 기존 청크 {deleted}개 삭제됨")
            result["details"]["chunks_deleted"] = deleted
        except Exception as e:
            print(f"[WARN] 기존 청크 삭제 실패: {e}")
            result["details"]["delete_error"] = str(e)
        
        # 11. 환경변수 임시 비활성화 (중복 체크 우회)
        old_dedup = os.environ.get("RAG_DEDUP_MANIFEST")
        old_skip = os.environ.get("RAG_SKIP_IF_EXISTS")
        old_replace = os.environ.get("RAG_REPLACE_DOC")
        
        try:
            # 재인덱싱 모드: 중복 체크 완전 비활성화
            os.environ["RAG_DEDUP_MANIFEST"] = "0"
            os.environ["RAG_SKIP_IF_EXISTS"] = "0"
            os.environ["RAG_REPLACE_DOC"] = "0"
            
            # 12. 임베딩 + Milvus 삽입
            print(f"[Milvus] 임베딩 및 삽입 중...")
            insert_result = mvs.insert(
                doc_id=doc_id,
                chunks=chunks,
                embed_fn=embed
            )
            
        finally:
            # 환경변수 복원
            if old_dedup is not None:
                os.environ["RAG_DEDUP_MANIFEST"] = old_dedup
            else:
                os.environ.pop("RAG_DEDUP_MANIFEST", None)
            
            if old_skip is not None:
                os.environ["RAG_SKIP_IF_EXISTS"] = old_skip
            else:
                os.environ.pop("RAG_SKIP_IF_EXISTS", None)
                
            if old_replace is not None:
                os.environ["RAG_REPLACE_DOC"] = old_replace
            else:
                os.environ.pop("RAG_REPLACE_DOC", None)
        
        inserted_count = insert_result.get("inserted", 0)
        
        if inserted_count == 0:
            result["message"] = f"Milvus 삽입 실패 - {insert_result.get('reason', 'unknown')}"
            result["details"]["step"] = "insertion"
            result["details"]["insert_result"] = insert_result
            
            if verbose:
                print(f"\n[ERROR] Milvus 삽입 결과:")
                print(f"  {json.dumps(insert_result, indent=2, ensure_ascii=False)}")
            
            return result
        
        print(f"[완료] {inserted_count}개 청크 인덱싱 완료")
        
        result["status"] = "success"
        result["chunks"] = inserted_count
        result["message"] = f"{inserted_count}개 청크 인덱싱 완료"
        result["details"]["insert_result"] = insert_result
        
        return result
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        result["message"] = error_msg
        result["status"] = "error"
        result["details"]["exception"] = traceback.format_exc()
        
        print(f"\n[ERROR] {doc_id or object_name}")
        print(f"  에러: {error_msg}")
        
        if verbose or not skip_errors:
            print(f"\n상세 트레이스:")
            traceback.print_exc()
        
        return result


def main():
    parser = argparse.ArgumentParser(description="MinIO → Milvus 재인덱싱 (OCR 폴백 지원)")
    parser.add_argument("--dry-run", action="store_true", help="목록만 출력")
    parser.add_argument("--limit", type=int, help="처리할 문서 개수 제한")
    parser.add_argument("--force", action="store_true", help="강제 재처리")
    parser.add_argument("--doc-id", help="특정 문서만")
    parser.add_argument("--skip-errors", action="store_true", help="에러 시 계속 진행")
    parser.add_argument("--verbose", action="store_true", help="상세 디버깅 로그")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MinIO → Milvus 전체 재인덱싱 (OCR 폴백 지원)")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.verbose:
        print("모드: VERBOSE (상세 디버깅)")
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
            skip_errors=args.skip_errors,
            verbose=args.verbose
        )
        
        status = result["status"]
        results[status].append(result)
        
        if status == "success":
            print(f"  성공: {result['chunks']}개 청크")
        elif status == "skipped":
            print(f"  스킵: {result['message']}")
        else:
            print(f"  실패: {result['message']}")
            if args.verbose and result.get("details"):
                print(f"  상세 정보: {json.dumps(result['details'], indent=4, ensure_ascii=False)}")
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
    print(f"  스킵: {len(results['skipped'])}개")
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