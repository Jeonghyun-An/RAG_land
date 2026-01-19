# # #!/usr/bin/env python3
# # """
# # MinIO에서 uploaded/sc/*.pdf 파일들을 uploaded/*.pdf로 이동하고
# # meta.json의 pdf_key도 업데이트
# # (get_bytes + upload_bytes 방식 사용)
# # """
# # import os
# # import sys

# # sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # from app.services.minio_store import MinIOStore

# # def META_KEY(doc_id: str) -> str:
# #     return f"uploaded/__meta__/{doc_id}/meta.json"

# # def move_sc_files_with_meta():
# #     """uploaded/sc/ 파일들을 uploaded/로 이동하고 meta.json 업데이트"""
# #     try:
# #         m = MinIOStore()
        
# #         if not m.healthcheck():
# #             print(" MinIO 연결 실패")
# #             return 1
        
# #         print(" MinIO 연결 성공")
# #         print()
        
# #         # uploaded/sc/ 파일 검색
# #         prefix = "uploaded/sc/"
# #         print(f" {prefix} 경로의 파일 검색 중...")
        
# #         files = m.list_files(prefix)
        
# #         if not files:
# #             print(f" {prefix} 경로에 파일이 없습니다.")
# #             return 0
        
# #         print(f"발견된 파일: {len(files)}개")
# #         print()
        
# #         # 이동 계획
# #         move_plan = []
# #         for src in files:
# #             filename = src.replace(prefix, "")
# #             dst = f"uploaded/{filename}"
            
# #             # doc_id 추출 (754.pdf -> 754)
# #             doc_id = filename.replace(".pdf", "")
            
# #             move_plan.append({
# #                 "src": src,
# #                 "dst": dst,
# #                 "doc_id": doc_id,
# #                 "filename": filename
# #             })
            
# #             print(f"  [{doc_id}] {src} → {dst}")
        
# #         print()
# #         print(f" 총 {len(move_plan)}개 파일을 이동하고 meta.json을 업데이트합니다.")
        
# #         confirm = input("계속하시겠습니까? (yes/no): ").strip().lower()
        
# #         if confirm != "yes":
# #             print(" 이동 취소됨")
# #             return 0
        
# #         print()
# #         print(" 파일 이동 및 메타 업데이트 중...")
# #         print()
        
# #         success_count = 0
# #         failed_count = 0
        
# #         for item in move_plan:
# #             src = item["src"]
# #             dst = item["dst"]
# #             doc_id = item["doc_id"]
            
# #             try:
# #                 # 1. 파일 다운로드 (bytes)
# #                 print(f"   [{doc_id}] 다운로드: {src}")
# #                 file_bytes = m.get_bytes(src)
                
# #                 # 2. 새 경로에 업로드
# #                 print(f"   [{doc_id}] 업로드: {dst}")
# #                 m.upload_bytes(
# #                     data=file_bytes,
# #                     object_name=dst,
# #                     content_type="application/pdf",
# #                     length=len(file_bytes)
# #                 )
# #                 print(f"  ✓ [{doc_id}] 이동 완료")
                
# #                 # 3. meta.json 업데이트
# #                 meta_key = META_KEY(doc_id)
# #                 meta = {}
                
# #                 try:
# #                     if m.exists(meta_key):
# #                         meta = m.get_json(meta_key) or {}
# #                         old_pdf_key = meta.get("pdf_key")
# #                         print(f"  ⏳ [{doc_id}] meta.json 업데이트 중...")
# #                 except:
# #                     old_pdf_key = None
                
# #                 # pdf_key 업데이트
# #                 meta["pdf_key"] = dst
# #                 meta["object_key"] = dst  # backward compat
                
# #                 m.put_json(meta_key, meta)
# #                 if old_pdf_key:
# #                     print(f"  ✓ [{doc_id}] meta: {old_pdf_key} → {dst}")
# #                 else:
# #                     print(f"  ✓ [{doc_id}] meta 생성: {dst}")
                
# #                 # 4. 원본 삭제
# #                 print(f"   [{doc_id}] 원본 삭제: {src}")
# #                 m.delete(src)
# #                 print(f"  ✓ [{doc_id}] 삭제 완료")
# #                 print()
                
# #                 success_count += 1
                
# #             except Exception as e:
# #                 failed_count += 1
# #                 print(f"  ✗ [{doc_id}] 실패: {src}")
# #                 print(f"    오류: {e}")
# #                 print()
        
# #         print("=" * 60)
# #         print(f" 이동 완료: {success_count}개")
        
# #         if failed_count > 0:
# #             print(f"  실패: {failed_count}개")
        
# #         return 0
        
# #     except Exception as e:
# #         print(f" 오류 발생: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         return 1

# # if __name__ == "__main__":
# #     exit_code = move_sc_files_with_meta()
# #     sys.exit(exit_code)
    
#     #!/usr/bin/env python3
# """
# SC 문서들의 meta.json에서 title의 sc/ 경로 제거
# """
# import os
# import sys

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from app.services.minio_store import MinIOStore

# def META_KEY(doc_id: str) -> str:
#     return f"uploaded/__meta__/{doc_id}/meta.json"

# def fix_sc_titles():
#     """SC 문서들의 meta.json에서 title 수정"""
#     try:
#         m = MinIOStore()
        
#         if not m.healthcheck():
#             print("❌ MinIO 연결 실패")
#             return 1
        
#         print("✅ MinIO 연결 성공")
#         print()
        
#         # SC 문서 ID들
#         sc_doc_ids = ["681", "747", "753", "754", "743", "829", "830"]
        
#         print("📝 SC 문서 title 수정 중...")
#         print("=" * 80)
#         print()
        
#         fixed_count = 0
#         skip_count = 0
#         error_count = 0
        
#         for doc_id in sc_doc_ids:
#             meta_key = META_KEY(doc_id)
            
#             try:
#                 if not m.exists(meta_key):
#                     print(f"⚠️  [{doc_id}] meta.json이 없습니다")
#                     skip_count += 1
#                     continue
                
#                 meta = m.get_json(meta_key)
                
#                 old_title = meta.get("title", "")
#                 old_pdf_key = meta.get("pdf_key", "")
                
#                 # title에서 sc/ 제거
#                 new_title = old_title.replace("sc/", "")
                
#                 # pdf_key에서도 sc/ 제거 (혹시 몰라서)
#                 new_pdf_key = old_pdf_key.replace("uploaded/sc/", "uploaded/")
                
#                 # 변경사항이 있으면 업데이트
#                 if new_title != old_title or new_pdf_key != old_pdf_key:
#                     meta["title"] = new_title
#                     meta["pdf_key"] = new_pdf_key
#                     meta["object_key"] = new_pdf_key  # backward compat
                    
#                     m.put_json(meta_key, meta)
                    
#                     print(f"✓ [{doc_id}] 수정 완료")
#                     print(f"   title: {old_title} → {new_title}")
#                     if old_pdf_key != new_pdf_key:
#                         print(f"   pdf_key: {old_pdf_key} → {new_pdf_key}")
#                     print()
                    
#                     fixed_count += 1
#                 else:
#                     print(f"○ [{doc_id}] 수정 불필요")
#                     print(f"   title: {old_title}")
#                     print()
#                     skip_count += 1
                
#             except Exception as e:
#                 print(f"✗ [{doc_id}] 오류: {e}")
#                 print()
#                 error_count += 1
        
#         print("=" * 80)
#         print(f"✅ 수정 완료: {fixed_count}개")
#         print(f"○ 변경 불필요: {skip_count}개")
#         if error_count > 0:
#             print(f"✗ 오류: {error_count}개")
        
#         return 0
        
#     except Exception as e:
#         print(f"❌ 오류 발생: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1

# if __name__ == "__main__":
#     exit_code = fix_sc_titles()
#     sys.exit(exit_code)

#!/usr/bin/env python3
# """
# Milvus에 저장된 SC 문서들의 doc_id를 sc/754 → 754로 수정
# MilvusStoreV2 사용
# """
# import os
# import sys

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from app.services.milvus_store_v2 import MilvusStoreV2
# from app.services.embedding_model import get_embedding_model

# def fix_milvus_sc_docids():
#     """Milvus doc_id에서 sc/ 제거"""
#     try:
#         # MilvusStoreV2 초기화
#         embed_model = get_embedding_model()
#         dim = embed_model.get_sentence_embedding_dimension()
#         mvs = MilvusStoreV2(dim=dim)
        
#         print("✅ Milvus 연결 성공")
#         print(f"   Collection: {mvs.collection_name}")
#         print()
        
#         # SC 문서 ID들 (sc/ 형식)
#         old_doc_ids = ["sc/681", "sc/747", "sc/753", "sc/754", "sc/743", "sc/829", "sc/830"]
        
#         print("🔍 Milvus에서 SC 문서 검색 중...")
#         print("=" * 80)
#         print()
        
#         # 실제 존재하는 doc_id 확인
#         existing_docs = {}
        
#         for old_id in old_doc_ids:
#             try:
#                 expr = f'doc_id == "{old_id}"'
#                 results = mvs.col.query(
#                     expr=expr,
#                     output_fields=["doc_id"],
#                     limit=1
#                 )
                
#                 if results:
#                     new_id = old_id.replace("sc/", "")
#                     existing_docs[old_id] = new_id
#                     print(f"✓ 발견: {old_id} → {new_id}로 변경 예정")
                
#             except Exception as e:
#                 print(f"✗ 오류: {old_id} - {e}")
        
#         if not existing_docs:
#             print("✅ 수정할 SC 문서가 없습니다.")
#             return 0
        
#         print()
#         print(f"⚠️  총 {len(existing_docs)}개 문서의 doc_id를 수정합니다:")
#         for old_id, new_id in existing_docs.items():
#             print(f"   {old_id} → {new_id}")
        
#         print()
#         confirm = input("계속하시겠습니까? (yes/no): ").strip().lower()
        
#         if confirm != "yes":
#             print("❌ 수정 취소됨")
#             return 0
        
#         print()
#         print("🔧 doc_id 수정 중...")
#         print()
        
#         success_count = 0
#         failed_count = 0
        
#         for old_id, new_id in existing_docs.items():
#             try:
#                 print(f"  ⏳ [{old_id}] 처리 중...")
                
#                 # 1. 기존 청크들 가져오기
#                 expr = f'doc_id == "{old_id}"'
#                 chunks = mvs.col.query(
#                     expr=expr,
#                     output_fields=["id", "doc_id", "seq", "page", "section", "chunk", "embedding"],
#                     limit=10000
#                 )
                
#                 if not chunks:
#                     print(f"  ○ [{old_id}] 청크가 없습니다")
#                     continue
                
#                 print(f"  ⏳ [{old_id}] {len(chunks)}개 청크 발견")
                
#                 # 2. doc_id 수정
#                 for chunk in chunks:
#                     chunk['doc_id'] = new_id
                
#                 # 3. 기존 청크 삭제
#                 print(f"  ⏳ [{old_id}] 기존 청크 삭제 중...")
#                 deleted = mvs._delete_by_doc_id(old_id)
#                 print(f"  ✓ [{old_id}] {deleted}개 청크 삭제됨")
                
#                 # 4. 새 doc_id로 재삽입
#                 print(f"  ⏳ [{new_id}] 새로운 청크 삽입 중...")
                
#                 # 데이터 준비 (MilvusStoreV2 스키마에 맞춤)
#                 insert_data = []
#                 for chunk in chunks:
#                     # id는 auto_id이므로 제외
#                     insert_data.append({
#                         "doc_id": chunk['doc_id'],
#                         "seq": chunk['seq'],
#                         "page": chunk['page'],
#                         "section": chunk.get('section', ''),
#                         "chunk": chunk['chunk'],
#                         "embedding": chunk['embedding']
#                     })
                
#                 # Milvus에 직접 삽입 (insert 메서드 사용)
#                 # MilvusStoreV2는 dict list를 받아야 하므로 변환
#                 doc_ids = [d['doc_id'] for d in insert_data]
#                 seqs = [d['seq'] for d in insert_data]
#                 pages = [d['page'] for d in insert_data]
#                 sections = [d['section'] for d in insert_data]
#                 chunks_text = [d['chunk'] for d in insert_data]
#                 embeddings = [d['embedding'] for d in insert_data]
                
#                 mvs.col.insert([doc_ids, seqs, pages, sections, chunks_text, embeddings])
#                 mvs.col.flush()
                
#                 print(f"  ✓ [{new_id}] {len(chunks)}개 청크 삽입됨")
#                 print()
                
#                 success_count += 1
                
#             except Exception as e:
#                 failed_count += 1
#                 print(f"  ✗ [{old_id}] 실패: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 print()
        
#         print("=" * 80)
#         print(f"✅ 수정 완료: {success_count}개")
        
#         if failed_count > 0:
#             print(f"⚠️  실패: {failed_count}개")
        
#         return 0
        
#     except Exception as e:
#         print(f"❌ 오류 발생: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1

# if __name__ == "__main__":
#     exit_code = fix_milvus_sc_docids()
#     sys.exit(exit_code)

#!/usr/bin/env python3
"""
MinIO meta.json 경로 이동: uploaded/__meta__/sc/754/ → uploaded/__meta__/754/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.minio_store import MinIOStore

def move_sc_meta_files():
    """SC 문서들의 meta.json 경로 이동"""
    try:
        m = MinIOStore()
        
        if not m.healthcheck():
            print("❌ MinIO 연결 실패")
            return 1
        
        print("✅ MinIO 연결 성공")
        print()
        
        # SC 경로의 meta.json 파일들 찾기
        print("🔍 SC 경로의 meta.json 검색 중...")
        all_meta_files = m.list_files("uploaded/__meta__/")
        sc_meta_files = [f for f in all_meta_files if '/sc/' in f]
        
        if not sc_meta_files:
            print("✅ 이동할 meta.json이 없습니다.")
            return 0
        
        print(f"발견된 파일: {len(sc_meta_files)}개")
        print("=" * 80)
        print()
        
        # 이동 계획 출력
        move_plan = []
        for src in sc_meta_files:
            # uploaded/__meta__/sc/754/meta.json → uploaded/__meta__/754/meta.json
            parts = src.split('/')
            if len(parts) >= 5 and parts[2] == 'sc':
                doc_id = parts[3]  # 754
                dst = f"uploaded/__meta__/{doc_id}/meta.json"
                move_plan.append((src, dst, doc_id))
                print(f"  [{doc_id}] {src}")
                print(f"       → {dst}")
                print()
        
        if not move_plan:
            print("⚠️  이동 가능한 파일이 없습니다.")
            return 0
        
        print("=" * 80)
        print(f"총 {len(move_plan)}개 파일을 이동합니다.")
        print()
        
        confirm = input("계속하시겠습니까? (yes/no): ").strip().lower()
        
        if confirm != "yes":
            print("❌ 이동 취소됨")
            return 0
        
        print()
        print("📦 파일 이동 중...")
        print()
        
        success_count = 0
        failed_count = 0
        
        for src, dst, doc_id in move_plan:
            try:
                print(f"  ⏳ [{doc_id}] 처리 중...")
                
                # 1. meta.json 읽기
                if not m.exists(src):
                    print(f"  ⚠️  [{doc_id}] 파일이 없습니다: {src}")
                    failed_count += 1
                    continue
                
                meta = m.get_json(src)
                
                # 2. meta.json 내용 수정 (doc_id에서 sc/ 제거)
                if 'doc_id' in meta:
                    old_doc_id = meta['doc_id']
                    new_doc_id = str(old_doc_id).replace('sc/', '').replace('SC/', '')
                    meta['doc_id'] = new_doc_id
                    print(f"  ✓ [{doc_id}] doc_id 수정: {old_doc_id} → {new_doc_id}")
                
                # pdf_key도 수정
                if 'pdf_key' in meta:
                    old_pdf = meta['pdf_key']
                    new_pdf = old_pdf.replace('uploaded/sc/', 'uploaded/')
                    meta['pdf_key'] = new_pdf
                    if old_pdf != new_pdf:
                        print(f"  ✓ [{doc_id}] pdf_key 수정: {old_pdf} → {new_pdf}")
                
                # object_key도 수정
                if 'object_key' in meta:
                    meta['object_key'] = meta['pdf_key']
                
                # 3. 새 경로에 저장
                m.put_json(dst, meta)
                print(f"  ✓ [{doc_id}] 새 경로에 저장: {dst}")
                
                # 4. 원본 삭제
                m.delete(src)
                print(f"  ✓ [{doc_id}] 원본 삭제: {src}")
                print()
                
                success_count += 1
                
            except Exception as e:
                failed_count += 1
                print(f"  ✗ [{doc_id}] 실패: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        print("=" * 80)
        print(f"✅ 이동 완료: {success_count}개")
        
        if failed_count > 0:
            print(f"⚠️  실패: {failed_count}개")
        
        print()
        print("🔄 브라우저를 강력 새로고침(Ctrl+Shift+R) 해주세요!")
        
        return 0
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = move_sc_meta_files()
    sys.exit(exit_code)