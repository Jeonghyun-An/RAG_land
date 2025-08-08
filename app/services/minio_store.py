# app/services/minio_store.py
from __future__ import annotations

import os
from datetime import timedelta
from typing import List, Optional

from minio import Minio
from minio.error import S3Error


def _as_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y"}


class MinIOStore:
    """
    MinIO 헬퍼 클래스
    - 로컬: 기본 localhost:9000
    - Docker: IS_DOCKER=true면 minio:9000 자동 사용
    - 버킷 없으면 자동 생성
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: Optional[bool] = None,
        bucket: Optional[str] = None,
    ) -> None:
        is_docker = _as_bool(os.getenv("IS_DOCKER"), False)

        # 엔드포인트 결정 (명시 > env > 도커/로컬 기본)
        endpoint = (
            endpoint
            or os.getenv("MINIO_ENDPOINT")
            or ("minio:9000" if is_docker else "localhost:9000")
        )

        access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        secure = secure if secure is not None else _as_bool(os.getenv("MINIO_SECURE"), False)
        self.bucket = bucket or os.getenv("MINIO_BUCKET", "rag-docs")

        # 클라이언트 생성
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

        # 버킷 보장 (경쟁상황도 안전하게 처리)
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error as e:
            # 이미 있거나 권한/레이스 케이스는 무시 가능
            if e.code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise

    # ---------- Public APIs ----------

    def upload(self, file_path: str, object_name: Optional[str] = None, content_type: Optional[str] = None) -> str:
        """
        파일을 버킷에 업로드하고 object_name을 반환.
        """
        object_name = object_name or os.path.basename(file_path)
        try:
            self.client.fput_object(
                bucket_name=self.bucket,
                object_name=object_name,
                file_path=file_path,
                content_type=content_type,
            )
            return object_name
        except S3Error as e:
            raise RuntimeError(f"MinIO 업로드 실패: {e}") from e

    def download(self, object_name: str, target_path: str) -> str:
        """
        버킷에서 파일 다운로드 후 저장 경로 반환.
        """
        try:
            self.client.fget_object(self.bucket, object_name, target_path)
            return target_path
        except S3Error as e:
            raise RuntimeError(f"MinIO 다운로드 실패: {e}") from e

    def list_files(self, prefix: str = "") -> List[str]:
        """
        버킷 내 파일 리스트(object_name) 반환.
        """
        try:
            objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            # 필요 시 빈 리스트 대신 예외를 올려도 됨
            print(f"❌ MinIO 리스트 조회 오류: {e}")
            return []

    def presigned_url(self, object_name: str, method: str = "GET", expires: timedelta = timedelta(hours=1)) -> str:
        """
        사전서명 URL 생성 (GET/PUT 등)
        """
        method = method.upper()
        try:
            if method == "GET":
                return self.client.presigned_get_object(self.bucket, object_name, expires=expires)
            if method == "PUT":
                return self.client.presigned_put_object(self.bucket, object_name, expires=expires)
            raise ValueError(f"지원하지 않는 method: {method}")
        except S3Error as e:
            raise RuntimeError(f"MinIO presigned URL 생성 실패: {e}") from e

    def healthcheck(self) -> bool:
        """
        간단한 헬스체크: 버킷 존재 여부 확인
        """
        try:
            return bool(self.client.bucket_exists(self.bucket))
        except Exception:
            return False
