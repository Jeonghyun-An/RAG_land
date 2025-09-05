# app/services/minio_store.py
from __future__ import annotations

import os
import json
from io import BytesIO
from datetime import timedelta
from typing import Iterable, List, Optional, Union, BinaryIO
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error


def _as_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y"}


def _guess_secure(endpoint: str, secure_env: Optional[bool]) -> bool:
    if secure_env is not None:
        return secure_env
    # endpoint가 스킴을 포함하면 그걸 따르고, 아니면 env 기본값 사용
    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    return parsed.scheme == "https"

def get_bytes(self, object_name: str) -> bytes:
    """MinIO에서 객체를 받아 바로 bytes로 반환"""
    try:
        resp = self.client.get_object(self.bucket, object_name)
        try:
            bio = BytesIO()
            for d in resp.stream(32 * 1024):
                bio.write(d)
            return bio.getvalue()
        finally:
            resp.close()
            resp.release_conn()
    except S3Error as e:
        raise RuntimeError(f"MinIO get_bytes 실패: {e}") from e
def upload_bytes(self, data: bytes, object_name: str, content_type: str | None = None):
    """
    로컬 파일 없이 bytes를 바로 MinIO에 업로드.
    """
    length = len(data)
    bio = BytesIO(data)
    self.client.put_object(
        self.bucket,
        object_name,
        data=bio,
        length=length,
        content_type=content_type or "application/octet-stream",
    )
    return {"bucket": self.bucket, "object_name": object_name, "size": length}
class MinIOStore:
    """
    MinIO 헬퍼 클래스
    - 도커: IS_DOCKER=true면 기본 endpoint를 minio:9000 으로
    - 버킷 이름: MINIO_BUCKET_NAME 또는 MINIO_BUCKET 를 우선순위로 사용
    - 주요 기능: 업/다운로드, 목록, presigned URL, 존재 확인/삭제, 바이트 업로드
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

        # secure 자동 판단 (https://... 면 True)
        secure = _guess_secure(endpoint, secure)

        # 버킷 이름: NAME > BUCKET > 기본
        self.bucket = (
            bucket
            or os.getenv("MINIO_BUCKET_NAME")
            or os.getenv("MINIO_BUCKET")
            or "rag-docs"
        )

        # MinIO 클라이언트
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

        self.ensure_bucket()

    # ---------- Bucket helpers ----------
    def ensure_bucket(self) -> None:
        """버킷이 없으면 생성 (경쟁 상황 안전 처리)"""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error as e:
            if e.code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise

    # ---------- Object APIs ----------
    def upload(
        self,
        file_path: str,
        object_name: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """로컬 파일 업로드 → object_name 반환"""
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

    def upload_bytes(
        self,
        data: Union[bytes, BinaryIO],
        object_name: str,
        content_type: Optional[str] = None,
        length: Optional[int] = None,
    ) -> str:
        """
        바이트/스트림 업로드 (대용량 스트리밍도 가능)
        - data: bytes 또는 파일-like 객체
        - length: 알면 지정(성능↑), 모르면 None (MinIO가 자동 처리 시도)
        """
        try:
            if isinstance(data, (bytes, bytearray)):
                length = length if length is not None else len(data)
                self.client.put_object(
                    self.bucket, object_name, data=data, length=length, content_type=content_type
                )
            else:
                # file-like object
                if length is None:
                    raise ValueError("스트림 업로드는 length를 지정해야 합니다.")
                self.client.put_object(
                    self.bucket, object_name, data=data, length=length, content_type=content_type
                )
            return object_name
        except S3Error as e:
            raise RuntimeError(f"MinIO 바이트 업로드 실패: {e}") from e

    def download(self, object_name: str, target_path: str) -> str:
        """오브젝트 다운로드 → 저장 경로 반환"""
        try:
            self.client.fget_object(self.bucket, object_name, target_path)
            return target_path
        except S3Error as e:
            raise RuntimeError(f"MinIO 다운로드 실패: {e}") from e

    def exists(self, object_name: str) -> bool:
        """오브젝트 존재 여부"""
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error as e:
            if e.code in {"NoSuchKey", "NoSuchObject"}:
                return False
            raise

    def delete(self, object_name: str) -> None:
        """오브젝트 삭제"""
        try:
            self.client.remove_object(self.bucket, object_name)
        except S3Error as e:
            raise RuntimeError(f"MinIO 삭제 실패: {e}") from e

    def list_files(self, prefix: str = "") -> List[str]:
        """버킷 내 파일 목록(object_name 리스트)"""
        try:
            objs = self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objs]
        except S3Error as e:
            print(f"❌ MinIO 리스트 조회 오류: {e}")
            return []

    # ---------- Presigned URLs ----------
    def presigned_url(
        self,
        object_name: str,
        method: str = "GET",
        expires: timedelta = timedelta(hours=1),
        response_headers: Optional[dict] = None,
    ) -> str:
        """
        사전서명 URL 생성
        - method: GET / PUT
        - response_headers: {"response-content-disposition": "attachment; filename=..."} 등
        """
        method = method.upper()
        try:
            if method == "GET":
                return self.client.presigned_get_object(
                    self.bucket, object_name, expires=expires, response_headers=response_headers
                )
            if method == "PUT":
                return self.client.presigned_put_object(
                    self.bucket, object_name, expires=expires
                )
            raise ValueError(f"지원하지 않는 method: {method}")
        except S3Error as e:
            raise RuntimeError(f"MinIO presigned URL 생성 실패: {e}") from e

    # ---------- Health ----------
    def healthcheck(self) -> bool:
        """간단한 헬스체크: 버킷 존재 확인"""
        try:
            return bool(self.client.bucket_exists(self.bucket))
        except Exception:
            return False
        
    def size(self, object_name: str) -> int:
        """객체 크기(바이트) 리턴. 없으면 예외."""
        stat = self.client.stat_object(self.bucket, object_name)
        return getattr(stat, "size", 0)

    def put_json(self, object_name: str, obj: dict) -> None:
        """딕셔너리를 JSON으로 직렬화하여 MinIO에 저장(상태체크 등)"""
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        bio = BytesIO(data)
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=object_name,
            data=bio,
            length=len(data),
            content_type="application/json",
        )
