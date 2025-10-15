# app/services/minio_store.py
from __future__ import annotations

import os
import json
from io import BytesIO
from datetime import timedelta
from typing import Iterable, List, Optional, Union, BinaryIO
from urllib.parse import quote, urlparse, urlunparse, quote_plus

from minio import Minio
from minio.error import S3Error


# ==============================
# Env helpers
# ==============================

def _as_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

def _guess_secure(endpoint: str, secure_env: Optional[bool]) -> bool:
    """endpoint에 스킴이 있으면 그걸로, 없으면 env/default로 HTTPS 여부 결정."""
    if secure_env is not None:
        return secure_env
    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    return parsed.scheme == "https"

def _strip_scheme(endpoint: str) -> str:
    """Minio() 생성자에 넘길 endpoint에서 스킴을 제거해 host:port 형태로 반환."""
    if "://" not in endpoint:
        return endpoint
    parsed = urlparse(endpoint)
    hostport = parsed.netloc or parsed.path
    return hostport.lstrip("/")


# ==============================
# Path conventions (표준 경로)
# ==============================

def std_pdf_key(doc_id: str) -> str:
    return f"uploaded/{doc_id}.pdf"

def orig_folder(doc_id: str) -> str:
    return f"uploaded/originals/{doc_id}/"

def build_orig_key(doc_id: str, original_name: str) -> str:
    return f"{orig_folder(doc_id)}{original_name}"


class MinIOStore:
    """
    MinIO 헬퍼 클래스 (+ 게이트웨이 프록시 대응)
    - 기본 endpoint 결정: 명시 > env > 도커 기본
    - 버킷 이름: MINIO_BUCKET_NAME 또는 MINIO_BUCKET
    - 게이트웨이 프록시 재작성 옵션:
        MINIO_PRESIGN_SCHEME=https|http
        MINIO_PRESIGN_HOST=gw.example.com:443
        MINIO_PRESIGN_PATH_PREFIX=/minio      (optional, ex: Nginx에서 /minio로 프록시)
      → presign_*()가 생성한 URL을 위 규칙으로 재작성하여 프론트에서 바로 사용 가능.
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
        env_secure = os.getenv("MINIO_SECURE")
        secure = _guess_secure(endpoint, secure) if env_secure is None else _as_bool(env_secure)

        # 버킷 이름
        self.bucket = (
            bucket
            or os.getenv("MINIO_BUCKET_NAME")
            or os.getenv("MINIO_BUCKET")
            or "rag-docs"
        )

        # MinIO 클라이언트 (스킴 제거한 host:port 전달)
        self.client = Minio(
            endpoint=_strip_scheme(endpoint),
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

        # 게이트웨이 프록시 재작성 설정
        self.rewrite_scheme = os.getenv("MINIO_PRESIGN_SCHEME")  # "https" / "http"
        self.rewrite_host   = os.getenv("MINIO_PRESIGN_HOST")    # "gw.example.com:443"
        # ex) Nginx 에서 /minio 로 프록시 라우팅하는 경우
        self.rewrite_prefix = os.getenv("MINIO_PRESIGN_PATH_PREFIX", "").rstrip("/")

        self.ensure_bucket()

    # ---------- Bucket helpers ----------
    def ensure_bucket(self) -> None:
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error as e:
            if e.code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise
            
    def _cd_header(self, disposition: str, filename: str) -> str:
        """
        RFC 6266 형식으로 Content-Disposition 값 생성.
        - ASCII 가능: filename="..."
        - 비ASCII: filename*=UTF-8''<percent-encoded>
        """
        try:
            filename.encode("latin-1")
            return f'{disposition}; filename="{filename}"'
        except UnicodeEncodeError:
            return f"{disposition}; filename*=UTF-8''{quote(filename)}"

    # ---------- Object APIs ----------
    def upload(
        self,
        file_path: str,
        object_name: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
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
        metadata: Optional[dict] = None,
    ) -> str:
        """
        바이트/스트림 업로드 (대용량 스트리밍 가능)
        - data: bytes 또는 파일-like 객체
        - length: 알면 지정(성능↑), 모르면 None (bytes라면 자동 산출)
        """
        try:
            meta_kw = {}
            if metadata:
                # MinIO python SDK는 x-amz-meta- 프리픽스 없이 dict를 받음
                meta_kw["metadata"] = metadata

            if isinstance(data, (bytes, bytearray)):
                length = length if length is not None else len(data)
                bio = BytesIO(bytes(data))
                self.client.put_object(
                    self.bucket, object_name, data=bio, length=length,
                    content_type=content_type, **meta_kw
                )
            else:
                if length is None:
                    raise ValueError("스트림 업로드는 length를 지정해야 합니다.")
                self.client.put_object(
                    self.bucket, object_name, data=data, length=length,
                    content_type=content_type, **meta_kw
                )
            return object_name
        except S3Error as e:
            raise RuntimeError(f"MinIO 바이트 업로드 실패: {e}") from e

    def copy_object(self, src_object: str, dst_object: str) -> None:
        """같은 버킷 내 객체 복사 (PDF 원본을 표준 경로로 복사 등)"""
        try:
            self.client.copy_object(
                self.bucket,
                dst_object,
                f"/{self.bucket}/{src_object}",
            )
        except S3Error as e:
            raise RuntimeError(f"MinIO 객체 복사 실패: {e}") from e

    def download(self, object_name: str, target_path: str) -> str:
        try:
            self.client.fget_object(self.bucket, object_name, target_path)
            return target_path
        except S3Error as e:
            raise RuntimeError(f"MinIO 다운로드 실패: {e}") from e

    def get_bytes(self, object_name: str) -> bytes:
        try:
            resp = self.client.get_object(self.bucket, object_name)
            try:
                bio = BytesIO()
                for chunk in resp.stream(32 * 1024):
                    bio.write(chunk)
                return bio.getvalue()
            finally:
                resp.close()
                resp.release_conn()
        except S3Error as e:
            raise RuntimeError(f"MinIO 바이트 다운로드 실패: {e}") from e

    def get_text(self, object_name: str, encoding: str = "utf-8") -> str:
        return self.get_bytes(object_name).decode(encoding, "ignore")

    def get_json(self, object_name: str):
        return json.loads(self.get_text(object_name))

    def exists(self, object_name: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error as e:
            if e.code in {"NoSuchKey", "NoSuchObject"}:
                return False
            raise

    def delete(self, object_name: str) -> None:
        try:
            self.client.remove_object(self.bucket, object_name)
        except S3Error as e:
            raise RuntimeError(f"MinIO 삭제 실패: {e}") from e
        
    def list(self, prefix: str = "") -> List[str]:
        return self.list_files(prefix)

    def list_files(self, prefix: str = "") -> List[str]:
        try:
            objs = self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objs]
        except S3Error as e:
            print(f"❌ MinIO 리스트 조회 오류: {e}")
            return []

    # ---------- Presigned URL 재작성 ----------
    def _rewrite_presigned(self, url: str, *, inline: bool = False, filename: Optional[str] = None) -> str:
        """
        presigned URL의 scheme/host/path를 게이트웨이 프록시에 맞춰 재작성.
        - inline=True  → Content-Disposition: inline
          inline=False → Content-Disposition: attachment
        - filename     → Content-Disposition 파일명
        """
        # Content-Disposition 헤더는 presigned 시점에 반영하는 게 원칙이지만,
        # 이미 get_presigned_url에 넣었어도 브라우저에 따라 안전하게 하기 위해
        # 쿼리 파라미터 기반 재작성만 책임지고, 헤더 전달은 presigned_* 호출부에 맡긴다.
        parsed = urlparse(url)

        scheme = self.rewrite_scheme or parsed.scheme
        netloc = self.rewrite_host   or parsed.netloc

        # path prefix 붙이기 (예: /minio + /bucket/object → /minio/bucket/object)
        path = parsed.path
        if self.rewrite_prefix:
            # prefix는 앞에만 한 번 붙임
            if not path.startswith(self.rewrite_prefix + "/"):
                path = f"{self.rewrite_prefix}{path}"

        # 최종 URL 구성
        rewritten = urlunparse((scheme, netloc, path, parsed.params, parsed.query, parsed.fragment))
        return rewritten

    def presigned_url(
        self,
        object_name: str,
        method: str = "GET",
        expires: timedelta = timedelta(hours=1),
        response_headers: Optional[dict] = None,
    ) -> str:
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

    def presign_download(self, object_name: str, filename: str | None = None, ttl_seconds: int = 3600) -> str:
        headers = None
        if filename:
            headers = {"response-content-disposition": self._cd_header("attachment", filename)}
        url = self.presigned_url(
            object_name,
            method="GET",
            expires=timedelta(seconds=ttl_seconds),
            response_headers=headers,
        )
        return self._rewrite_presigned(url)

    def presign_view(self, object_name: str, filename: str | None = None, ttl_seconds: int = 3600) -> str:
        headers = None
        if filename:
            headers = {"response-content-disposition": self._cd_header("inline", filename)}
        url = self.presigned_url(
            object_name,
            method="GET",
            expires=timedelta(seconds=ttl_seconds),
            response_headers=headers,
        )
        return self._rewrite_presigned(url)
    # ---------- Health ----------
    def healthcheck(self) -> bool:
        try:
            return bool(self.client.bucket_exists(self.bucket))
        except Exception:
            return False

    def size(self, object_name: str) -> int:
        stat = self.client.stat_object(self.bucket, object_name)
        return getattr(stat, "size", 0)

    def put_json(self, object_name: str, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        bio = BytesIO(data)
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=object_name,
            data=bio,
            length=len(data),
            content_type="application/json",
        )

    # ---------- Helpers for originals ----------
    def find_original_for_doc(self, doc_id: str) -> Optional[str]:
        """uploaded/originals/{doc_id}/ 아래 첫 번째 원본 파일 키 반환"""
        prefix = orig_folder(doc_id)
        items = self.list_files(prefix)
        return items[0] if items else None

# 호환용 별칭 (다른 파일에서 MinioStore로 import하는 경우 대비)
MinioStore = MinIOStore
