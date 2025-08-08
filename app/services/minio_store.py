from minio import Minio
import os
from minio.error import S3Error

class MinIOStore:
    def __init__(self):
        self.client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False,
        )
        self.bucket = "rag-docs"
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload(self, file_path: str, object_name: str = None):
        object_name = object_name or os.path.basename(file_path)
        self.client.fput_object(
            self.bucket, object_name, file_path
        )
        return f"Uploaded to MinIO: {object_name}"

    def download(self, object_name: str, target_path: str):
        self.client.fget_object(self.bucket, object_name, target_path)
        return f"Downloaded: {object_name}"
    
    def list_files(self, prefix: str = "") -> list[str]:
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            print(f"❌ MinIO 리스트 조회 오류: {e}")
            return []
