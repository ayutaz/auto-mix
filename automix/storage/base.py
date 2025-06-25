"""
Base storage interface for AutoMix
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO


class StorageBackend(ABC):
    """Base class for storage backends"""

    @abstractmethod
    def save(self, file_path: Path, file_object: BinaryIO) -> str:
        """
        Save a file to storage
        
        Args:
            file_path: Path where file should be saved
            file_object: File-like object to save
            
        Returns:
            URL or path to the saved file
        """
        pass

    @abstractmethod
    def load(self, file_path: str) -> BinaryIO:
        """
        Load a file from storage
        
        Args:
            file_path: Path to the file
            
        Returns:
            File-like object
        """
        pass

    @abstractmethod
    def delete(self, file_path: str) -> bool:
        """
        Delete a file from storage
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """
        Check if a file exists
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file exists
        """
        pass

    @abstractmethod
    def get_url(self, file_path: str) -> str:
        """
        Get a URL for the file
        
        Args:
            file_path: Path to the file
            
        Returns:
            URL to access the file
        """
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage backend"""

    def __init__(self, base_path: str = "storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

    def save(self, file_path: Path, file_object: BinaryIO) -> str:
        """Save file to local filesystem"""
        full_path = self.base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, "wb") as f:
            f.write(file_object.read())
        
        return str(full_path)

    def load(self, file_path: str) -> BinaryIO:
        """Load file from local filesystem"""
        full_path = self.base_path / file_path
        return open(full_path, "rb")

    def delete(self, file_path: str) -> bool:
        """Delete file from local filesystem"""
        full_path = self.base_path / file_path
        if full_path.exists():
            full_path.unlink()
            return True
        return False

    def exists(self, file_path: str) -> bool:
        """Check if file exists in local filesystem"""
        full_path = self.base_path / file_path
        return full_path.exists()

    def get_url(self, file_path: str) -> str:
        """Get local file path as URL"""
        return f"file://{self.base_path / file_path}"


class S3Storage(StorageBackend):
    """AWS S3 / Cloudflare R2 storage backend"""

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
    ):
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
        
        self.bucket_name = bucket_name
        
        # Use environment variables if not provided
        aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL")
        
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
            region_name=region_name,
        )

    def save(self, file_path: Path, file_object: BinaryIO) -> str:
        """Save file to S3"""
        key = str(file_path).replace("\\", "/")  # Ensure forward slashes
        self.s3_client.upload_fileobj(file_object, self.bucket_name, key)
        return self.get_url(key)

    def load(self, file_path: str) -> BinaryIO:
        """Load file from S3"""
        import io
        
        file_object = io.BytesIO()
        self.s3_client.download_fileobj(self.bucket_name, file_path, file_object)
        file_object.seek(0)
        return file_object

    def delete(self, file_path: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=file_path)
            return True
        except Exception:
            return False

    def exists(self, file_path: str) -> bool:
        """Check if file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=file_path)
            return True
        except:
            return False

    def get_url(self, file_path: str) -> str:
        """Get presigned URL for the file"""
        return self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": file_path},
            ExpiresIn=3600,  # 1 hour
        )


def get_storage_backend() -> StorageBackend:
    """
    Get the appropriate storage backend based on environment
    """
    # Check if S3 credentials are available
    if os.getenv("S3_BUCKET_NAME"):
        return S3Storage(
            bucket_name=os.getenv("S3_BUCKET_NAME", ""),
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        )
    
    # Default to local storage
    return LocalStorage()