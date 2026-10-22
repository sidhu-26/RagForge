"""Storage engine abstraction and implementations."""

import abc
import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StorageMetadata:
    """Metadata for a stored file object."""

    storage_key: str
    size_bytes: int
    modified_at: datetime


class StorageError(Exception):
    """Base exception for storage operations."""

    pass


class StorageFileNotFoundError(StorageError):
    """Raised when the specified storage key does not exist."""

    pass


class StorageSecurityError(StorageError):
    """Raised when a storage path violation or traversal attempt is detected."""

    pass


class StorageEngine(abc.ABC):
    """Abstract interface defining document storage capabilities.

    Designed to decouple application services and routers from specific physical
    storage implementations (e.g. LocalFilesystemStorage, AzureBlobStorage, S3).
    """

    @abc.abstractmethod
    async def save(
        self,
        storage_key: str,
        stream: AsyncIterator[bytes],
    ) -> int:
        """Stream and persist bytes to the destination under storage_key.

        Args:
            storage_key: Unique relative identifier for the file.
            stream: Asynchronous iterator of byte chunks.

        Returns:
            int: Total number of bytes written.

        Raises:
            StorageSecurityError: If storage_key violates path restrictions.
            StorageError: If writing to destination fails.
        """
        pass

    @abc.abstractmethod
    async def open_stream(
        self,
        storage_key: str,
        chunk_size: int = 64 * 1024,
    ) -> AsyncIterator[bytes]:
        """Stream bytes from storage for the specified storage_key.

        Args:
            storage_key: Unique identifier of the file.
            chunk_size: Size in bytes of each streamed chunk.

        Yields:
            bytes: Next chunk of data.

        Raises:
            StorageFileNotFoundError: If storage_key does not exist.
            StorageSecurityError: If storage_key violates path restrictions.
            StorageError: If reading fails.
        """
        pass

    @abc.abstractmethod
    async def read_bytes(self, storage_key: str) -> bytes:
        """Read and return all bytes for the given storage_key.

        Args:
            storage_key: Unique identifier of the file.

        Returns:
            bytes: Full file payload.

        Raises:
            StorageFileNotFoundError: If storage_key does not exist.
            StorageSecurityError: If storage_key violates path restrictions.
        """
        pass

    @abc.abstractmethod
    async def delete(self, storage_key: str) -> bool:
        """Delete the file associated with storage_key if it exists.

        Args:
            storage_key: Unique identifier of the file.

        Returns:
            bool: True if the file existed and was removed, False if it was not found.

        Raises:
            StorageSecurityError: If storage_key violates path restrictions.
            StorageError: If deletion fails.
        """
        pass

    @abc.abstractmethod
    async def exists(self, storage_key: str) -> bool:
        """Check whether an object exists at storage_key.

        Args:
            storage_key: Unique identifier of the file.

        Returns:
            bool: True if object exists, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def get_metadata(self, storage_key: str) -> StorageMetadata:
        """Retrieve metadata for the object at storage_key.

        Args:
            storage_key: Unique identifier of the file.

        Returns:
            StorageMetadata: Object containing size and last modified date.

        Raises:
            StorageFileNotFoundError: If storage_key does not exist.
        """
        pass

    @abc.abstractmethod
    async def generate_presigned_url(
        self,
        storage_key: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate a short-lived URL for downloading the file.

        Args:
            storage_key: Unique identifier of the file.
            expires_in: URL lifetime in seconds (default: 3600).

        Returns:
            str: Secure URL or file URI.

        Raises:
            StorageFileNotFoundError: If storage_key does not exist.
            StorageError: If URL generation fails.
        """
        pass


class LocalFilesystemStorage(StorageEngine):
    """Local filesystem storage implementation for development and testing.

    Files are stored under a configured base root directory outside publicly served paths.
    Enforces strict path normalization and sandbox verification to prevent path traversal attacks.
    """

    def __init__(self, root_dir: str | Path) -> None:
        """Initialize local filesystem storage with a root directory.

        Args:
            root_dir: Base directory path for storing files.
        """
        self._root_dir = Path(root_dir).resolve()
        # Ensure root directory exists with restricted permissions (drwxr-x---)
        self._root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root_dir(self) -> Path:
        """Resolved root path of the storage repository."""
        return self._root_dir

    def _resolve_safe_path(self, storage_key: str) -> Path:
        """Resolve storage_key against root_dir and verify sandbox confinement.

        Args:
            storage_key: Relative storage identifier.

        Returns:
            Path: Absolute resolved path guaranteed to be strictly within root_dir.

        Raises:
            StorageSecurityError: If path attempts to escape the root directory.
        """
        if not storage_key or not storage_key.strip():
            raise StorageSecurityError("Storage key must not be empty.")

        # Reject absolute paths or paths starting with slash/backslash
        if storage_key.startswith(("/", "\\")):
            raise StorageSecurityError(f"Absolute storage keys are forbidden: '{storage_key}'")

        raw_path = Path(storage_key)
        if raw_path.is_absolute():
            raise StorageSecurityError(f"Absolute storage keys are forbidden: '{storage_key}'")

        # Resolve path relative to root directory
        target_path = (self._root_dir / storage_key).resolve()

        # Enforce containment: target_path must have root_dir as parent/ancestor
        try:
            target_path.relative_to(self._root_dir)
        except ValueError:
            raise StorageSecurityError(
                f"Path traversal detected for storage key: '{storage_key}'"
            ) from None

        return target_path

    async def save(
        self,
        storage_key: str,
        stream: AsyncIterator[bytes],
    ) -> int:
        """Stream chunks from async iterator and write directly to disk safely."""
        target_path = self._resolve_safe_path(storage_key)

        # Create parent directories asynchronously
        parent_dir = target_path.parent
        await asyncio.to_thread(parent_dir.mkdir, parents=True, exist_ok=True)

        bytes_written = 0
        try:
            # Write to file with 0o640 permissions (owner read/write, group read)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

            def _open_file() -> int:
                return os.open(str(target_path), flags, 0o640)

            fd = await asyncio.to_thread(_open_file)

            def _write_chunk(file_desc: int, data: bytes) -> None:
                os.write(file_desc, data)

            def _close_file(file_desc: int) -> None:
                os.close(file_desc)

            try:
                async for chunk in stream:
                    if chunk:
                        await asyncio.to_thread(_write_chunk, fd, chunk)
                        bytes_written += len(chunk)
            finally:
                await asyncio.to_thread(_close_file, fd)

        except StorageSecurityError:
            raise
        except Exception as e:
            # Clean up partial file on write failure
            def _cleanup() -> None:
                if target_path.exists():
                    target_path.unlink(missing_ok=True)

            await asyncio.to_thread(_cleanup)
            if isinstance(e, OSError):
                raise StorageError(f"Failed to write file to local storage: {e}") from e
            raise

        return bytes_written

    async def open_stream(
        self,
        storage_key: str,
        chunk_size: int = 64 * 1024,
    ) -> AsyncIterator[bytes]:
        """Stream chunks of bytes from disk."""
        target_path = self._resolve_safe_path(storage_key)

        def _check_file() -> bool:
            return target_path.is_file()

        is_file = await asyncio.to_thread(_check_file)
        if not is_file:
            raise StorageFileNotFoundError(f"Storage object '{storage_key}' not found.")

        async def _generator() -> AsyncIterator[bytes]:
            def _open_rb() -> int:
                return os.open(str(target_path), os.O_RDONLY)

            fd = await asyncio.to_thread(_open_rb)

            def _read_chunk(file_desc: int, size: int) -> bytes:
                return os.read(file_desc, size)

            def _close_rb(file_desc: int) -> None:
                os.close(file_desc)

            try:
                while True:
                    chunk = await asyncio.to_thread(_read_chunk, fd, chunk_size)
                    if not chunk:
                        break
                    yield chunk
            finally:
                await asyncio.to_thread(_close_rb, fd)

        return _generator()

    async def read_bytes(self, storage_key: str) -> bytes:
        """Read and return complete byte contents from disk."""
        target_path = self._resolve_safe_path(storage_key)

        def _sync_read() -> bytes:
            if not target_path.is_file():
                raise StorageFileNotFoundError(f"Storage object '{storage_key}' not found.")
            return target_path.read_bytes()

        try:
            return await asyncio.to_thread(_sync_read)
        except StorageFileNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to read file from storage: {e}") from e

    async def delete(self, storage_key: str) -> bool:
        """Safely delete file from disk if it exists."""
        target_path = self._resolve_safe_path(storage_key)

        def _sync_delete() -> bool:
            if not target_path.exists():
                return False
            target_path.unlink()
            return True

        try:
            return await asyncio.to_thread(_sync_delete)
        except Exception as e:
            raise StorageError(f"Failed to delete file '{storage_key}': {e}") from e

    async def exists(self, storage_key: str) -> bool:
        """Check if file exists on disk."""
        try:
            target_path = self._resolve_safe_path(storage_key)
            return await asyncio.to_thread(target_path.is_file)
        except StorageSecurityError:
            return False

    async def get_metadata(self, storage_key: str) -> StorageMetadata:
        """Get file stat metadata from disk."""
        target_path = self._resolve_safe_path(storage_key)

        def _sync_stat() -> StorageMetadata:
            if not target_path.is_file():
                raise StorageFileNotFoundError(f"Storage object '{storage_key}' not found.")
            st = target_path.stat()
            mtime = datetime.fromtimestamp(st.st_mtime, tz=UTC)
            return StorageMetadata(
                storage_key=storage_key,
                size_bytes=st.st_size,
                modified_at=mtime,
            )

        return await asyncio.to_thread(_sync_stat)

    async def generate_presigned_url(
        self,
        storage_key: str,
        expires_in: int = 3600,
    ) -> str:
        """Return local file URI for development."""
        _ = expires_in
        target_path = self._resolve_safe_path(storage_key)
        if not await self.exists(storage_key):
            raise StorageFileNotFoundError(f"Storage object '{storage_key}' not found.")
        return target_path.as_uri()


class S3Storage(StorageEngine):
    """Production AWS S3-backed storage engine.

    Uploads, streams, downloads, and removes documents in Amazon S3.
    Integrates with standard AWS credential provider chain while supporting explicit credentials.
    """

    def __init__(
        self,
        bucket_name: str,
        region_name: str = "ap-south-1",
        prefix: str = "prod/staging/posts/572/",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        endpoint_url: str | None = None,
        s3_client: Any = None,
    ) -> None:
        if not bucket_name or not bucket_name.strip():
            raise StorageError("S3 bucket name must not be empty.")

        self._bucket_name = bucket_name.strip()
        self._region_name = region_name.strip()
        self._prefix = prefix.strip()
        if self._prefix and not self._prefix.endswith("/"):
            self._prefix = f"{self._prefix}/"

        self._endpoint_url = endpoint_url
        self._client = s3_client

        if self._client is None:
            import boto3
            from botocore.config import Config

            client_kwargs: dict[str, Any] = {
                "region_name": self._region_name,
                "config": Config(
                    retries={"max_attempts": 3, "mode": "standard"},
                    connect_timeout=10,
                    read_timeout=30,
                ),
            }
            if endpoint_url:
                client_kwargs["endpoint_url"] = endpoint_url
            if aws_access_key_id and aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = aws_access_key_id
                client_kwargs["aws_secret_access_key"] = aws_secret_access_key
                if aws_session_token:
                    client_kwargs["aws_session_token"] = aws_session_token

            self._client = boto3.client("s3", **client_kwargs)

    @property
    def bucket_name(self) -> str:
        return self._bucket_name

    @property
    def region_name(self) -> str:
        return self._region_name

    @property
    def prefix(self) -> str:
        return self._prefix

    def _sanitize_storage_key(self, storage_key: str) -> str:
        """Validate and sanitize S3 object key."""
        clean_key = storage_key.strip()
        if not clean_key:
            raise StorageSecurityError("Storage key must not be empty.")
        if clean_key.startswith("/"):
            clean_key = clean_key.lstrip("/")
        return clean_key

    async def save(
        self,
        storage_key: str,
        stream: AsyncIterator[bytes],
    ) -> int:
        """Upload stream of bytes to S3 using multipart upload or single put."""
        clean_key = self._sanitize_storage_key(storage_key)

        # Consume stream into chunks
        chunks: list[bytes] = []
        total_bytes = 0
        async for chunk in stream:
            if chunk:
                chunks.append(chunk)
                total_bytes += len(chunk)

        body_bytes = b"".join(chunks)

        # Detect ContentType if possible from storage_key extension
        import mimetypes

        content_type, _ = mimetypes.guess_type(clean_key)
        if not content_type:
            content_type = "application/octet-stream"

        def _sync_put() -> None:
            from botocore.exceptions import ClientError

            try:
                self._client.put_object(
                    Bucket=self._bucket_name,
                    Key=clean_key,
                    Body=body_bytes,
                    ContentType=content_type,
                )
            except ClientError as e:
                raise StorageError(f"Failed to put object in S3 ({clean_key}): {e}") from e

        await asyncio.to_thread(_sync_put)
        return total_bytes

    async def open_stream(
        self,
        storage_key: str,
        chunk_size: int = 64 * 1024,
    ) -> AsyncIterator[bytes]:
        """Stream bytes chunk by chunk from S3 object."""
        clean_key = self._sanitize_storage_key(storage_key)

        def _get_streaming_body() -> Any:
            from botocore.exceptions import ClientError

            try:
                response = self._client.get_object(
                    Bucket=self._bucket_name,
                    Key=clean_key,
                )
                return response["Body"]
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "404"):
                    raise StorageFileNotFoundError(f"S3 object '{clean_key}' not found.") from e
                raise StorageError(f"Failed to get S3 object '{clean_key}': {e}") from e

        body = await asyncio.to_thread(_get_streaming_body)

        async def _generator() -> AsyncIterator[bytes]:
            try:
                while True:
                    chunk = await asyncio.to_thread(body.read, chunk_size)
                    if not chunk:
                        break
                    yield chunk
            finally:
                await asyncio.to_thread(body.close)

        return _generator()

    async def read_bytes(self, storage_key: str) -> bytes:
        """Download complete object bytes from S3."""
        clean_key = self._sanitize_storage_key(storage_key)

        def _sync_read() -> bytes:
            from botocore.exceptions import ClientError

            try:
                response = self._client.get_object(
                    Bucket=self._bucket_name,
                    Key=clean_key,
                )
                body_content: bytes = response["Body"].read()
                return body_content

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "404"):
                    raise StorageFileNotFoundError(f"S3 object '{clean_key}' not found.") from e
                raise StorageError(f"Failed to read S3 object '{clean_key}': {e}") from e

        return await asyncio.to_thread(_sync_read)

    async def delete(self, storage_key: str) -> bool:
        """Delete object from S3 if it exists."""
        clean_key = self._sanitize_storage_key(storage_key)

        # Check existence first
        exists = await self.exists(clean_key)
        if not exists:
            return False

        def _sync_delete() -> None:
            from botocore.exceptions import ClientError

            try:
                self._client.delete_object(
                    Bucket=self._bucket_name,
                    Key=clean_key,
                )
            except ClientError as e:
                raise StorageError(f"Failed to delete S3 object '{clean_key}': {e}") from e

        await asyncio.to_thread(_sync_delete)
        return True

    async def exists(self, storage_key: str) -> bool:
        """Check if object exists in S3 via head_object."""
        clean_key = self._sanitize_storage_key(storage_key)

        def _sync_head() -> bool:
            from botocore.exceptions import ClientError

            try:
                self._client.head_object(
                    Bucket=self._bucket_name,
                    Key=clean_key,
                )
                return True
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "404"):
                    return False
                logger.warning(
                    "S3 head_object error checking existence",
                    extra={"key": clean_key, "error": str(e)},
                )
                return False

        return await asyncio.to_thread(_sync_head)

    async def get_metadata(self, storage_key: str) -> StorageMetadata:
        """Get object size and last modified date from S3."""
        clean_key = self._sanitize_storage_key(storage_key)

        def _sync_meta() -> StorageMetadata:
            from botocore.exceptions import ClientError

            try:
                response = self._client.head_object(
                    Bucket=self._bucket_name,
                    Key=clean_key,
                )
                content_length = response.get("ContentLength", 0)
                last_modified = response.get("LastModified", datetime.now(UTC))
                if last_modified.tzinfo is None:
                    last_modified = last_modified.replace(tzinfo=UTC)
                return StorageMetadata(
                    storage_key=clean_key,
                    size_bytes=content_length,
                    modified_at=last_modified,
                )
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "404"):
                    raise StorageFileNotFoundError(f"S3 object '{clean_key}' not found.") from e
                raise StorageError(f"Failed to get S3 object metadata '{clean_key}': {e}") from e

        return await asyncio.to_thread(_sync_meta)

    async def generate_presigned_url(
        self,
        storage_key: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate a short-lived presigned URL for secure download."""
        clean_key = self._sanitize_storage_key(storage_key)

        if not await self.exists(clean_key):
            raise StorageFileNotFoundError(f"S3 object '{clean_key}' not found.")

        def _sync_presign() -> str:
            from botocore.exceptions import ClientError

            try:
                url = self._client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={
                        "Bucket": self._bucket_name,
                        "Key": clean_key,
                    },
                    ExpiresIn=expires_in,
                )
                return str(url)
            except ClientError as e:
                msg = f"Failed to generate presigned URL for '{clean_key}': {e}"
                raise StorageError(msg) from e

        return await asyncio.to_thread(_sync_presign)
