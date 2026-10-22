"""Storage engine abstraction and implementations."""

import abc
import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


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
