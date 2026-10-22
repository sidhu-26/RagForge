"""Unit tests for storage engine abstraction and LocalFilesystemStorage."""

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from ragforge.services.storage import (
    LocalFilesystemStorage,
    StorageFileNotFoundError,
    StorageSecurityError,
)


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Provide an isolated temporary directory for local filesystem storage."""
    storage_path = tmp_path / "documents"
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path


@pytest.fixture
def storage_engine(temp_storage_dir: Path) -> LocalFilesystemStorage:
    """Provide a LocalFilesystemStorage instance configured with a temporary root."""
    return LocalFilesystemStorage(temp_storage_dir)


async def _async_chunks(data: bytes, chunk_size: int = 16) -> AsyncIterator[bytes]:
    """Helper to convert bytes into an async generator of chunks."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


@pytest.mark.asyncio
async def test_save_and_read_bytes(storage_engine: LocalFilesystemStorage) -> None:
    """Test saving a byte stream and reading it back in full."""
    content = b"Hello, RAGForge local storage!"
    key = "docs/2026/10/test.txt"

    written = await storage_engine.save(key, _async_chunks(content))
    assert written == len(content)

    assert await storage_engine.exists(key) is True
    read_data = await storage_engine.read_bytes(key)
    assert read_data == content


@pytest.mark.asyncio
async def test_open_stream(storage_engine: LocalFilesystemStorage) -> None:
    """Test reading stored file via open_stream generator in chunks."""
    content = b"A" * 1024 + b"B" * 1024
    key = "large/test_stream.bin"

    await storage_engine.save(key, _async_chunks(content))

    stream = await storage_engine.open_stream(key, chunk_size=512)
    collected = bytearray()
    async for chunk in stream:
        collected.extend(chunk)

    assert bytes(collected) == content


@pytest.mark.asyncio
async def test_get_metadata(storage_engine: LocalFilesystemStorage) -> None:
    """Test retrieving file metadata (size and mtime)."""
    content = b"Sample metadata payload"
    key = "meta/test.txt"

    await storage_engine.save(key, _async_chunks(content))
    meta = await storage_engine.get_metadata(key)

    assert meta.storage_key == key
    assert meta.size_bytes == len(content)
    assert meta.modified_at is not None


@pytest.mark.asyncio
async def test_delete(storage_engine: LocalFilesystemStorage) -> None:
    """Test deleting an existing file and deleting a non-existent file."""
    content = b"Delete me soon"
    key = "delete/sample.txt"

    await storage_engine.save(key, _async_chunks(content))
    assert await storage_engine.exists(key) is True

    deleted = await storage_engine.delete(key)
    assert deleted is True
    assert await storage_engine.exists(key) is False

    # Deleting again should return False without raising
    deleted_again = await storage_engine.delete(key)
    assert deleted_again is False


@pytest.mark.asyncio
async def test_read_missing_file_raises_error(storage_engine: LocalFilesystemStorage) -> None:
    """Test that reading a non-existent file raises StorageFileNotFoundError."""
    with pytest.raises(StorageFileNotFoundError):
        await storage_engine.read_bytes("missing/file.txt")

    with pytest.raises(StorageFileNotFoundError):
        await storage_engine.open_stream("missing/file.txt")

    with pytest.raises(StorageFileNotFoundError):
        await storage_engine.get_metadata("missing/file.txt")


@pytest.mark.asyncio
async def test_path_traversal_prevention(storage_engine: LocalFilesystemStorage) -> None:
    """Test that path traversal attempts raise StorageSecurityError."""
    traversal_keys = [
        "../secret.txt",
        "../../etc/passwd",
        "docs/../../passwords",
        "/absolute/path/file.txt",
        "",
        "   ",
    ]

    for bad_key in traversal_keys:
        with pytest.raises(StorageSecurityError):
            await storage_engine.save(bad_key, _async_chunks(b"malicious"))

        with pytest.raises(StorageSecurityError):
            await storage_engine.read_bytes(bad_key)

        with pytest.raises(StorageSecurityError):
            await storage_engine.delete(bad_key)

        assert await storage_engine.exists(bad_key) is False


@pytest.mark.asyncio
async def test_safe_file_permissions(
    storage_engine: LocalFilesystemStorage, temp_storage_dir: Path
) -> None:
    """Verify that stored files are written with restricted permissions (0o640)."""
    key = "secure/file.txt"
    await storage_engine.save(key, _async_chunks(b"confidential"))

    target_path = temp_storage_dir / key
    file_mode = oct(os.stat(target_path).st_mode & 0o777)
    # Mode should be 0o640 (owner rw, group r, others none)
    # Depending on umask, it might mask group write if any
    assert file_mode in ("0o640", "0o600")
