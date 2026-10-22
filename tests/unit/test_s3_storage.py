"""Unit tests for S3 storage engine and storage provider factory.

All AWS S3 interactions are mocked using unittest.mock to ensure no live AWS network
calls or real credentials are ever required during testing.
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest

from ragforge.core.config import Settings, StorageBackendType
from ragforge.services.storage import (
    LocalFilesystemStorage,
    S3Storage,
    StorageError,
    StorageFileNotFoundError,
    StorageSecurityError,
)
from ragforge.services.storage_provider import create_storage_engine


async def _async_chunks(data: bytes, chunk_size: int = 16) -> AsyncIterator[bytes]:
    """Helper to convert bytes into an async generator of chunks."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


@pytest.fixture
def mock_s3_client() -> MagicMock:
    """Mock boto3 S3 client."""
    return MagicMock()


@pytest.fixture
def s3_storage(mock_s3_client: MagicMock) -> S3Storage:
    """Instantiate S3Storage with a mocked S3 client."""
    with patch("boto3.client", return_value=mock_s3_client):
        storage = S3Storage(
            bucket_name="sociofitt-community",
            region_name="ap-south-1",
            prefix="prod/staging/posts/572/",
            aws_access_key_id="mock_key",
            aws_secret_access_key="mock_secret",
        )
        storage._client = mock_s3_client
        return storage


@pytest.mark.asyncio
async def test_s3_storage_save_success(s3_storage: S3Storage, mock_s3_client: MagicMock) -> None:
    """Test saving a file stream to S3 successfully with correct bucket and key."""
    mock_s3_client.put_object.return_value = {"ETag": '"test-etag"'}
    content = b"Sample S3 upload payload content"
    key = "prod/staging/posts/572/doc-uuid-123.pdf"

    written = await s3_storage.save(key, _async_chunks(content))
    assert written == len(content)

    mock_s3_client.put_object.assert_called_once()
    call_kwargs = mock_s3_client.put_object.call_args.kwargs
    assert call_kwargs["Bucket"] == "sociofitt-community"
    assert call_kwargs["Key"] == key
    assert call_kwargs["Body"] == content
    assert call_kwargs["ContentType"] == "application/pdf"


@pytest.mark.asyncio
async def test_s3_storage_save_client_error(
    s3_storage: S3Storage, mock_s3_client: MagicMock
) -> None:
    """Test handling of ClientError during put_object."""
    error_response = {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}
    mock_s3_client.put_object.side_effect = botocore.exceptions.ClientError(
        error_response, "PutObject"
    )

    with pytest.raises(StorageError) as exc_info:
        await s3_storage.save("prod/staging/posts/572/doc.pdf", _async_chunks(b"data"))
    assert "Failed to put object in S3" in str(exc_info.value)


@pytest.mark.asyncio
async def test_s3_storage_read_bytes_success(
    s3_storage: S3Storage, mock_s3_client: MagicMock
) -> None:
    """Test reading raw bytes from S3 object."""
    mock_body = MagicMock()
    content = b"Retrieved S3 document bytes"
    mock_body.read.return_value = content
    mock_s3_client.get_object.return_value = {"Body": mock_body}

    key = "prod/staging/posts/572/test.txt"
    result = await s3_storage.read_bytes(key)

    assert result == content
    mock_s3_client.get_object.assert_called_once_with(Bucket="sociofitt-community", Key=key)


@pytest.mark.asyncio
async def test_s3_storage_read_missing_key_raises_not_found(
    s3_storage: S3Storage, mock_s3_client: MagicMock
) -> None:
    """Test reading non-existent key raises StorageFileNotFoundError."""
    error_response = {"Error": {"Code": "NoSuchKey", "Message": "Key does not exist"}}
    mock_s3_client.get_object.side_effect = botocore.exceptions.ClientError(
        error_response, "GetObject"
    )

    with pytest.raises(StorageFileNotFoundError):
        await s3_storage.read_bytes("prod/staging/posts/572/missing.txt")


@pytest.mark.asyncio
async def test_s3_storage_open_stream_success(
    s3_storage: S3Storage, mock_s3_client: MagicMock
) -> None:
    """Test streaming chunks from S3 object."""
    data = b"ChunkA" + b"ChunkB"
    mock_stream = MagicMock()
    mock_stream.read.side_effect = [b"ChunkA", b"ChunkB", b""]
    mock_s3_client.get_object.return_value = {"Body": mock_stream}

    key = "prod/staging/posts/572/stream.bin"
    stream = await s3_storage.open_stream(key, chunk_size=6)
    collected = bytearray()
    async for chunk in stream:
        collected.extend(chunk)

    assert bytes(collected) == data


@pytest.mark.asyncio
async def test_s3_storage_exists(s3_storage: S3Storage, mock_s3_client: MagicMock) -> None:
    """Test exists check using head_object."""
    # Exists: head_object succeeds
    mock_s3_client.head_object.return_value = {"ContentLength": 100}
    assert await s3_storage.exists("prod/staging/posts/572/exists.pdf") is True

    # Not exists: 404 NoSuchKey
    error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
    mock_s3_client.head_object.side_effect = botocore.exceptions.ClientError(
        error_response, "HeadObject"
    )
    assert await s3_storage.exists("prod/staging/posts/572/not_there.pdf") is False


@pytest.mark.asyncio
async def test_s3_storage_delete_existing_and_missing(
    s3_storage: S3Storage, mock_s3_client: MagicMock
) -> None:
    """Test delete returns True if file existed, False if not."""
    # When file exists:
    mock_s3_client.head_object.return_value = {"ContentLength": 50}
    mock_s3_client.delete_object.return_value = {}

    deleted = await s3_storage.delete("prod/staging/posts/572/delete_me.pdf")
    assert deleted is True
    mock_s3_client.delete_object.assert_called_once_with(
        Bucket="sociofitt-community", Key="prod/staging/posts/572/delete_me.pdf"
    )

    # When file does not exist:
    mock_s3_client.head_object.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "HeadObject"
    )
    deleted_missing = await s3_storage.delete("prod/staging/posts/572/already_gone.pdf")
    assert deleted_missing is False


@pytest.mark.asyncio
async def test_s3_storage_get_metadata(s3_storage: S3Storage, mock_s3_client: MagicMock) -> None:
    """Test retrieving S3 object metadata."""
    now = datetime.now(UTC)
    mock_s3_client.head_object.return_value = {
        "ContentLength": 2048,
        "LastModified": now,
        "ContentType": "application/pdf",
    }

    meta = await s3_storage.get_metadata("prod/staging/posts/572/doc.pdf")
    assert meta.storage_key == "prod/staging/posts/572/doc.pdf"
    assert meta.size_bytes == 2048
    assert meta.modified_at == now


@pytest.mark.asyncio
async def test_s3_storage_presigned_url(s3_storage: S3Storage, mock_s3_client: MagicMock) -> None:
    """Test generation of short-lived presigned download URL."""
    fake_presigned = "https://sociofitt-community.s3.ap-south-1.amazonaws.com/prod/staging/posts/572/test.pdf?X-Amz-Signature=xyz"
    mock_s3_client.head_object.return_value = {"ContentLength": 100}
    mock_s3_client.generate_presigned_url.return_value = fake_presigned

    url = await s3_storage.generate_presigned_url(
        "prod/staging/posts/572/test.pdf", expires_in=1800
    )
    assert url == fake_presigned
    mock_s3_client.generate_presigned_url.assert_called_once_with(
        ClientMethod="get_object",
        Params={"Bucket": "sociofitt-community", "Key": "prod/staging/posts/572/test.pdf"},
        ExpiresIn=1800,
    )


@pytest.mark.asyncio
async def test_s3_security_validation(s3_storage: S3Storage) -> None:
    """Test invalid or whitespace-only keys raise StorageSecurityError."""
    with pytest.raises(StorageSecurityError):
        await s3_storage.save("   ", _async_chunks(b"bad"))

    with pytest.raises(StorageSecurityError):
        await s3_storage.read_bytes("")


def test_create_storage_engine_factory() -> None:
    """Test storage provider factory creates correct engine for settings."""
    # 1. Local backend
    local_settings = Settings(STORAGE_BACKEND=StorageBackendType.LOCAL)
    local_engine = create_storage_engine(local_settings)
    assert isinstance(local_engine, LocalFilesystemStorage)

    # 2. S3 backend
    s3_settings = Settings(
        STORAGE_BACKEND=StorageBackendType.S3,
        S3_BUCKET="sociofitt-community",
        S3_REGION="ap-south-1",
        S3_PREFIX="prod/staging/posts/572/",
    )
    with patch("boto3.client"):
        s3_engine = create_storage_engine(s3_settings)
        assert isinstance(s3_engine, S3Storage)
        assert s3_engine.bucket_name == "sociofitt-community"
        assert s3_engine.region_name == "ap-south-1"
        assert s3_engine.prefix == "prod/staging/posts/572/"
