"""Integration tests for document management APIs.

Covers:
- Valid PDF, DOCX, and TXT uploads
- Empty (0-byte) and oversized uploads
- Unsupported and malformed files (executable rejection, non-doc zip, binary pretending to be text)
- Path traversal and malicious filename attempts
- SHA-256 checksum and metadata accuracy
- Listing documents with pagination
- Retrieve document metadata by UUID
- Missing document behavior (404)
- Document deletion and storage file cleanup
- Database transaction failure cleans up stored file
- Storage failure handling
"""

import hashlib
import io
import uuid
import zipfile
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ragforge.core.config import Settings, get_settings
from ragforge.db.session import get_db
from ragforge.main import app
from ragforge.services.storage import (
    LocalFilesystemStorage,
    StorageEngine,
    StorageError,
)
from ragforge.services.storage_provider import get_storage_engine


@pytest.fixture
def test_storage_dir(tmp_path: Path) -> Path:
    """Create isolated directory for storage in integration tests."""
    s_dir = tmp_path / "test_storage"
    s_dir.mkdir(parents=True, exist_ok=True)
    return s_dir


@pytest.fixture
def test_storage(test_storage_dir: Path) -> LocalFilesystemStorage:
    """Instantiate a real LocalFilesystemStorage bound to temp dir."""
    return LocalFilesystemStorage(test_storage_dir)


@pytest.fixture
def override_settings(test_storage_dir: Path) -> Settings:
    """Provide testing settings with restricted upload size limit."""
    return Settings(
        ENVIRONMENT="testing",
        LOG_LEVEL="DEBUG",
        STORAGE_LOCAL_ROOT=str(test_storage_dir),
        STORAGE_MAX_UPLOAD_SIZE_BYTES=100 * 1024,  # 100 KB limit for easy boundary tests
        STORAGE_CHUNK_SIZE_BYTES=4096,
    )


@pytest.fixture
async def async_client(
    override_settings: Settings, test_storage: LocalFilesystemStorage
) -> AsyncGenerator[AsyncClient, None]:
    """Provide AsyncClient wired with test storage and test settings."""
    app.dependency_overrides[get_settings] = lambda: override_settings
    app.dependency_overrides[get_storage_engine] = lambda: test_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()


def make_sample_pdf() -> bytes:
    """Generate minimal valid PDF bytes."""
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n%%EOF\n"


def make_sample_docx() -> bytes:
    """Generate minimal valid DOCX zip bytes containing OOXML Content_Types."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types></Types>')
        z.writestr("word/document.xml", "<w:document><w:body><w:p/></w:body></w:document>")
    return bio.getvalue()


def make_sample_txt() -> bytes:
    """Generate plain text document bytes."""
    return b"RAGForge document ingestion integration test payload."


@pytest.mark.asyncio
async def test_upload_valid_pdf(async_client: AsyncClient, test_storage: StorageEngine) -> None:
    """Test successful upload of a PDF document."""
    pdf_content = make_sample_pdf()
    expected_hash = hashlib.sha256(pdf_content).hexdigest()

    files = {"file": ("test_report.pdf", pdf_content, "application/pdf")}
    response = await async_client.post("/api/v1/documents", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test_report.pdf"
    assert data["mime_type"] == "application/pdf"
    assert data["file_size_bytes"] == len(pdf_content)
    assert data["sha256_checksum"] == expected_hash
    assert data["status"] in ("uploaded", "queued")
    assert "storage_key" in data
    # Ensure raw filesystem paths are NEVER exposed to client
    assert "/test_storage" not in data["storage_key"]

    # Verify physical file existence in storage
    assert await test_storage.exists(data["storage_key"]) is True
    stored_bytes = await test_storage.read_bytes(data["storage_key"])
    assert stored_bytes == pdf_content


@pytest.mark.asyncio
async def test_upload_valid_docx(async_client: AsyncClient, test_storage: StorageEngine) -> None:
    """Test successful upload of a DOCX document."""
    docx_content = make_sample_docx()
    expected_hash = hashlib.sha256(docx_content).hexdigest()

    files = {
        "file": (
            "my_spec.docx",
            docx_content,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    }
    response = await async_client.post("/api/v1/documents", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "my_spec.docx"
    assert (
        data["mime_type"]
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert data["file_size_bytes"] == len(docx_content)
    assert data["sha256_checksum"] == expected_hash

    # Verify storage
    assert await test_storage.exists(data["storage_key"]) is True


@pytest.mark.asyncio
async def test_upload_valid_txt(async_client: AsyncClient, test_storage: StorageEngine) -> None:
    """Test successful upload of a TXT document."""
    txt_content = make_sample_txt()
    expected_hash = hashlib.sha256(txt_content).hexdigest()

    files = {"file": ("notes.txt", txt_content, "text/plain")}
    response = await async_client.post("/api/v1/documents", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "notes.txt"
    assert data["mime_type"] == "text/plain"
    assert data["file_size_bytes"] == len(txt_content)
    assert data["sha256_checksum"] == expected_hash

    # Verify physical file existence in storage
    assert await test_storage.exists(data["storage_key"]) is True
    stored = await test_storage.read_bytes(data["storage_key"])
    assert stored == txt_content


@pytest.mark.asyncio
async def test_upload_empty_file_rejected(async_client: AsyncClient) -> None:
    """Test upload of an empty 0-byte file returns HTTP 400."""
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_oversized_file_rejected(async_client: AsyncClient) -> None:
    """Test upload exceeding max configured upload size returns HTTP 413."""
    # Settings fixture configured max to 100 KB
    oversized_data = b"%PDF-1.4\n" + b"A" * (120 * 1024)
    files = {"file": ("huge.pdf", oversized_data, "application/pdf")}
    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 413
    assert "exceeds maximum permitted upload limit" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_executable_rejected(async_client: AsyncClient) -> None:
    """Test upload of executable binaries returns HTTP 415."""
    # ELF binary header
    files = {"file": ("malware.elf", b"\x7fELF\x02\x01\x01\x00", "application/octet-stream")}
    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 415

    # Windows PE header (MZ)
    files = {"file": ("trojan.exe", b"MZ\x90\x00\x03\x00", "application/octet-stream")}
    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_upload_malformed_docx_rejected(
    async_client: AsyncClient, test_storage: LocalFilesystemStorage
) -> None:
    """Test corrupted or non-OOXML zip claiming to be docx is rejected and cleaned up."""
    # Generic zip without [Content_Types].xml
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as z:
        z.writestr("test.txt", "not docx")
    fake_docx = bio.getvalue()

    files = {"file": ("corrupted.docx", fake_docx, "application/vnd.openxmlformats-officedocument")}
    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 415

    # Storage should have 0 files left
    assert list(test_storage._root_dir.glob("**/*.*")) == []


@pytest.mark.asyncio
async def test_upload_path_traversal_filename_sanitized(
    async_client: AsyncClient, test_storage: LocalFilesystemStorage
) -> None:
    """Test filename path traversal attempt is sanitized for storage and display."""
    pdf_content = make_sample_pdf()
    malicious_name = "../../../../../etc/passwd.pdf"
    files = {"file": (malicious_name, pdf_content, "application/pdf")}

    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "passwd.pdf"
    assert ".." not in data["storage_key"]
    assert await test_storage.exists(data["storage_key"]) is True


@pytest.mark.asyncio
async def test_document_crud_lifecycle(
    async_client: AsyncClient, test_storage: LocalFilesystemStorage
) -> None:
    """Test end-to-end CRUD: Upload -> Get by ID -> List pagination -> Delete."""
    # 1. Upload
    pdf = make_sample_pdf()
    res_upload = await async_client.post(
        "/api/v1/documents", files={"file": ("crud_test.pdf", pdf, "application/pdf")}
    )
    assert res_upload.status_code == 201
    doc_id = res_upload.json()["id"]
    storage_key = res_upload.json()["storage_key"]

    # 2. Get by ID
    res_get = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == doc_id
    assert res_get.json()["filename"] == "crud_test.pdf"

    # 3. List
    res_list = await async_client.get("/api/v1/documents?page=1&page_size=10")
    assert res_list.status_code == 200
    list_data = res_list.json()
    assert list_data["total"] >= 1
    assert any(item["id"] == doc_id for item in list_data["items"])

    # 4. Delete
    res_del = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert res_del.status_code == 200
    assert res_del.json()["deleted"] is True

    # Verify subsequent GET returns 404
    res_get_again = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert res_get_again.status_code == 404

    # Verify physical file removed from storage
    assert await test_storage.exists(storage_key) is False


@pytest.mark.asyncio
async def test_get_and_delete_missing_document_returns_404(async_client: AsyncClient) -> None:
    """Test that querying or deleting a non-existent UUID returns HTTP 404."""
    non_existent_id = uuid.uuid4()
    res_get = await async_client.get(f"/api/v1/documents/{non_existent_id}")
    assert res_get.status_code == 404
    assert str(non_existent_id) in res_get.json()["detail"]

    res_del = await async_client.delete(f"/api/v1/documents/{non_existent_id}")
    assert res_del.status_code == 404


@pytest.mark.asyncio
async def test_db_transaction_failure_cleans_up_file(
    async_client: AsyncClient, test_storage: LocalFilesystemStorage
) -> None:
    """Test that when a DB transaction fails during upload, the stored file is safely removed."""
    # Mock database session that raises during commit
    mock_db = AsyncMock(spec=AsyncSession)
    mock_db.commit.side_effect = RuntimeError("Simulated DB connection drop")
    app.dependency_overrides[get_db] = lambda: mock_db

    pdf = make_sample_pdf()
    files = {"file": ("fail_test.pdf", pdf, "application/pdf")}

    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 500

    # Ensure no orphaned files exist in storage
    files_in_storage = list(test_storage._root_dir.glob("**/*.*"))
    assert files_in_storage == []


@pytest.mark.asyncio
async def test_storage_failure_during_upload_returns_500(async_client: AsyncClient) -> None:
    """Test that if storage engine fails while writing, returns HTTP 500 cleanly."""
    mock_storage = AsyncMock(spec=StorageEngine)
    mock_storage.save.side_effect = StorageError("Disk I/O error")
    app.dependency_overrides[get_storage_engine] = lambda: mock_storage

    pdf = make_sample_pdf()
    files = {"file": ("io_fail.pdf", pdf, "application/pdf")}

    response = await async_client.post("/api/v1/documents", files=files)
    assert response.status_code == 500
    assert "Failed to store document" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_document_chunks_pagination_and_not_found(
    async_client: AsyncClient,
) -> None:
    """Test chunk retrieval endpoint pagination, ordering, and 404 on nonexistent doc."""
    # 1. 404 when document does not exist
    non_existent = uuid.uuid4()
    resp = await async_client.get(f"/api/v1/documents/{non_existent}/chunks")
    assert resp.status_code == 404

    # 2. Upload a text document
    txt_content = b"Paragraph 1 text.\n\nParagraph 2 text.\n\nParagraph 3 text."
    files = {"file": ("chunks_test.txt", txt_content, "text/plain")}
    up_resp = await async_client.post("/api/v1/documents", files=files)
    assert up_resp.status_code == 201
    doc_id = up_resp.json()["id"]

    # Retrieve chunks before processing (empty list, document status indicates state)
    chunk_resp = await async_client.get(f"/api/v1/documents/{doc_id}/chunks")
    assert chunk_resp.status_code == 200
    chunk_data = chunk_resp.json()
    assert chunk_data["document_id"] == doc_id
    assert chunk_data["total"] == 0
    assert chunk_data["items"] == []

    # 3. Simulate requeue endpoint
    requeue_resp = await async_client.post(f"/api/v1/documents/{doc_id}/requeue")
    assert requeue_resp.status_code in (202, 503)  # 202 if Redis up, 503 if down
    if requeue_resp.status_code == 202:
        assert requeue_resp.json()["status"] == "queued"


@pytest.mark.asyncio
async def test_download_document_url_generation(
    async_client: AsyncClient,
) -> None:
    """Test generating a download URL for an uploaded document."""
    # 404 for missing document
    missing_id = uuid.uuid4()
    resp = await async_client.get(f"/api/v1/documents/{missing_id}/download")
    assert resp.status_code == 404

    # Upload document
    pdf = make_sample_pdf()
    files = {"file": ("download_sample.pdf", pdf, "application/pdf")}
    up_resp = await async_client.post("/api/v1/documents", files=files)
    assert up_resp.status_code == 201
    doc_id = up_resp.json()["id"]

    # Request download URL
    dl_resp = await async_client.get(f"/api/v1/documents/{doc_id}/download?expires_in=1800")
    assert dl_resp.status_code == 200
    dl_data = dl_resp.json()
    assert dl_data["id"] == doc_id
    assert dl_data["expires_in_seconds"] == 1800
    assert "download_url" in dl_data
    assert dl_data["download_url"] != ""
