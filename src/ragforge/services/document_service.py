"""Document service for handling file validation, storage, and database persistence."""

import hashlib
import io
import logging
import re
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ragforge.core.config import Settings
from ragforge.db.models import Document, DocumentChunk, DocumentStatus
from ragforge.schemas.chunk import DocumentChunkListResponse, DocumentChunkResponse
from ragforge.schemas.document import DocumentListResponse, DocumentResponse
from ragforge.services.queue import QueueError, QueueService
from ragforge.services.storage import StorageEngine, StorageError

logger = logging.getLogger(__name__)

# Allowed MIME types and mapping
SUPPORTED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
}


class DocumentValidationError(Exception):
    """Raised when an uploaded document fails validation checks."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DocumentNotFoundError(Exception):
    """Raised when a requested document is not found."""

    def __init__(self, document_id: uuid.UUID) -> None:
        super().__init__(f"Document '{document_id}' was not found.")
        self.document_id = document_id


def sanitize_filename(filename: str | None) -> str:
    """Sanitize and normalize client-provided filename for safe display and logging.

    Strips directory separators, null bytes, non-ASCII controls, and caps length.
    """
    if not filename or not filename.strip():
        return "unnamed_document"

    # Normalize backslashes to forward slashes (handling Windows paths on Unix)
    normalized = filename.replace("\\", "/")
    # Take only the base name (prevents directory traversal patterns)
    base = Path(normalized).name
    # Strip null bytes and control characters
    base = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", base)
    # Remove dangerous characters
    base = re.sub(r'[/*?:"<>|]', "", base).strip()

    if not base or base in (".", ".."):
        return "unnamed_document"

    # Limit to 200 characters preserving extension if possible
    if len(base) > 200:
        stem = Path(base).stem[:180]
        ext = Path(base).suffix[:20]
        base = f"{stem}{ext}"

    return base


def inspect_file_content(head_bytes: bytes, filename: str) -> str:
    """Inspect leading bytes of a file to detect and validate format.

    Does NOT rely solely on client-supplied Content-Type.
    Validates:
      - PDF: starts with %PDF-
      - DOCX: starts with PK\\x03\\x04 and contains OOXML [Content_Types].xml structure
      - TXT: valid UTF-8/ASCII printable characters, no null bytes, no ELF/PE/Mach-O headers

    Returns:
        str: Validated MIME type string.

    Raises:
        DocumentValidationError: If file content signature does not match supported types.
    """
    if not head_bytes:
        raise DocumentValidationError("Upload file is empty.", status_code=400)

    # 1. Executable and binary format rejections (ELF, PE, Mach-O, shell scripts)
    if head_bytes.startswith(b"\x7fELF"):
        raise DocumentValidationError("Executable ELF binary files are rejected.", status_code=415)
    if head_bytes.startswith(b"MZ"):
        raise DocumentValidationError(
            "Windows PE / DOS executable files are rejected.", status_code=415
        )
    if head_bytes.startswith(
        (b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe")
    ):
        raise DocumentValidationError("Mach-O binary files are rejected.", status_code=415)

    # 2. PDF check: Magic number %PDF-
    if head_bytes.startswith(b"%PDF-"):
        return "application/pdf"

    # 3. DOCX check: Zip archive starting with PK\x03\x04
    if head_bytes.startswith(b"PK\x03\x04"):
        # Verify it is indeed an Office Open XML document
        # DOCX archives contain '[Content_Types].xml' and typically 'word/' directory
        if b"[Content_Types].xml" in head_bytes:
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        # If head_bytes is too short to see '[Content_Types].xml' or if it's a general zip,
        # we check the extension and zip table if possible
        lower_name = filename.lower()
        if lower_name.endswith(".docx"):
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        raise DocumentValidationError(
            "Zip archive is not a valid DOCX document.",
            status_code=415,
        )

    # 4. TXT check: must be valid text, not binary
    # If null byte present, it is definitely binary
    if b"\x00" in head_bytes:
        raise DocumentValidationError(
            "Unsupported binary file format or corrupted document.",
            status_code=415,
        )

    # Verify UTF-8 or ASCII decodable
    try:
        head_bytes.decode("utf-8")
        return "text/plain"
    except UnicodeDecodeError:
        raise DocumentValidationError(
            "Text file is not valid UTF-8 encoded text.",
            status_code=415,
        ) from None


class DocumentService:
    """Service layer encapsulating document business logic, validation, and storage coordination."""

    def __init__(
        self,
        db: AsyncSession,
        storage: StorageEngine,
        settings: Settings,
        queue: QueueService | None = None,
    ) -> None:
        self.db = db
        self.storage = storage
        self.settings = settings
        self.queue = queue

    async def upload_document(
        self,
        upload_file: UploadFile,
        auto_enqueue: bool = True,
    ) -> DocumentResponse:
        """Stream upload file in chunks, validate signatures, compute SHA-256, write to storage,
        and persist metadata in PostgreSQL. Cleans up stored file if database transaction fails.
        If auto_enqueue is True and queue is configured, enqueues background processing job.
        """
        raw_filename = upload_file.filename
        safe_filename = sanitize_filename(raw_filename)
        max_bytes = self.settings.STORAGE_MAX_UPLOAD_SIZE_BYTES
        chunk_size = self.settings.STORAGE_CHUNK_SIZE_BYTES

        # Generate a unique document ID and server-controlled storage key
        doc_id = uuid.uuid4()
        now = datetime.now(UTC)
        year_str = now.strftime("%Y")
        month_str = now.strftime("%m")
        day_str = now.strftime("%d")

        # Read first chunk to inspect file signature and magic bytes
        first_chunk = await upload_file.read(chunk_size)
        if not first_chunk:
            raise DocumentValidationError("Uploaded file is empty (0 bytes).", status_code=400)

        # Detect and validate MIME type from content inspection
        detected_mime = inspect_file_content(first_chunk, safe_filename)

        # Determine file extension based on validated MIME
        ext = SUPPORTED_MIME_TYPES[detected_mime]
        storage_key = f"docs/{year_str}/{month_str}/{day_str}/{doc_id.hex}{ext}"

        # Setup streaming, hashing, and size limit tracking
        sha256 = hashlib.sha256()
        sha256.update(first_chunk)
        total_bytes = len(first_chunk)

        if total_bytes > max_bytes:
            raise DocumentValidationError(
                f"File size exceeds maximum permitted upload limit of {max_bytes} bytes.",
                status_code=413,
            )

        # Buffer for deeper format validation if needed (e.g. DOCX zip structure verification)
        header_buffer = bytearray(first_chunk)

        from collections.abc import AsyncIterator

        async def stream_generator() -> AsyncIterator[bytes]:
            nonlocal total_bytes
            yield first_chunk
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise DocumentValidationError(
                        f"File size exceeds maximum permitted upload limit of {max_bytes} bytes.",
                        status_code=413,
                    )
                if len(header_buffer) < 256 * 1024:
                    header_buffer.extend(chunk)
                sha256.update(chunk)
                yield chunk

        # Stream directly into storage engine
        file_stored = False
        try:
            bytes_written = await self.storage.save(storage_key, stream_generator())
            file_stored = True
        except DocumentValidationError:
            raise
        except StorageError as e:
            logger.error("Storage write error during document upload", exc_info=True)
            raise DocumentValidationError(
                "Failed to store document payload.",
                status_code=500,
            ) from e

        # Perform additional format integrity validation if DOCX
        docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if detected_mime == docx_mime:
            # Verify valid zip containing word/document.xml or [Content_Types].xml
            is_valid_docx = False
            try:
                full_bytes = await self.storage.read_bytes(storage_key)
                with zipfile.ZipFile(io.BytesIO(full_bytes)) as z:
                    names = z.namelist()
                    if "[Content_Types].xml" in names:
                        is_valid_docx = True
            except Exception:
                is_valid_docx = False

            if not is_valid_docx:
                await self.storage.delete(storage_key)
                raise DocumentValidationError(
                    "Uploaded file is corrupted or not a valid Microsoft Word (.docx) document.",
                    status_code=415,
                )

        # Compute checksum hex digest
        checksum = sha256.hexdigest()

        # Create SQLAlchemy Document entity with initial UPLOADED status
        document = Document(
            id=doc_id,
            filename=safe_filename,
            storage_key=storage_key,
            mime_type=detected_mime,
            file_size_bytes=bytes_written,
            sha256_checksum=checksum,
            status=DocumentStatus.UPLOADED.value,
        )

        try:
            self.db.add(document)
            await self.db.commit()
            await self.db.refresh(document)
        except Exception as exc:
            # Transaction failed! Clean up the orphaned file from storage
            logger.critical(
                "Database transaction failed while saving document metadata. Cleaning up storage.",
                extra={"storage_key": storage_key, "error": str(exc)},
            )
            await self.db.rollback()
            if file_stored:
                try:
                    await self.storage.delete(storage_key)
                except Exception:
                    logger.error(
                        "Failed to clean up storage after db transaction failure",
                        extra={"storage_key": storage_key},
                    )
            raise

        logger.info(
            "Document successfully uploaded and metadata registered",
            extra={
                "document_id": str(document.id),
                "filename": document.filename,
                "mime_type": document.mime_type,
                "size_bytes": document.file_size_bytes,
                "storage_key": document.storage_key,
            },
        )

        # Enqueue background processing job if requested
        if auto_enqueue and self.queue is not None:
            try:
                await self.queue.enqueue_document_processing(document.id)
                document.status = DocumentStatus.QUEUED.value
                await self.db.commit()
                await self.db.refresh(document)
            except Exception as e:
                # Dual-write consideration: Redis enqueue failed, but document was saved.
                # Leave document in 'uploaded' status so it can be retried/requeued later.
                # DO NOT delete the uploaded file or document!
                logger.warning(
                    "Document was uploaded but failed to enqueue background job. "
                    "Status remains UPLOADED.",
                    extra={"document_id": str(document.id), "error": str(e)},
                )

        return DocumentResponse.model_validate(document)

    async def enqueue_document(self, document_id: uuid.UUID) -> DocumentResponse:
        """Enqueue an uploaded or failed document for background processing."""
        stmt = select(Document).where(Document.id == document_id)
        result = await self.db.execute(stmt)
        doc = result.scalar_one_or_none()
        if doc is None:
            raise DocumentNotFoundError(document_id)

        if doc.status == DocumentStatus.PROCESSING.value:
            raise DocumentValidationError("Document is currently being processed.", status_code=409)

        if self.queue is None:
            raise DocumentValidationError("Background job queue is not available.", status_code=503)

        try:
            await self.queue.enqueue_document_processing(doc.id)
            doc.status = DocumentStatus.QUEUED.value
            doc.error_code = None
            doc.error_message = None
            await self.db.commit()
            await self.db.refresh(doc)
            return DocumentResponse.model_validate(doc)
        except QueueError as exc:
            raise DocumentValidationError(
                f"Failed to enqueue document job: {exc}", status_code=503
            ) from exc

    async def get_document_chunks(
        self,
        document_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> DocumentChunkListResponse:
        """Retrieve paginated persisted chunks for a document ordered by chunk_index."""
        stmt = select(Document).where(Document.id == document_id)
        result = await self.db.execute(stmt)
        doc = result.scalar_one_or_none()
        if doc is None:
            raise DocumentNotFoundError(document_id)

        if page < 1:
            page = 1
        if page_size < 1 or page_size > 100:
            page_size = 20

        # Query total chunk count
        count_stmt = (
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
        )
        count_res = await self.db.execute(count_stmt)
        total = count_res.scalar_one()

        # Query paginated chunk records
        offset = (page - 1) * page_size
        chunk_stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .offset(offset)
            .limit(page_size)
        )
        chunk_res = await self.db.execute(chunk_stmt)
        chunk_rows = list(chunk_res.scalars().all())

        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        items = [DocumentChunkResponse.model_validate(c) for c in chunk_rows]

        return DocumentChunkListResponse(
            document_id=doc.id,
            status=doc.status,
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    async def get_document(self, document_id: uuid.UUID) -> DocumentResponse:
        """Fetch document metadata by UUID."""
        stmt = select(Document).where(Document.id == document_id)
        result = await self.db.execute(stmt)
        doc = result.scalar_one_or_none()
        if doc is None:
            raise DocumentNotFoundError(document_id)
        return DocumentResponse.model_validate(doc)

    async def list_documents(self, page: int = 1, page_size: int = 20) -> DocumentListResponse:
        """Fetch paginated document metadata ordered by creation date descending."""
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 100:
            page_size = 20

        # Query total count
        count_stmt = select(func.count()).select_from(Document)
        count_res = await self.db.execute(count_stmt)
        total = count_res.scalar_one()

        # Query records
        offset = (page - 1) * page_size
        stmt = select(Document).order_by(Document.created_at.desc()).offset(offset).limit(page_size)
        result = await self.db.execute(stmt)
        documents = list(result.scalars().all())

        total_pages = (total + page_size - 1) // page_size if total > 0 else 0

        items = [DocumentResponse.model_validate(d) for d in documents]
        return DocumentListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """Delete document record from database and safely remove its storage payload."""
        stmt = select(Document).where(Document.id == document_id)
        result = await self.db.execute(stmt)
        doc = result.scalar_one_or_none()
        if doc is None:
            raise DocumentNotFoundError(document_id)

        storage_key = doc.storage_key

        # Delete from database first in transaction (cascades to document_chunks)
        await self.db.delete(doc)
        await self.db.commit()

        # Now remove stored file
        try:
            await self.storage.delete(storage_key)
        except Exception as e:
            logger.warning(
                "Failed to delete stored file from storage engine after DB deletion",
                extra={
                    "document_id": str(document_id),
                    "storage_key": storage_key,
                    "error": str(e),
                },
            )
