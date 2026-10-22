"""Document management API endpoints."""

import logging
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from ragforge.core.config import Settings, get_settings
from ragforge.db.session import get_db
from ragforge.schemas.chunk import DocumentChunkListResponse
from ragforge.schemas.document import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
)
from ragforge.services.document_service import (
    DocumentNotFoundError,
    DocumentService,
    DocumentValidationError,
)
from ragforge.services.queue import QueueService, get_queue_service
from ragforge.services.storage import StorageEngine
from ragforge.services.storage_provider import get_storage_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_service(
    db: AsyncSession = Depends(get_db),
    storage: StorageEngine = Depends(get_storage_engine),
    settings: Settings = Depends(get_settings),
    queue: QueueService = Depends(get_queue_service),
) -> DocumentService:
    """Dependency provider for DocumentService."""
    return DocumentService(db=db, storage=storage, settings=settings, queue=queue)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new document",
    description=(
        "Upload a document file (PDF, DOCX, or TXT) via multipart form data. "
        "Validates magic bytes, streams to secure storage, and persists metadata."
    ),
    responses={
        status.HTTP_201_CREATED: {
            "description": "Document uploaded successfully and metadata persisted.",
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Empty file or invalid upload request.",
        },
        status.HTTP_413_CONTENT_TOO_LARGE: {
            "description": "File exceeds the configured maximum upload size limit.",
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "description": "File content signature is malformed or an unsupported format.",
        },
    },
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (PDF, DOCX, or TXT)"),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Handle document upload with content signature verification and streaming storage."""
    try:
        return await doc_service.upload_document(file)
    except DocumentValidationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during document upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document upload.",
        ) from exc


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List uploaded documents",
    description=(
        "Retrieve a paginated collection of document metadata, ordered by creation date descending."
    ),
)
async def list_documents(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=20, ge=1, le=100, description="Number of items per page (max 100)"
    ),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List documents with bounded pagination."""
    return await doc_service.list_documents(page=page, page_size=page_size)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve document metadata",
    description="Fetch metadata for a specific document by its UUID.",
    responses={
        status.HTTP_200_OK: {
            "description": "Document metadata retrieved successfully.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Document not found.",
        },
    },
)
async def get_document(
    document_id: uuid.UUID,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Retrieve metadata for a specific document."""
    try:
        return await doc_service.get_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found.",
        ) from exc


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a document",
    description="Safely delete document metadata and remove the underlying stored file payload.",
    responses={
        status.HTTP_200_OK: {
            "description": "Document successfully deleted.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Document not found.",
        },
    },
)
async def delete_document(
    document_id: uuid.UUID,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentDeleteResponse:
    """Delete a document and its storage payload."""
    try:
        await doc_service.delete_document(document_id)
        return DocumentDeleteResponse(id=document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found.",
        ) from exc


@router.post(
    "/{document_id}/requeue",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Requeue a document for background processing",
    description=(
        "Manually enqueue or retry processing for a previously uploaded or failed document. "
        "Transitions document status back to 'queued'."
    ),
    responses={
        status.HTTP_202_ACCEPTED: {
            "description": "Document successfully requeued for background parsing.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Document not found.",
        },
        status.HTTP_409_CONFLICT: {
            "description": "Document is currently in processing state.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Background processing queue is currently unreachable.",
        },
    },
)
async def requeue_document(
    document_id: uuid.UUID,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Requeue a document without re-uploading file bytes."""
    try:
        return await doc_service.enqueue_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found.",
        ) from exc
    except DocumentValidationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{document_id}/chunks",
    response_model=DocumentChunkListResponse,
    status_code=status.HTTP_200_OK,
    summary="List extracted text chunks for a document",
    description=(
        "Retrieve paginated text chunks extracted and persisted for a document. "
        "Ordered deterministically by zero-based chunk_index. "
        "Returns empty list with status indication if document is not yet processed."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "Chunks retrieved successfully.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Document not found.",
        },
    },
)
async def get_document_chunks(
    document_id: uuid.UUID,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=20, ge=1, le=100, description="Number of chunks per page (max 100)"
    ),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentChunkListResponse:
    """Retrieve paginated persisted chunks for a specific document."""
    try:
        return await doc_service.get_document_chunks(
            document_id=document_id,
            page=page,
            page_size=page_size,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found.",
        ) from exc
