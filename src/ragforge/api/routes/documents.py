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
from ragforge.services.storage import StorageEngine
from ragforge.services.storage_provider import get_storage_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_service(
    db: AsyncSession = Depends(get_db),
    storage: StorageEngine = Depends(get_storage_engine),
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    """Dependency provider for DocumentService."""
    return DocumentService(db=db, storage=storage, settings=settings)


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
