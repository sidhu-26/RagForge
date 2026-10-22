"""Document text extraction services.

Provides robust, format-specific extractors for PDF, DOCX, and TXT files.
Extracts page-aware text, enforces length and page bounds, normalizes control characters,
and detects empty or unextractable (e.g. image-only) documents.
"""

import io
import re
from collections.abc import Sequence
from dataclasses import dataclass

import docx
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ExtractionError(Exception):
    """Base error for document extraction failures."""

    def __init__(self, message: str, error_code: str = "EXTRACTION_FAILED") -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class EmptyDocumentError(ExtractionError):
    """Raised when a document contains no extractable text (e.g. image-only PDF, blank doc)."""

    def __init__(
        self, message: str = "Document contains no extractable text. OCR may be required."
    ) -> None:
        super().__init__(message, error_code="DOCUMENT_EMPTY")


class DocumentBoundsExceededError(ExtractionError):
    """Raised when a document exceeds configured page count or extracted text size limits."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="BOUNDS_EXCEEDED")


class MalformedDocumentError(ExtractionError):
    """Raised when a document cannot be parsed or decoded due to corrupt structure."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="MALFORMED_DOCUMENT")


@dataclass(frozen=True)
class ExtractedPage:
    """Represents text extracted from a distinct document segment or page.

    For PDFs, `page_number` is a 1-based index corresponding to the PDF page.
    For DOCX and TXT, `page_number` is None.
    """

    page_number: int | None
    text: str


def sanitize_text(text: str) -> str:
    """Normalize line endings and strip control characters while preserving meaningful formatting.

    Converts CRLF and CR to LF.
    Removes ASCII/Unicode non-printable control characters (except \\n, \\t, \\r).
    Preserves unicode punctuation, spaces, and code formatting.
    """
    if not text:
        return ""

    # Normalize newlines: \r\n -> \n, \r -> \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printable control characters (keep \t and \n)
    # \x00-\x08, \x0b-\x0c, \x0e-\x1f, \x7f-\x9f
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    return text.strip()


class DocumentExtractor:
    """Extractor handling text extraction across supported MIME types."""

    def __init__(
        self,
        max_pages: int = 500,
        max_text_size_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.max_pages = max_pages
        self.max_text_size_bytes = max_text_size_bytes

    def extract_pdf(self, file_bytes: bytes) -> list[ExtractedPage]:
        """Extract text from a PDF file page by page using pypdf.

        Enforces page limits and text size limits. Detects image-only/empty PDFs.
        """
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            num_pages = len(reader.pages)
        except (PdfReadError, Exception) as exc:
            raise MalformedDocumentError("Malformed or corrupted PDF file.") from exc

        if num_pages == 0:
            raise EmptyDocumentError("PDF file contains no pages.")

        if num_pages > self.max_pages:
            raise DocumentBoundsExceededError(
                f"PDF exceeds maximum page limit of {self.max_pages} pages (found {num_pages})."
            )

        pages: list[ExtractedPage] = []
        total_text_length = 0

        for idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                raise MalformedDocumentError(
                    f"Failed to extract text from PDF page {idx}."
                ) from exc

            clean_text = sanitize_text(page_text)
            if clean_text:
                total_text_length += len(clean_text.encode("utf-8"))
                if total_text_length > self.max_text_size_bytes:
                    raise DocumentBoundsExceededError(
                        "Extracted document text exceeds limit of "
                        f"{self.max_text_size_bytes} bytes."
                    )
                pages.append(ExtractedPage(page_number=idx, text=clean_text))

        if not pages:
            raise EmptyDocumentError(
                "PDF contains no extractable text. Scanned or image-only documents "
                "require OCR, which is not supported."
            )

        return pages

    def extract_docx(self, file_bytes: bytes) -> list[ExtractedPage]:
        """Extract text paragraphs from a Microsoft Word (.docx) document in order.

        Uses python-docx. Page numbers are set to None.
        """
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
        except Exception as exc:
            raise MalformedDocumentError("Malformed or corrupted DOCX file.") from exc

        paragraphs: list[str] = []
        total_bytes = 0

        for p in doc.paragraphs:
            text = sanitize_text(p.text)
            if text:
                paragraphs.append(text)
                total_bytes += len(text.encode("utf-8"))
                if total_bytes > self.max_text_size_bytes:
                    raise DocumentBoundsExceededError(
                        "Extracted document text exceeds limit of "
                        f"{self.max_text_size_bytes} bytes."
                    )

        # Also extract table text in order if present
        for table in doc.tables:
            for row in table.rows:
                row_cells = [sanitize_text(c.text) for c in row.cells if sanitize_text(c.text)]
                if row_cells:
                    row_text = " | ".join(row_cells)
                    paragraphs.append(row_text)
                    total_bytes += len(row_text.encode("utf-8"))
                    if total_bytes > self.max_text_size_bytes:
                        raise DocumentBoundsExceededError(
                            "Extracted document text exceeds limit of "
                            f"{self.max_text_size_bytes} bytes."
                        )

        if not paragraphs:
            raise EmptyDocumentError("DOCX file contains no extractable text.")

        combined_text = "\n\n".join(paragraphs)
        return [ExtractedPage(page_number=None, text=combined_text)]

    def extract_txt(self, file_bytes: bytes) -> list[ExtractedPage]:
        """Safely decode UTF-8 plain text file.

        Page number is set to None.
        """
        try:
            raw_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MalformedDocumentError("Text file is not valid UTF-8 encoded text.") from exc

        clean_text = sanitize_text(raw_text)
        if not clean_text:
            raise EmptyDocumentError(
                "Text file is empty or contains only whitespace/control characters."
            )

        if len(clean_text.encode("utf-8")) > self.max_text_size_bytes:
            raise DocumentBoundsExceededError(
                f"Extracted document text exceeds limit of {self.max_text_size_bytes} bytes."
            )

        return [ExtractedPage(page_number=None, text=clean_text)]

    def extract(self, mime_type: str, file_bytes: bytes) -> Sequence[ExtractedPage]:
        """Dispatch extraction according to MIME type."""
        if mime_type == "application/pdf":
            return self.extract_pdf(file_bytes)
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self.extract_docx(file_bytes)
        elif mime_type == "text/plain":
            return self.extract_txt(file_bytes)
        else:
            raise MalformedDocumentError(
                f"Unsupported document MIME type for extraction: '{mime_type}'."
            )
