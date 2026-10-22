"""Unit tests for text extraction services across PDF, DOCX, and TXT files."""

import io

import docx
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ragforge.services.extractor import (
    DocumentBoundsExceededError,
    DocumentExtractor,
    EmptyDocumentError,
    MalformedDocumentError,
    sanitize_text,
)


def create_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to generate valid multi-page PDF byte stream with text readable by pypdf."""
    writer = PdfWriter()

    # Define standard Type 1 font
    font_dict = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font_dict)

    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        res_dict = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): font_ref,
                    }
                )
            }
        )
        page[NameObject("/Resources")] = res_dict

        content_bytes = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET".encode("latin-1")
        stream = DecodedStreamObject()
        stream.set_data(content_bytes)
        stream_ref = writer._add_object(stream)
        page[NameObject("/Contents")] = stream_ref

    bio = io.BytesIO()
    writer.write(bio)
    return bio.getvalue()


def create_docx_bytes(paragraphs: list[str]) -> bytes:
    """Helper to generate a valid DOCX byte stream."""
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


def test_sanitize_text() -> None:
    """Verify text normalization and control character stripping."""
    raw = "Hello\r\nWorld\r\nTest\x00\x05\x1fwith\tTabs and Unicode: café! 🚀"
    cleaned = sanitize_text(raw)
    assert cleaned == "Hello\nWorld\nTestwith\tTabs and Unicode: café! 🚀"
    assert "\x00" not in cleaned
    assert "\r" not in cleaned


def test_extract_valid_pdf_multiple_pages() -> None:
    """Verify extracting multiple pages from a valid PDF preserves page attribution."""
    pdf_bytes = create_pdf_bytes(["Page One Content", "Page Two Content"])
    extractor = DocumentExtractor()
    pages = extractor.extract_pdf(pdf_bytes)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Page One Content" in pages[0].text
    assert pages[1].page_number == 2
    assert "Page Two Content" in pages[1].text


def test_extract_empty_or_image_only_pdf_raises_empty_error() -> None:
    """Verify empty pages (e.g. image-only PDF needing OCR) raise EmptyDocumentError."""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    bio = io.BytesIO()
    writer.write(bio)
    empty_pdf = bio.getvalue()

    extractor = DocumentExtractor()
    with pytest.raises(EmptyDocumentError) as exc_info:
        extractor.extract_pdf(empty_pdf)
    assert "OCR" in str(exc_info.value)
    assert exc_info.value.error_code == "DOCUMENT_EMPTY"


def test_extract_malformed_pdf_raises_error() -> None:
    """Verify corrupted PDF bytes raise MalformedDocumentError."""
    extractor = DocumentExtractor()
    with pytest.raises(MalformedDocumentError):
        extractor.extract_pdf(b"%PDF-1.4\nCorrupted garbage content %%EOF")


def test_extract_pdf_page_limit_exceeded() -> None:
    """Verify exceeding configured max_pages raises DocumentBoundsExceededError."""
    pdf_bytes = create_pdf_bytes(["P1", "P2", "P3"])
    extractor = DocumentExtractor(max_pages=2)
    with pytest.raises(DocumentBoundsExceededError):
        extractor.extract_pdf(pdf_bytes)


def test_extract_valid_docx_multiple_paragraphs() -> None:
    """Verify extracting DOCX paragraphs in order."""
    docx_bytes = create_docx_bytes(["First Paragraph.", "Second Paragraph with detail."])
    extractor = DocumentExtractor()
    pages = extractor.extract_docx(docx_bytes)

    assert len(pages) == 1
    assert pages[0].page_number is None
    assert "First Paragraph." in pages[0].text
    assert "Second Paragraph with detail." in pages[0].text
    assert "\n\n" in pages[0].text


def test_extract_empty_docx_raises_empty_error() -> None:
    """Verify DOCX with no paragraphs raises EmptyDocumentError."""
    docx_bytes = create_docx_bytes([])
    extractor = DocumentExtractor()
    with pytest.raises(EmptyDocumentError):
        extractor.extract_docx(docx_bytes)


def test_extract_malformed_docx_raises_error() -> None:
    """Verify corrupted DOCX raises MalformedDocumentError."""
    extractor = DocumentExtractor()
    with pytest.raises(MalformedDocumentError):
        extractor.extract_docx(b"PK\x03\x04not-a-valid-docx-zip")


def test_extract_valid_txt() -> None:
    """Verify extracting UTF-8 text file."""
    txt_bytes = "Plain text line 1.\nPlain text line 2.\nUnicode: नमस्ते".encode()
    extractor = DocumentExtractor()
    pages = extractor.extract_txt(txt_bytes)

    assert len(pages) == 1
    assert pages[0].page_number is None
    assert "Plain text line 1." in pages[0].text
    assert "नमस्ते" in pages[0].text


def test_extract_invalid_encoding_txt_raises_error() -> None:
    """Verify invalid UTF-8 bytes raise MalformedDocumentError."""
    invalid_bytes = b"\xff\xfe\x00\x00Invalid UTF-8"
    extractor = DocumentExtractor()
    with pytest.raises(MalformedDocumentError):
        extractor.extract_txt(invalid_bytes)


def test_extract_empty_txt_raises_error() -> None:
    """Verify empty text file raises EmptyDocumentError."""
    extractor = DocumentExtractor()
    with pytest.raises(EmptyDocumentError):
        extractor.extract_txt(b"   \n\t  ")


def test_extract_text_size_limit_exceeded() -> None:
    """Verify extracted text exceeding max_text_size_bytes raises bounds error."""
    extractor = DocumentExtractor(max_text_size_bytes=50)
    large_text = ("A" * 100).encode("utf-8")
    with pytest.raises(DocumentBoundsExceededError):
        extractor.extract_txt(large_text)
