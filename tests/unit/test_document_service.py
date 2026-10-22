"""Unit tests for document service validation and sanitization helpers."""

import io
import zipfile

import pytest

from ragforge.services.document_service import (
    DocumentValidationError,
    inspect_file_content,
    sanitize_filename,
)


def test_sanitize_filename() -> None:
    """Test filename sanitization against path traversal, control chars, and long names."""
    # Normal filename
    assert sanitize_filename("report.pdf") == "report.pdf"

    # Traversal attempts
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\calc.exe") == "calc.exe"

    # Dangerous characters and control characters
    assert sanitize_filename("my*doc?name<>|.txt") == "mydocname.txt"
    assert sanitize_filename("file\x00with\x1fcontrols.pdf") == "filewithcontrols.pdf"

    # Empty and edge cases
    assert sanitize_filename("") == "unnamed_document"
    assert sanitize_filename("   ") == "unnamed_document"
    assert sanitize_filename(".") == "unnamed_document"
    assert sanitize_filename("..") == "unnamed_document"

    # Length capping
    long_name = "a" * 250 + ".pdf"
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) <= 200
    assert sanitized.endswith(".pdf")


def test_inspect_file_content_valid_pdf() -> None:
    """Test PDF content inspection passes for %PDF- signature."""
    pdf_bytes = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n"
    mime = inspect_file_content(pdf_bytes, "doc.pdf")
    assert mime == "application/pdf"


def test_inspect_file_content_valid_docx() -> None:
    """Test DOCX content inspection passes for zip containing OOXML [Content_Types].xml."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?>')
        z.writestr("word/document.xml", "<w:document></w:document>")
    docx_bytes = bio.getvalue()

    mime = inspect_file_content(docx_bytes, "doc.docx")
    assert mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_inspect_file_content_valid_txt() -> None:
    """Test plain text content inspection passes for UTF-8 and ASCII text."""
    txt_bytes = "This is a plain text file for RAGForge indexing.\nWith UTF-8 unicode: 🚀".encode()
    mime = inspect_file_content(txt_bytes, "notes.txt")
    assert mime == "text/plain"


def test_inspect_file_content_rejects_empty() -> None:
    """Test rejection of empty 0-byte files."""
    with pytest.raises(DocumentValidationError) as exc:
        inspect_file_content(b"", "empty.txt")
    assert exc.value.status_code == 400


def test_inspect_file_content_rejects_executables() -> None:
    """Test rejection of executable binaries (ELF, PE, Mach-O)."""
    # ELF
    with pytest.raises(DocumentValidationError) as exc:
        inspect_file_content(b"\x7fELF\x02\x01\x01\x00", "program")
    assert exc.value.status_code == 415

    # Windows PE (MZ)
    with pytest.raises(DocumentValidationError) as exc:
        inspect_file_content(b"MZ\x90\x00\x03\x00\x00\x00", "trojan.exe")
    assert exc.value.status_code == 415

    # Mach-O
    with pytest.raises(DocumentValidationError) as exc:
        inspect_file_content(b"\xfe\xed\xfa\xcf\x00\x00", "binary.macho")
    assert exc.value.status_code == 415


def test_inspect_file_content_rejects_arbitrary_binary_as_txt() -> None:
    """Test binary with null bytes or non-text bytes is rejected."""
    with pytest.raises(DocumentValidationError) as exc:
        inspect_file_content(b"Some text\x00with null byte", "fake.txt")
    assert exc.value.status_code == 415

    # Invalid UTF-8 sequence
    with pytest.raises(DocumentValidationError) as exc:
        inspect_file_content(b"\xff\xfe\x99\x88", "bad_encoding.txt")
    assert exc.value.status_code == 415


def test_inspect_file_content_rejects_generic_zip_as_docx() -> None:
    """Test that a non-DOCX zip is rejected if extension does not claim docx."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as z:
        z.writestr("some_data.csv", "1,2,3")
    generic_zip = bio.getvalue()

    with pytest.raises(DocumentValidationError) as exc:
        inspect_file_content(generic_zip, "archive.zip")
    assert exc.value.status_code == 415
