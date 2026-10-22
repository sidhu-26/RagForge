"""Unit tests for deterministic TextChunker service."""

import pytest

from ragforge.services.chunker import (
    ChunkingError,
    ChunkLimitExceededError,
    TextChunker,
    estimate_token_count,
)
from ragforge.services.extractor import ExtractedPage


def test_estimate_token_count() -> None:
    """Verify approximate token estimation heuristic."""
    assert estimate_token_count("") == 0
    assert estimate_token_count("Hi") == 1
    assert estimate_token_count("12345678") == 2
    assert estimate_token_count("A" * 100) == 25


def test_chunking_short_document() -> None:
    """Verify that a document shorter than target_chunk_size produces exactly one chunk."""
    chunker = TextChunker(target_chunk_size=1000, overlap=150)
    pages = [ExtractedPage(page_number=1, text="Short document text.")]
    chunks = chunker.chunk_document(pages)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == "Short document text."
    assert chunks[0].page_number == 1
    assert chunks[0].character_count == len("Short document text.")
    assert chunks[0].token_count_estimate == estimate_token_count("Short document text.")


def test_chunking_preserves_deterministic_order() -> None:
    """Verify identical inputs produce identical chunks with sequential indexing."""
    text = (
        "Introduction to Retrieval-Augmented Generation (RAG). "
        "RAG systems enhance language models by grounding them in factual retrieved documents. "
        "This pipeline segments text into discrete chunks."
    )
    pages = [ExtractedPage(page_number=None, text=text)]
    chunker = TextChunker(target_chunk_size=80, overlap=20)

    run_1 = chunker.chunk_document(pages)
    run_2 = chunker.chunk_document(pages)

    assert len(run_1) == len(run_2)
    for c1, c2 in zip(run_1, run_2, strict=True):
        assert c1.chunk_index == c2.chunk_index
        assert c1.content == c2.content
        assert c1.character_count == c2.character_count
        assert c1.token_count_estimate == c2.token_count_estimate
        assert c1.page_number == c2.page_number


def test_chunking_multiple_paragraphs_and_overlap() -> None:
    """Verify paragraph boundaries are respected and overlap is maintained."""
    p1 = "Paragraph 1: " + ("Alpha " * 20)
    p2 = "Paragraph 2: " + ("Beta " * 20)
    p3 = "Paragraph 3: " + ("Gamma " * 20)
    full_text = f"{p1}\n\n{p2}\n\n{p3}"

    chunker = TextChunker(target_chunk_size=120, overlap=30, min_chunk_size=20)
    pages = [ExtractedPage(page_number=1, text=full_text)]
    chunks = chunker.chunk_document(pages)

    assert len(chunks) >= 3
    # Verify no chunk is empty
    for c in chunks:
        assert len(c.content) > 0
        assert c.character_count > 0

    # Verify overlap exists between consecutive chunks
    overlap_found = False
    for i in range(len(chunks) - 1):
        words_curr = set(chunks[i].content.split()[-3:])
        words_next = set(chunks[i + 1].content.split()[:5])
        if words_curr.intersection(words_next):
            overlap_found = True
            break
    assert overlap_found is True


def test_long_paragraph_splitting() -> None:
    """Verify very long paragraph without paragraph breaks is split cleanly without losing text."""
    sentences = [f"Sentence number {i} with descriptive detail." for i in range(30)]
    long_para = " ".join(sentences)

    chunker = TextChunker(target_chunk_size=150, overlap=30)
    pages = [ExtractedPage(page_number=None, text=long_para)]
    chunks = chunker.chunk_document(pages)

    assert len(chunks) > 1
    # Check that key beginning and end words are present
    assert "Sentence number 0" in chunks[0].content
    assert "Sentence number 29" in chunks[-1].content


def test_pdf_page_attribution() -> None:
    """Verify page attribution adheres strictly to PDF page numbers."""
    pages = [
        ExtractedPage(page_number=1, text="First page content of the PDF."),
        ExtractedPage(
            page_number=2, text="Second page content of the PDF with distinct information."
        ),
    ]
    chunker = TextChunker(target_chunk_size=1000, overlap=50)
    chunks = chunker.chunk_document(pages)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].content == "First page content of the PDF."
    assert chunks[1].page_number == 2
    assert chunks[1].content == "Second page content of the PDF with distinct information."
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_max_chunks_limit_exceeded() -> None:
    """Verify exceeding configured maximum chunks raises ChunkLimitExceededError."""
    chunker = TextChunker(target_chunk_size=50, overlap=10, max_chunks_per_document=3)
    text = "Word " * 100
    pages = [ExtractedPage(page_number=None, text=text)]

    with pytest.raises(ChunkLimitExceededError):
        chunker.chunk_document(pages)


def test_no_empty_chunks_from_whitespace_or_empty_pages() -> None:
    """Verify empty or whitespace pages do not produce empty chunks."""
    chunker = TextChunker()
    pages = [ExtractedPage(page_number=1, text="   \n\n\t  ")]
    with pytest.raises(ChunkingError):
        chunker.chunk_document(pages)


def test_unicode_and_punctuation_preservation() -> None:
    """Verify Unicode characters, punctuation, and code snippets are preserved accurately."""
    content = (
        "Mathematical expression: ∑(x_i^2) ≥ 0. German: Übergrößenträger. "
        "Japanese: 日本語. Code: def foo(): pass"
    )
    chunker = TextChunker(target_chunk_size=500, overlap=50)
    pages = [ExtractedPage(page_number=None, text=content)]
    chunks = chunker.chunk_document(pages)

    assert len(chunks) == 1
    assert "∑(x_i^2) ≥ 0" in chunks[0].content
    assert "Übergrößenträger" in chunks[0].content
    assert "日本語" in chunks[0].content
    assert "def foo(): pass" in chunks[0].content
