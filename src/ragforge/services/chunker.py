"""Deterministic text chunking service for RAGForge.

Improves retrieval quality by splitting text along natural semantic boundaries
(paragraphs, sentences, and words) while respecting configured target chunk size,
overlap, minimum chunk size, and maximum chunks per document limits.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ragforge.services.extractor import ExtractedPage


class ChunkingError(Exception):
    """Base error for chunking failures."""

    def __init__(self, message: str, error_code: str = "CHUNKING_FAILED") -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class ChunkLimitExceededError(ChunkingError):
    """Raised when generated chunks exceed the configured document limit."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="CHUNK_LIMIT_EXCEEDED")


@dataclass(frozen=True)
class TextChunk:
    """Represents a discrete text chunk produced by the chunker."""

    chunk_index: int
    content: str
    page_number: int | None
    character_count: int
    token_count_estimate: int


def estimate_token_count(text: str) -> int:
    """Compute an approximate token count estimate using character heuristics.

    A widely accepted standard heuristic for English/general text is ~4 characters per token.
    This is explicitly documented and tracked as an estimate, NOT a real tokenizer output.
    """
    if not text:
        return 0
    # ~4 characters per token, minimum 1 token for non-empty text
    return max(1, (len(text) + 3) // 4)


class TextChunker:
    """Deterministic, boundary-aware text chunker.

    Rules:
    - Target chunk size: approx. target_chunk_size characters.
    - Overlap: approx. overlap characters maintained across adjacent chunks.
    - Boundaries: Splits at paragraph breaks ('\\n\\n') first, then sentence breaks
      ('. ', '? ', '! '), then words/spaces.
    - PDF Page Attribution: Prefers chunking within page boundaries. If a chunk crosses a boundary,
      attribution policy assigns the starting page number.
    - Deterministic: Identical inputs produce identical chunks in identical order.
    """

    def __init__(
        self,
        target_chunk_size: int = 1000,
        overlap: int = 150,
        min_chunk_size: int = 50,
        max_chunks_per_document: int = 2000,
    ) -> None:
        if target_chunk_size <= 0:
            raise ValueError("target_chunk_size must be positive.")
        if overlap < 0 or overlap >= target_chunk_size:
            raise ValueError(
                "overlap must be non-negative and strictly less than target_chunk_size."
            )
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size must be non-negative.")

        self.target_chunk_size = target_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunks_per_document = max_chunks_per_document

    def chunk_document(self, pages: Sequence[ExtractedPage]) -> list[TextChunk]:
        """Chunk a sequence of extracted pages deterministically.

        Respects page boundaries for PDFs so each chunk retains its source page number.
        """
        all_chunks: list[TextChunk] = []
        current_chunk_idx = 0

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            page_chunks = self._chunk_text(
                text=page_text,
                page_number=page.page_number,
                start_index=current_chunk_idx,
            )

            for chunk in page_chunks:
                all_chunks.append(chunk)
                current_chunk_idx += 1
                if current_chunk_idx > self.max_chunks_per_document:
                    raise ChunkLimitExceededError(
                        f"Document exceeded maximum limit of {self.max_chunks_per_document} chunks."
                    )

        if not all_chunks:
            raise ChunkingError("Document produced 0 valid chunks.")

        return all_chunks

    def _chunk_text(
        self,
        text: str,
        page_number: int | None,
        start_index: int,
    ) -> list[TextChunk]:
        """Split arbitrary text into overlapping chunks around target size."""
        if len(text) <= self.target_chunk_size:
            char_count = len(text)
            tokens = estimate_token_count(text)
            return [
                TextChunk(
                    chunk_index=start_index,
                    content=text,
                    page_number=page_number,
                    character_count=char_count,
                    token_count_estimate=tokens,
                )
            ]

        # Break text into atomic segments: paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        atomic_units: list[str] = []
        for para in paragraphs:
            if len(para) <= self.target_chunk_size:
                atomic_units.append(para)
            else:
                # Sub-split long paragraph into sentences or smaller clauses
                sentences = self._split_sentences(para)
                for sentence in sentences:
                    if len(sentence) <= self.target_chunk_size:
                        atomic_units.append(sentence)
                    else:
                        # Sub-split long sentence into words
                        words = self._split_words(sentence, self.target_chunk_size)
                        atomic_units.extend(words)

        # Assemble atomic units into overlapping chunks
        raw_chunks: list[str] = []
        current_buf: list[str] = []
        current_len = 0

        i = 0
        while i < len(atomic_units):
            unit = atomic_units[i]
            unit_len = len(unit)

            if not current_buf:
                current_buf.append(unit)
                current_len = unit_len
                i += 1
                continue

            # Check if adding unit fits in target size (with separation)
            projected_len = current_len + 1 + unit_len
            if projected_len <= self.target_chunk_size:
                current_buf.append(unit)
                current_len = projected_len
                i += 1
            else:
                # Finalize chunk
                chunk_str = " ".join(current_buf).strip()
                if chunk_str:
                    raw_chunks.append(chunk_str)

                # Compute overlap for the next chunk
                if self.overlap > 0:
                    overlap_buf: list[str] = []
                    overlap_len = 0
                    for rev_unit in reversed(current_buf):
                        if overlap_len + len(rev_unit) + 1 <= self.overlap:
                            overlap_buf.insert(0, rev_unit)
                            overlap_len += len(rev_unit) + 1
                        else:
                            break
                    current_buf = list(overlap_buf)
                    current_len = sum(len(u) for u in current_buf) + max(0, len(current_buf) - 1)
                else:
                    current_buf = []
                    current_len = 0

                # Ensure we add the current unit that didn't fit, advancing i
                current_buf.append(unit)
                current_len += 1 + unit_len if current_len > 0 else unit_len
                i += 1

        if current_buf:
            tail_str = " ".join(current_buf).strip()
            # If tail chunk is smaller than min_chunk_size and we already have chunks,
            # merge it with the last chunk if possible, or append it
            if raw_chunks and len(tail_str) < self.min_chunk_size:
                merged = f"{raw_chunks[-1]} {tail_str}".strip()
                raw_chunks[-1] = merged
            elif tail_str:
                raw_chunks.append(tail_str)

        result_chunks: list[TextChunk] = []
        for offset, chunk_text in enumerate(raw_chunks):
            result_chunks.append(
                TextChunk(
                    chunk_index=start_index + offset,
                    content=chunk_text,
                    page_number=page_number,
                    character_count=len(chunk_text),
                    token_count_estimate=estimate_token_count(chunk_text),
                )
            )

        return result_chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split a paragraph into sentences preserving ending punctuation."""
        # Regex splits on punctuation (. ! ?) followed by space or newline
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_words(self, text: str, max_len: int) -> list[str]:
        """Split a long sentence into chunks of words <= max_len."""
        words = text.split()
        segments: list[str] = []
        current: list[str] = []
        current_len = 0

        for w in words:
            if current_len + len(w) + 1 <= max_len or not current:
                current.append(w)
                current_len += len(w) + 1
            else:
                segments.append(" ".join(current))
                current = [w]
                current_len = len(w)

        if current:
            segments.append(" ".join(current))
        return segments
