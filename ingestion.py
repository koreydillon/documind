"""
ingestion.py
------------
PDF text extraction + page-aware chunking.

Each chunk carries the page number it originated from so the UI can render
precise source citations (e.g. "[1] page 12") rather than opaque passage
indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Chunk:
    text: str
    page: int  # 1-indexed page number in the source PDF


def extract_pages_from_pdf(pdf_bytes: bytes) -> tuple[list[tuple[int, str]], int]:
    """Extract text on a per-page basis from a PDF.

    Returns
    -------
    tuple[list[tuple[int, str]], int]
        A list of ``(page_num, page_text)`` tuples (1-indexed) plus the total
        page count.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(doc, start=1):
        pages.append((i, page.get_text()))
    page_count = len(doc)
    doc.close()
    return pages, page_count


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    """Backwards-compatible whole-document text extraction.

    Kept because ``summarize_document`` still wants a single concatenated
    string for the full-document summary prompt.
    """
    pages, page_count = extract_pages_from_pdf(pdf_bytes)
    full_text = "\n".join(text for _, text in pages)
    return full_text, page_count


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split per-page text into overlapping word-based chunks, preserving the
    originating page number on every chunk.

    A chunk never crosses a page boundary — each page is chunked independently.
    This keeps page citations exact at the cost of slightly more chunks for
    PDFs with very short pages, which is the right trade-off for enterprise
    Q&A where "which page did this come from" is a frequent follow-up.
    """
    chunks: list[Chunk] = []
    for page_num, page_text in pages:
        words = page_text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(text=chunk_text, page=page_num))
            start += chunk_size - overlap
    return chunks
