"""
ingestion.py
------------
Handles PDF text extraction via PyMuPDF and splits the resulting text into
overlapping word-based chunks suitable for embedding.
"""

from __future__ import annotations

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract all text from a PDF supplied as raw bytes.

    Parameters
    ----------
    pdf_bytes:
        Raw binary content of the PDF file (e.g. from an uploaded Streamlit
        file buffer).

    Returns
    -------
    tuple[str, int]
        A 2-tuple of (full_text, page_count) where *full_text* is the
        concatenated content of every page separated by newlines and
        *page_count* is the total number of pages in the document.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[str] = []

    for page in doc:
        pages.append(page.get_text())

    full_text = "\n".join(pages)
    page_count = len(doc)
    doc.close()

    return full_text, page_count


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split *text* into overlapping word-based chunks.

    A word-based sliding window is used instead of character counts so that
    chunk boundaries never split mid-word, which produces cleaner embeddings.

    Parameters
    ----------
    text:
        The full document text to be chunked.
    chunk_size:
        Approximate number of words per chunk (default 500).
    overlap:
        Number of words from the previous chunk to repeat at the start of the
        next chunk (default 50). Overlap preserves context across boundaries.

    Returns
    -------
    list[str]
        Ordered list of text chunks.  Each chunk is a plain string; no
        metadata is attached here.
    """
    words = text.split()
    chunks: list[str] = []

    if not words:
        return chunks

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Advance by (chunk_size - overlap) so successive chunks overlap
        start += chunk_size - overlap

    return chunks
