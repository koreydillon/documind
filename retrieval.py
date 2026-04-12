"""
retrieval.py
------------
Top-k semantic retrieval from the FAISS index, returning Chunk objects
(text + page number) rather than bare strings so the UI can render citations.
"""

from __future__ import annotations

import faiss

from embeddings import embed_texts
from ingestion import Chunk


def retrieve_top_k(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[Chunk],
    k: int = 5,
) -> list[Chunk]:
    query_vector = embed_texts([query])
    effective_k = min(k, index.ntotal)
    _distances, indices = index.search(query_vector, effective_k)
    return [chunks[i] for i in indices[0] if i != -1]
