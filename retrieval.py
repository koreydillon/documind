"""
retrieval.py
------------
Embeds a user query and retrieves the top-k most semantically similar chunks
from the FAISS index built during the ingestion phase.
"""

from __future__ import annotations

import numpy as np
import faiss

from embeddings import embed_chunks


def retrieve_top_k(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    k: int = 5,
) -> list[str]:
    """Embed *query* and return the *k* most relevant document chunks.

    The same embedding model used for indexing is applied to the query so that
    distances are computed in the same vector space.  FAISS returns chunk
    indices sorted ascending by L2 distance (closest first).

    Parameters
    ----------
    query:
        The user's natural-language question.
    index:
        Populated FAISS index containing the document chunk embeddings.
    chunks:
        Ordered list of raw text chunks whose positions correspond to the
        FAISS index rows.
    k:
        Number of nearest neighbours to retrieve.  Defaults to 5.

    Returns
    -------
    list[str]
        Up to *k* text chunks ordered from most to least relevant.
    """
    # Embed the query; embed_chunks expects a list, so wrap in one
    query_vector: np.ndarray = embed_chunks([query])  # shape (1, dim)

    # Clamp k so we never ask for more results than there are chunks
    effective_k = min(k, index.ntotal)

    # FAISS returns (distances, indices) both of shape (n_queries, k)
    _distances, indices = index.search(query_vector, effective_k)

    # indices[0] is the array of nearest-neighbour positions for the single query
    return [chunks[i] for i in indices[0] if i != -1]
