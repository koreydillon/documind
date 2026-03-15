"""
embeddings.py
-------------
Loads the all-MiniLM-L6-v2 Sentence-BERT model, converts text chunks into
dense vectors, and builds / manages a FAISS flat L2 index.
"""

from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Model singleton — loaded once per process so Streamlit reruns don't reload
# ---------------------------------------------------------------------------
_MODEL: SentenceTransformer | None = None

MODEL_NAME = "all-MiniLM-L6-v2"


def get_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer model instance, loading it on
    first call.

    Using a module-level singleton avoids reloading the ~80 MB model weights
    on every Streamlit rerun, which would be prohibitively slow.

    Returns
    -------
    SentenceTransformer
        The loaded ``all-MiniLM-L6-v2`` model.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Convert a list of text chunks into a 2-D float32 embedding matrix.

    Parameters
    ----------
    chunks:
        List of text strings to embed.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(chunks), embedding_dim)`` with dtype
        ``float32``.  FAISS requires float32 inputs.
    """
    model = get_model()
    # show_progress_bar=False keeps Streamlit's stdout clean
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS IndexFlatL2 from a precomputed embedding matrix.

    ``IndexFlatL2`` performs exact nearest-neighbour search using squared
    Euclidean distance.  It is appropriate here because our document chunk
    counts are small enough that an approximate index would add complexity
    without a meaningful speed benefit.

    Parameters
    ----------
    embeddings:
        Float32 array of shape ``(n_chunks, embedding_dim)``.

    Returns
    -------
    faiss.IndexFlatL2
        Populated FAISS index ready for querying.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
