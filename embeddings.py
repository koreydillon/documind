"""
embeddings.py
-------------
Sentence-BERT embeddings + FAISS index management.
"""

from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ingestion import Chunk

_MODEL: SentenceTransformer | None = None
MODEL_NAME = "all-MiniLM-L6-v2"


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of plain strings (used for queries)."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    """Embed a list of Chunk objects (used at index-build time)."""
    return embed_texts([c.text for c in chunks])


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
