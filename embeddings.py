"""
embeddings.py
-------------
Embeddings via fastembed (ONNX Runtime) + FAISS index management.

Uses the same all-MiniLM-L6-v2 model as sentence-transformers but runs it
through ONNX Runtime instead of PyTorch, which cuts runtime memory from
~400 MB to ~100 MB — the difference between fitting on Render's Starter
plan (512 MB) and OOM-ing.
"""

from __future__ import annotations

import numpy as np
import faiss
from fastembed import TextEmbedding

from ingestion import Chunk

_MODEL: TextEmbedding | None = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_model() -> TextEmbedding:
    """Return the process-wide embedding model, loading it on first call.

    fastembed lazily downloads the ONNX weights to a local cache on first
    use (~90 MB). Subsequent instantiations are instant.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = TextEmbedding(model_name=MODEL_NAME)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of plain strings. Used for queries."""
    model = get_model()
    vectors = list(model.embed(texts))
    return np.array(vectors, dtype=np.float32)


def embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    """Embed a list of Chunk objects. Used at index-build time."""
    return embed_texts([c.text for c in chunks])


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
