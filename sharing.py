"""
sharing.py
----------
Encode/decode compact shareable links for InferLens answers.

A share token is a url-safe base64-encoded JSON blob carrying:
    * the kind of document — "sample" or "upload"
    * the reference — sample slug or stored-upload UUID
    * the question text

When visited, InferLens decodes the token, loads the corresponding document
(either a bundled sample or a server-stored upload), and runs the question
so the recipient sees the exact same answer regenerate live.
"""

from __future__ import annotations

import base64
import json


def encode_share(kind: str, doc_ref: str, question: str) -> str:
    """Encode a share token. kind is "sample" or "upload"."""
    payload = json.dumps(
        {"k": kind, "d": doc_ref, "q": question},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_share(token: str) -> tuple[str, str, str] | None:
    """Return (kind, doc_ref, question) or None on any decode error."""
    try:
        padding = 4 - (len(token) % 4)
        if padding != 4:
            token = token + "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(token).decode())
        kind = payload.get("k", "sample")  # legacy tokens default to sample
        return kind, payload["d"], payload["q"]
    except Exception:
        return None
