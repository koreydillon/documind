"""
sharing.py
----------
Encode/decode compact shareable links for InferLens answers.

A share token is a url-safe base64-encoded JSON blob carrying the sample
document slug and the question text. When visited, InferLens automatically
loads the sample document and runs the question, so a prospect can click
a shared URL and see the exact same answer regenerate in real time.

Only *sample* documents are shareable (they have stable, reproducible
content). Uploaded PDFs cannot be shared since their content is
session-local.
"""

from __future__ import annotations

import base64
import json


def encode_share(doc_slug: str, question: str) -> str:
    payload = json.dumps({"d": doc_slug, "q": question}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_share(token: str) -> tuple[str, str] | None:
    """Return (doc_slug, question) or None on any decode error."""
    try:
        padding = 4 - (len(token) % 4)
        if padding != 4:
            token = token + "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(token).decode())
        return payload["d"], payload["q"]
    except Exception:
        return None
