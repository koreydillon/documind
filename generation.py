"""
generation.py
-------------
Anthropic API wrappers for grounded Q&A and full-document summarisation.

The Q&A prompt now instructs Claude to emit inline ``[1]``, ``[2]`` citation
markers tied to numbered source chunks so the UI can render clickable source
references inline in the answer text.
"""

from __future__ import annotations

import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

from ingestion import Chunk

load_dotenv()

MODEL_ID = "claude-sonnet-4-5-20250929"
MAX_TOKENS_ANSWER = 1024
MAX_TOKENS_SUMMARY = 2048

SUMMARY_CHAR_LIMIT = 40_000


def _get_client() -> anthropic.Anthropic:
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        api_key = None
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file locally, or to Secrets in Streamlit Cloud."
        )
    return anthropic.Anthropic(api_key=api_key)


def answer_from_context(query: str, context_chunks: list[Chunk]) -> str:
    """Generate an answer grounded in the retrieved chunks.

    Claude is instructed to emit ``[N]`` citation markers inline — these are
    1-indexed to match the passage numbering sent in the prompt. The UI then
    renders those markers as clickable references to the source passage list.
    """
    client = _get_client()

    formatted_context = "\n\n".join(
        f"[{i + 1}] (page {chunk.page})\n{chunk.text}"
        for i, chunk in enumerate(context_chunks)
    )

    system_prompt = (
        "You are a precise document-analysis assistant for enterprise users. "
        "Answer the question using ONLY the numbered context passages below. "
        "Cite the passages you used inline with bracketed numbers like [1], [2], [3] "
        "immediately after the claim they support. Multiple citations are fine: [1][3]. "
        "Every factual claim in your answer MUST be followed by a citation marker. "
        "If the context does not contain enough information to answer, reply exactly: "
        "'The document does not contain enough information to answer this question.' "
        "Do not invent facts, page numbers, or passages that were not provided."
    )

    user_message = (
        f"Context passages:\n\n{formatted_context}\n\nQuestion: {query}"
    )

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS_ANSWER,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


def summarize_document(full_text: str) -> str:
    client = _get_client()

    truncated = len(full_text) > SUMMARY_CHAR_LIMIT
    text_to_summarise = full_text[:SUMMARY_CHAR_LIMIT]

    truncation_note = (
        "\n\n[NOTE: The document was truncated to the first ~10 000 words "
        "for this summary. Mention this in your response.]"
        if truncated
        else ""
    )

    system_prompt = (
        "You are an expert document summariser. "
        "Produce a clear, structured summary using the following sections:\n"
        "1. **Overview** — one-paragraph high-level description\n"
        "2. **Key Topics** — bullet-point list of the main subjects covered\n"
        "3. **Important Details** — notable facts, figures, or arguments\n"
        "4. **Conclusion / Takeaways** — what the reader should remember\n\n"
        "Use markdown formatting."
    )

    user_message = (
        f"Please summarise the following document text:{truncation_note}\n\n"
        f"{text_to_summarise}"
    )

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS_SUMMARY,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text
