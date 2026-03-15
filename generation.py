"""
generation.py
-------------
Wraps all calls to the Anthropic API.

Two entry points are provided:
  * ``answer_from_context``  — RAG-style Q&A grounded in retrieved chunks.
  * ``summarize_document``   — Free-form structured summary of a full document.
"""

from __future__ import annotations

import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

# Load from .env when running locally (no-op if file doesn't exist)
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "claude-sonnet-4-20250514"
MAX_TOKENS_ANSWER = 1024
MAX_TOKENS_SUMMARY = 2048

# Summarisation truncation: Claude's context window is large but we cap the
# text sent to avoid excessive token costs on very large PDFs.
SUMMARY_CHAR_LIMIT = 40_000  # ~10 000 words — enough for a solid summary


def _get_client() -> anthropic.Anthropic:
    """Instantiate and return an Anthropic client.

    The API key is read from the ``ANTHROPIC_API_KEY`` environment variable
    (populated via python-dotenv from ``.env``).

    Returns
    -------
    anthropic.Anthropic
        Configured Anthropic SDK client.

    Raises
    ------
    EnvironmentError
        If the environment variable is absent.
    """
    # Prefer Streamlit secrets (used on Streamlit Cloud), fall back to env var
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file locally, or to Secrets in Streamlit Cloud."
        )
    return anthropic.Anthropic(api_key=api_key)


def answer_from_context(query: str, context_chunks: list[str]) -> str:
    """Generate a grounded answer using RAG-retrieved chunks as context.

    The system prompt instructs the model to answer *only* from the supplied
    context and to cite the relevant passage when possible.  If the context
    does not contain enough information the model is instructed to say so
    explicitly rather than hallucinating.

    Parameters
    ----------
    query:
        The user's natural-language question.
    context_chunks:
        List of document chunks retrieved by the FAISS nearest-neighbour
        search (ordered most-relevant first).

    Returns
    -------
    str
        The model's plain-text answer.
    """
    client = _get_client()

    # Format the retrieved chunks as a numbered list for easy citation
    formatted_context = "\n\n".join(
        f"[Passage {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    system_prompt = (
        "You are a precise document-analysis assistant. "
        "Answer the question using ONLY the context passages provided below. "
        "If the answer is not contained in the context, say explicitly: "
        "'The document does not contain enough information to answer this question.' "
        "When your answer is supported by a specific passage, reference it as "
        "'(Passage N)' at the end of the relevant sentence."
    )

    user_message = (
        f"Context passages from the document:\n\n"
        f"{formatted_context}\n\n"
        f"Question: {query}"
    )

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS_ANSWER,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


def summarize_document(full_text: str) -> str:
    """Produce a structured summary of *full_text* using the Anthropic API.

    Very long documents are truncated to ``SUMMARY_CHAR_LIMIT`` characters
    before being sent.  A note is appended to the prompt when truncation
    occurs so the model can qualify its summary accordingly.

    Parameters
    ----------
    full_text:
        The complete extracted text of the uploaded PDF document.

    Returns
    -------
    str
        A structured plain-text summary with clearly labelled sections.
    """
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
