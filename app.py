"""
app.py
------
InferLens — RAG-powered document intelligence for regulated industries.

Features:
    * Multi-document knowledge bases with per-KB chat history
    * Streaming Claude responses rendered live then re-formatted with citations
    * Page-aware chunking; inline [N] citation markers tied to source pages
    * Sample documents with scenario framing and suggested questions
    * Shareable answer URLs for sample docs (base64 query param)
    * Email gate + per-email daily rate limit
    * SQLite analytics logging + /?admin=TOKEN admin view
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv

from ingestion import Chunk, extract_pages_from_pdf, extract_text_from_pdf, chunk_pages
from embeddings import embed_chunks, build_faiss_index
from retrieval import retrieve_top_k
from generation import (
    stream_answer_from_context,
    summarize_document,
    MODEL_ID,
)
from samples import SAMPLES, get_sample, get_sample_pdf_bytes
import analytics
import sharing

load_dotenv()

# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="InferLens — Document Intelligence",
    page_icon="https://www.latentaxis.io/favicon.ico",
    layout="wide",
)

analytics.init_db()

ADMIN_TOKEN = os.getenv("INFERLENS_ADMIN_TOKEN", "")
PUBLIC_BASE_URL = os.getenv("INFERLENS_PUBLIC_URL", "https://inferlens.latentaxis.io")


# ---------------------------------------------------------------------------
# SVG icon helpers
# ---------------------------------------------------------------------------
def svg(path: str, size: int = 16, color: str = "#d97757") -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round" style="display:inline-block;vertical-align:middle">'
        f"{path}</svg>"
    )


ICON_CHAT = svg('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>')
ICON_SPARK = svg('<polygon points="12 2 15 9 22 9 17 14 19 22 12 17 5 22 7 14 2 9 9 9 12 2"/>')
ICON_LINK = svg(
    '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>'
    '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>'
)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0B0F19; color: #ececec; }

    [data-testid="stSidebar"] { background-color: #0a0d16; border-right: 1px solid #1f2330; }
    [data-testid="stSidebar"] * { color: #d1d1d1 !important; }

    [data-testid="stFileUploader"] {
        background-color: #12161f; border: 1px dashed #2a2f3d;
        border-radius: 10px; padding: 8px;
    }

    .stat-card {
        background: #12161f; border: 1px solid #1f2330;
        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    }
    .stat-label {
        font-size: 10px; font-weight: 600; color: #6b7280 !important;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
    }
    .stat-value {
        font-size: 15px; font-weight: 600; color: #ececec !important;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }

    .stButton > button {
        background: #d97757 !important; color: #fff !important;
        border: none !important; border-radius: 8px !important;
        font-weight: 500 !important; font-size: 13px !important;
        padding: 9px 14px !important;
        transition: background 0.15s, transform 0.1s !important;
    }
    .stButton > button:hover {
        background: #c4623f !important; transform: translateY(-1px) !important;
    }
    .stButton > button:disabled {
        background: #2a2f3d !important; color: #6b7280 !important; cursor: not-allowed;
    }

    /* Active KB button variant (ghost style) */
    [data-testid="stSidebar"] .stButton > button {
        background: #12161f !important; color: #d1d1d1 !important;
        border: 1px solid #1f2330 !important; text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1a1f2b !important; border-color: #3a3f4d !important;
    }

    .main-header {
        display: flex; align-items: center; gap: 12px;
        padding: 28px 0 10px 0; border-bottom: 1px solid #1f2330; margin-bottom: 20px;
    }
    .main-header h1 {
        font-size: 20px; font-weight: 600; color: #ececec;
        margin: 0; letter-spacing: -0.01em;
    }
    .main-header .sub { font-size: 12px; color: #6b7280; }

    .hero-card {
        background: linear-gradient(180deg, #12161f 0%, #0d1119 100%);
        border: 1px solid #1f2330; border-radius: 16px;
        padding: 32px 36px; margin: 20px 0 28px 0;
    }
    .hero-kicker {
        font-size: 10px; font-weight: 600; color: #d97757;
        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;
    }
    .hero-title {
        font-size: 26px; font-weight: 600; color: #ececec;
        letter-spacing: -0.01em; margin: 0 0 10px 0; line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 14px; color: #9ca3af; line-height: 1.6;
        max-width: 640px; margin: 0;
    }

    .samples-label {
        font-size: 10px; font-weight: 600; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin: 28px 0 14px 0;
    }
    .sample-card {
        background: #12161f; border: 1px solid #1f2330;
        border-radius: 12px; padding: 20px 22px; height: 100%;
        transition: border-color 0.15s;
    }
    .sample-card:hover { border-color: #3a3f4d; }
    .sample-card .tag {
        display: inline-block;
        font-size: 10px; font-weight: 600;
        color: #d97757; background: #2a1a14;
        border: 1px solid #4a2a1a; border-radius: 4px;
        padding: 2px 7px; letter-spacing: 0.05em;
        text-transform: uppercase; margin-bottom: 10px;
    }
    .sample-card .title {
        font-size: 14px; font-weight: 600; color: #ececec;
        margin: 0 0 8px 0; line-height: 1.35;
    }
    .sample-card .scenario {
        font-size: 12px; color: #9ca3af; line-height: 1.55;
        margin: 0 0 14px 0;
    }

    .chips-label {
        font-size: 10px; font-weight: 600; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin: 24px 0 10px 0;
    }

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 12px 0 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #141821 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 6px !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #10141c !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 6px !important;
    }
    [data-testid="stChatInput"] {
        background-color: #141821 !important;
        border: 1px solid #1f2330 !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important; color: #ececec !important; font-size: 14px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: #6b7280 !important; }

    .cite {
        display: inline-block; font-size: 10px; font-weight: 700;
        color: #d97757; background: #2a1a14;
        border: 1px solid #4a2a1a; border-radius: 4px;
        padding: 0 5px; margin: 0 2px;
        vertical-align: super; line-height: 1.2;
        cursor: help;
    }

    .sources-block {
        margin-top: 14px; padding-top: 14px;
        border-top: 1px solid #1f2330;
    }
    .sources-label {
        font-size: 10px; font-weight: 600; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 10px;
    }
    .source-row {
        display: flex; gap: 10px; margin-bottom: 10px;
        font-size: 12px; color: #9ca3af; line-height: 1.55;
    }
    .source-num {
        flex-shrink: 0;
        font-size: 10px; font-weight: 700;
        color: #d97757; background: #2a1a14;
        border: 1px solid #4a2a1a; border-radius: 4px;
        padding: 1px 6px; height: fit-content;
    }
    .source-meta {
        font-size: 10px; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 3px;
    }

    .msg-meta {
        font-size: 11px; color: #6b7280;
        margin-top: 6px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    }
    .msg-meta .badge {
        background: #141821; border: 1px solid #1f2330;
        border-radius: 4px; padding: 1px 7px;
        font-size: 10px; font-weight: 500;
        color: #9ca3af !important; letter-spacing: 0.03em;
    }

    .app-footer {
        position: fixed; bottom: 70px; right: 24px;
        font-size: 10px; color: #6b7280;
        display: flex; align-items: center; gap: 6px;
        pointer-events: none;
    }
    .app-footer .dot {
        width: 5px; height: 5px; border-radius: 50%;
        background: #d97757; opacity: 0.7;
    }

    .sidebar-brand {
        display: flex; align-items: center; gap: 10px;
        padding: 4px 0 16px 0;
    }
    .sidebar-brand .brand-name {
        font-size: 17px; font-weight: 600; color: #ececec !important;
        letter-spacing: -0.01em;
    }
    .sidebar-brand .brand-tag {
        font-size: 10px; font-weight: 500; color: #d97757 !important;
        background: #2a1a14; border: 1px solid #4a2a1a; border-radius: 4px;
        padding: 1px 6px; text-transform: uppercase; letter-spacing: 0.06em;
    }

    .kb-row-active {
        background: #1a1f2b !important;
        border-color: #d97757 !important;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0B0F19; }
    ::-webkit-scrollbar-thumb { background: #1f2330; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #2a2f3d; }

    hr { border-color: #1f2330 !important; }
    [data-testid="stAlert"] { border-radius: 8px !important; font-size: 13px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "kbs": {},                    # slug -> kb dict
        "active_kb_slug": None,
        "session_id": uuid.uuid4().hex,
        "email": None,
        "pending_query": None,
        "pending_share": None,        # (doc_slug, question) from share URL
        "share_processed": False,
        "show_share_for_entry": None, # index of chat entry to show share UI for
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Query param handling — process on first run
# ---------------------------------------------------------------------------
def _process_query_params():
    params = st.query_params
    share_token = params.get("s")
    if share_token and not st.session_state.share_processed:
        decoded = sharing.decode_share(share_token)
        if decoded:
            st.session_state.pending_share = decoded
        st.session_state.share_processed = True
        # Clear the share param so it doesn't re-process on rerun
        try:
            del st.query_params["s"]
        except KeyError:
            pass


_process_query_params()

_is_admin = bool(ADMIN_TOKEN) and st.query_params.get("admin") == ADMIN_TOKEN


# ---------------------------------------------------------------------------
# KB helpers
# ---------------------------------------------------------------------------
def _kb_from_bytes(
    slug: str,
    name: str,
    source: str,
    pdf_bytes: bytes,
    suggested_questions: list[str] | None = None,
) -> dict:
    pages, page_count = extract_pages_from_pdf(pdf_bytes)
    full_text, _ = extract_text_from_pdf(pdf_bytes)
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    embeddings = embed_chunks(chunks)
    faiss_index = build_faiss_index(embeddings)
    return {
        "slug": slug,
        "name": name,
        "source": source,
        "chunks": chunks,
        "faiss_index": faiss_index,
        "full_text": full_text,
        "page_count": page_count,
        "suggested_questions": suggested_questions or [],
        "chat": [],
    }


def _add_sample_kb(slug: str) -> None:
    if slug in st.session_state.kbs:
        st.session_state.active_kb_slug = slug
        return
    sample = get_sample(slug)
    pdf_bytes = get_sample_pdf_bytes(slug)
    kb = _kb_from_bytes(
        slug=slug,
        name=sample.title,
        source="sample",
        pdf_bytes=pdf_bytes,
        suggested_questions=sample.suggested_questions,
    )
    st.session_state.kbs[slug] = kb
    st.session_state.active_kb_slug = slug


def _add_upload_kb(filename: str, pdf_bytes: bytes) -> None:
    slug = f"upload_{uuid.uuid4().hex[:8]}"
    kb = _kb_from_bytes(
        slug=slug, name=filename, source="upload", pdf_bytes=pdf_bytes,
    )
    st.session_state.kbs[slug] = kb
    st.session_state.active_kb_slug = slug


def _remove_kb(slug: str) -> None:
    st.session_state.kbs.pop(slug, None)
    if st.session_state.active_kb_slug == slug:
        remaining = list(st.session_state.kbs.keys())
        st.session_state.active_kb_slug = remaining[0] if remaining else None


def _active_kb() -> dict | None:
    slug = st.session_state.active_kb_slug
    return st.session_state.kbs.get(slug) if slug else None


# ---------------------------------------------------------------------------
# Citation rendering
# ---------------------------------------------------------------------------
_CITE_PATTERN = re.compile(r"\[(\d+)\]")


def _render_answer_with_citations(answer: str, sources: list[Chunk]) -> str:
    safe = answer.replace("<", "&lt;").replace(">", "&gt;")

    def _sub(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= len(sources):
            page = sources[n - 1].page
            return f'<span class="cite" title="Passage {n} · page {page}">[{n}]</span>'
        return m.group(0)

    return _CITE_PATTERN.sub(_sub, safe)


def _render_sources(sources: list[Chunk]) -> str:
    if not sources:
        return ""
    rows = []
    for i, chunk in enumerate(sources, start=1):
        preview = chunk.text.strip()
        if len(preview) > 380:
            preview = preview[:380] + "..."
        preview = preview.replace("<", "&lt;").replace(">", "&gt;")
        rows.append(
            f'<div class="source-row">'
            f'<span class="source-num">{i}</span>'
            f'<div>'
            f'<div class="source-meta">Page {chunk.page}</div>'
            f'<div>{preview}</div>'
            f'</div></div>'
        )
    return (
        '<div class="sources-block">'
        '<div class="sources-label">Sources</div>'
        + "".join(rows)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Email gate (Streamlit dialog)
# ---------------------------------------------------------------------------
@st.dialog("Welcome to InferLens")
def _email_gate():
    st.markdown(
        "Enter your email to continue. We use it to rate-limit demo usage "
        "(25 queries / 24 hours) and occasionally reach out about enterprise "
        "deployments. No spam — promise."
    )
    email_input = st.text_input("Work email", key="email_input_field")
    if st.button("Continue", use_container_width=True):
        cleaned = (email_input or "").strip().lower()
        if "@" not in cleaned or "." not in cleaned.split("@")[-1]:
            st.error("Please enter a valid email.")
            return
        st.session_state.email = cleaned
        analytics.upsert_session(cleaned)
        st.rerun()


# ---------------------------------------------------------------------------
# Admin view
# ---------------------------------------------------------------------------
def _render_admin():
    st.markdown(
        f"""
        <div class="main-header">
            {ICON_SPARK}
            <div><h1>InferLens Admin</h1>
            <div class="sub">Usage analytics for the live demo.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    stats = analytics.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total queries", stats["total_queries"])
    c2.metric("Unique emails", stats["unique_emails"])
    c3.metric("Queries (24h)", stats["queries_last_24h"])
    c4.metric("Avg latency", f"{stats['avg_latency_ms']} ms")

    st.markdown("### Top documents")
    if stats["top_docs"]:
        st.dataframe(
            {"document": [d for d, _ in stats["top_docs"]],
             "queries": [c for _, c in stats["top_docs"]]},
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No queries yet.")

    st.markdown("### Recent queries")
    if stats["recent"]:
        st.dataframe(stats["recent"], use_container_width=True, hide_index=True)
    else:
        st.caption("No queries yet.")


# ---------------------------------------------------------------------------
# Query execution (streaming)
# ---------------------------------------------------------------------------
def _run_query(query: str, kb: dict) -> None:
    # Rate limit check
    email = st.session_state.get("email")
    if email:
        limited, count = analytics.is_rate_limited(email)
        if limited:
            st.error(
                f"Daily rate limit reached ({count} queries in the last 24h). "
                f"Try again tomorrow or contact us for extended demo access."
            )
            return

    ts_now = _dt.datetime.now().strftime("%I:%M %p")
    started = _dt.datetime.now()

    # Retrieval
    try:
        relevant_chunks = retrieve_top_k(
            query, kb["faiss_index"], kb["chunks"], k=5,
        )
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        return

    # Render user message + streaming assistant message
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        try:
            for text_chunk in stream_answer_from_context(query, relevant_chunks):
                full_answer += text_chunk
                placeholder.markdown(full_answer + " ▌")
        except Exception as exc:
            full_answer = f"An error occurred while generating the answer: {exc}"
            relevant_chunks = []

        # Final rewrite with citation badges + sources block
        rendered = _render_answer_with_citations(full_answer, relevant_chunks)
        sources_html = _render_sources(relevant_chunks)
        placeholder.markdown(rendered + sources_html, unsafe_allow_html=True)

    elapsed_ms = int((_dt.datetime.now() - started).total_seconds() * 1000)

    # Append to KB chat history
    kb["chat"].append(
        {
            "question": query,
            "answer": full_answer,
            "sources": relevant_chunks,
            "timestamp": ts_now,
            "elapsed_ms": elapsed_ms,
        }
    )

    # Log analytics
    try:
        analytics.log_query(
            session_id=st.session_state.session_id,
            email=email,
            doc_slug=kb["slug"],
            question=query,
            answer_length=len(full_answer),
            source_count=len(relevant_chunks),
            latency_ms=elapsed_ms,
        )
    except Exception:
        pass  # never let analytics failure break the UX

    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar():
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                {svg(
                    '<rect x="3" y="3" width="18" height="18" rx="4"/>'
                    '<path d="M9 9h6v6H9z"/>',
                    size=22,
                )}
                <div class="brand-name">InferLens</div>
                <span class="brand-tag">RAG</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:0 0 16px 0'>", unsafe_allow_html=True)

        # Knowledge base list
        if st.session_state.kbs:
            st.markdown(
                "<p style='font-size:10px;font-weight:600;color:#6b7280;text-transform:uppercase;"
                "letter-spacing:0.12em;margin-bottom:10px;'>Knowledge Base</p>",
                unsafe_allow_html=True,
            )
            for slug, kb in list(st.session_state.kbs.items()):
                is_active = slug == st.session_state.active_kb_slug
                display_name = kb["name"]
                if len(display_name) > 24:
                    display_name = display_name[:21] + "..."
                label_prefix = "● " if is_active else "○ "
                cols = st.columns([4, 1])
                with cols[0]:
                    if st.button(
                        f"{label_prefix}{display_name}",
                        key=f"kb_select_{slug}",
                        use_container_width=True,
                    ):
                        st.session_state.active_kb_slug = slug
                        st.rerun()
                with cols[1]:
                    if st.button("×", key=f"kb_remove_{slug}", use_container_width=True):
                        _remove_kb(slug)
                        st.rerun()

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Add document section
        st.markdown(
            "<p style='font-size:10px;font-weight:600;color:#6b7280;text-transform:uppercase;"
            "letter-spacing:0.12em;margin-bottom:10px;'>Add Document</p>",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            label_visibility="collapsed",
            help="PDFs up to 200 MB. Processed in memory — nothing is stored.",
            key="pdf_uploader",
        )
        if uploaded_file is not None:
            existing_upload = any(
                kb["source"] == "upload" and kb["name"] == uploaded_file.name
                for kb in st.session_state.kbs.values()
            )
            if not existing_upload:
                with st.spinner("Indexing upload..."):
                    _add_upload_kb(uploaded_file.name, uploaded_file.read())
                st.rerun()

        # Active KB stats + actions
        kb = _active_kb()
        if kb is not None:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:10px;font-weight:600;color:#6b7280;text-transform:uppercase;"
                "letter-spacing:0.12em;margin-bottom:10px;'>Active Document</p>",
                unsafe_allow_html=True,
            )
            display_name = kb["name"]
            if len(display_name) > 28:
                display_name = display_name[:25] + "..."
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">File</div>
                    <div class="stat-value" title="{kb['name']}">{display_name}</div>
                </div>
                <div style="display:flex;gap:8px;">
                    <div class="stat-card" style="flex:1">
                        <div class="stat-label">Pages</div>
                        <div class="stat-value">{kb['page_count']}</div>
                    </div>
                    <div class="stat-card" style="flex:1">
                        <div class="stat-label">Chunks</div>
                        <div class="stat-value">{len(kb['chunks'])}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("Summarize this document", use_container_width=True):
                with st.spinner("Generating summary..."):
                    try:
                        summary = summarize_document(kb["full_text"])
                        kb["chat"].append(
                            {
                                "question": "Summarize Document",
                                "answer": summary,
                                "sources": [],
                                "timestamp": _dt.datetime.now().strftime("%I:%M %p"),
                                "elapsed_ms": None,
                            }
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Summarization failed: {exc}")

        # Usage display
        if st.session_state.get("email"):
            _, count = analytics.is_rate_limited(st.session_state.email)
            remaining = max(0, analytics.RATE_LIMIT_PER_DAY - count)
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='font-size:10px;color:#6b7280;text-align:center;'>"
                f"{remaining} of {analytics.RATE_LIMIT_PER_DAY} queries remaining today"
                f"</p>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def _render_empty_state():
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">{ICON_SPARK}&nbsp;&nbsp;Try it in 30 seconds</div>
            <h2 class="hero-title">Turn a 100-page document into a conversation.</h2>
            <p class="hero-subtitle">
                Pick a sample below or upload your own PDF. Every answer is grounded in the source
                document with inline page citations — built for teams that can't afford hallucinations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="samples-label">Sample documents</div>', unsafe_allow_html=True)

    cols = st.columns(len(SAMPLES))
    for col, sample in zip(cols, SAMPLES):
        with col:
            st.markdown(
                f"""
                <div class="sample-card">
                    <span class="tag">{sample.description}</span>
                    <p class="title">{sample.title}</p>
                    <p class="scenario">{sample.scenario}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Try this sample", key=f"sample_{sample.slug}", use_container_width=True):
                with st.spinner(f"Loading {sample.title}..."):
                    _add_sample_kb(sample.slug)
                st.rerun()


def _render_chat_history(kb: dict) -> None:
    for entry_idx, entry in enumerate(kb["chat"]):
        with st.chat_message("user"):
            st.write(entry["question"])
            ts = entry.get("timestamp", "")
            wc = len(entry["question"].split())
            st.markdown(
                f'<div class="msg-meta">{ts}<span class="badge">{wc} words</span></div>',
                unsafe_allow_html=True,
            )

        with st.chat_message("assistant"):
            rendered = _render_answer_with_citations(entry["answer"], entry.get("sources", []))
            sources_html = _render_sources(entry.get("sources", []))
            st.markdown(rendered + sources_html, unsafe_allow_html=True)

            answer_wc = len(entry["answer"].split())
            src_count = len(entry.get("sources", []))
            elapsed = entry.get("elapsed_ms")
            elapsed_badge = f'<span class="badge">{elapsed} ms</span>' if elapsed else ""
            st.markdown(
                f'<div class="msg-meta">{entry.get("timestamp", "")}'
                f'<span class="badge">{answer_wc} words</span>'
                f'<span class="badge">{src_count} sources</span>'
                f"{elapsed_badge}</div>",
                unsafe_allow_html=True,
            )

            # Share button — only for sample KBs
            if kb["source"] == "sample":
                share_col1, share_col2 = st.columns([1, 5])
                with share_col1:
                    if st.button("Share", key=f"share_{kb['slug']}_{entry_idx}"):
                        st.session_state.show_share_for_entry = (kb["slug"], entry_idx)
                if st.session_state.show_share_for_entry == (kb["slug"], entry_idx):
                    token = sharing.encode_share(kb["slug"], entry["question"])
                    share_url = f"{PUBLIC_BASE_URL}/?s={token}"
                    st.code(share_url, language=None)
                    st.caption("Copy this link — anyone who opens it will see the same answer regenerate.")


def _render_suggested_questions(kb: dict) -> None:
    if not kb["suggested_questions"] or kb["chat"]:
        return
    st.markdown('<div class="chips-label">Suggested questions</div>', unsafe_allow_html=True)
    cols = st.columns(len(kb["suggested_questions"]))
    for col, q in zip(cols, kb["suggested_questions"]):
        with col:
            if st.button(q, key=f"sq_{kb['slug']}_{hash(q)}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()


# ---------------------------------------------------------------------------
# Admin short-circuit
# ---------------------------------------------------------------------------
if _is_admin:
    _render_admin()
    st.stop()


# ---------------------------------------------------------------------------
# Email gate — required before any queries
# ---------------------------------------------------------------------------
if not st.session_state.email:
    _email_gate()
    st.stop()


# ---------------------------------------------------------------------------
# Handle pending share link (auto-load sample + queue question)
# ---------------------------------------------------------------------------
if st.session_state.pending_share:
    share_doc, share_q = st.session_state.pending_share
    st.session_state.pending_share = None
    try:
        _add_sample_kb(share_doc)
        st.session_state.pending_query = share_q
        st.rerun()
    except Exception as exc:
        st.warning(f"Could not load shared demo: {exc}")


# ---------------------------------------------------------------------------
# Sidebar + header
# ---------------------------------------------------------------------------
_render_sidebar()

st.markdown(
    f"""
    <div class="main-header">
        {ICON_CHAT}
        <div>
            <h1>Document Q&amp;A</h1>
            <div class="sub">Ask anything. Every answer is grounded with page-level citations.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
active = _active_kb()

if active is None:
    _render_empty_state()
else:
    _render_chat_history(active)
    _render_suggested_questions(active)

    # Process pending query (from suggested question button or share link)
    if st.session_state.pending_query:
        q = st.session_state.pending_query
        st.session_state.pending_query = None
        _run_query(q, active)

    # Chat input
    user_query = st.chat_input("Ask anything about your document...")
    if user_query and user_query.strip():
        _run_query(user_query.strip(), active)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-footer">
        <div class="dot"></div>
        InferLens &nbsp;·&nbsp; {MODEL_ID} &nbsp;·&nbsp; all-MiniLM-L6-v2 &nbsp;·&nbsp; latentaxis.io
    </div>
    """,
    unsafe_allow_html=True,
)
