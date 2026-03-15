"""
app.py
------
Streamlit entry point for the AI-Powered Document Intelligence & Semantic
Search System.

Run with:
    streamlit run app.py

Architecture (RAG pipeline):
    PDF Upload → Text Extraction → Chunking → Embedding → FAISS Index
                                                              ↓
    User Query → Query Embedding → FAISS Search → Top-k Chunks
                                                              ↓
                                         Anthropic API → Grounded Answer
"""

from __future__ import annotations

import datetime
import streamlit as st
from dotenv import load_dotenv

from ingestion import extract_text_from_pdf, chunk_text
from embeddings import embed_chunks, build_faiss_index
from retrieval import retrieve_top_k
from generation import answer_from_context, summarize_document

# ---------------------------------------------------------------------------
# Environment & page config
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark enterprise theme inspired by Claude / ChatGPT
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background-color: #1a1a1a;
        color: #ececec;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #2a2a2a;
    }
    [data-testid="stSidebar"] * {
        color: #d1d1d1 !important;
    }

    /* ── Sidebar file uploader ── */
    [data-testid="stFileUploader"] {
        background-color: #1e1e1e;
        border: 1px dashed #3a3a3a;
        border-radius: 10px;
        padding: 8px;
    }

    /* ── Stat cards in sidebar ── */
    .stat-card {
        background: #1e1e1e;
        border: 1px solid #2e2e2e;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .stat-label {
        font-size: 11px;
        font-weight: 500;
        color: #888 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }
    .stat-value {
        font-size: 15px;
        font-weight: 600;
        color: #ececec !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Primary button (Summarize) ── */
    .stButton > button {
        background: #d97757 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 10px 0 !important;
        transition: background 0.2s ease, transform 0.1s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: #c4623f !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Main area header ── */
    .main-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 28px 0 8px 0;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 24px;
    }
    .main-header h1 {
        font-size: 22px;
        font-weight: 600;
        color: #ececec;
        margin: 0;
    }
    .main-header span {
        font-size: 13px;
        color: #666;
        margin-top: 2px;
    }

    /* ── Empty state ── */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 80px 20px;
        text-align: center;
        color: #555;
    }
    .empty-state .icon {
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.5;
    }
    .empty-state h3 {
        font-size: 18px;
        font-weight: 500;
        color: #777;
        margin-bottom: 8px;
    }
    .empty-state p {
        font-size: 14px;
        color: #555;
        max-width: 340px;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 12px 0 !important;
    }

    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #202020 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 6px !important;
    }

    /* Assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #191919 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 6px !important;
    }

    /* ── Chat input bar ── */
    [data-testid="stChatInput"] {
        background-color: #202020 !important;
        border: 1px solid #333 !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #ececec !important;
        font-size: 14px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #666 !important;
    }

    /* ── Expander (Source passages) ── */
    [data-testid="stExpander"] {
        background: #1e1e1e !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        margin-top: 8px !important;
    }
    [data-testid="stExpander"] summary {
        color: #888 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
    }
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        font-size: 13px !important;
        color: #999 !important;
        line-height: 1.6 !important;
    }

    /* ── Success / info / error banners ── */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        font-size: 13px !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a1a; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #444; }

    /* ── Dividers ── */
    hr { border-color: #2a2a2a !important; }

    /* ── Message timestamp ── */
    .msg-meta {
        font-size: 11px;
        color: #555;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .msg-meta .badge {
        background: #2a2a2a;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 10px;
        font-weight: 500;
        color: #777 !important;
        letter-spacing: 0.03em;
    }

    /* ── Animated thinking dots ── */
    @keyframes blink {
        0%   { opacity: 0.2; }
        20%  { opacity: 1;   }
        100% { opacity: 0.2; }
    }
    .thinking-dot {
        display: inline-block;
        width: 6px; height: 6px;
        background: #d97757;
        border-radius: 50%;
        margin: 0 2px;
        animation: blink 1.4s infinite both;
    }
    .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(3) { animation-delay: 0.4s; }

    /* ── Footer ── */
    .app-footer {
        position: fixed;
        bottom: 70px;
        right: 24px;
        font-size: 10px;
        color: #444;
        display: flex;
        align-items: center;
        gap: 6px;
        pointer-events: none;
    }
    .app-footer .dot {
        width: 5px; height: 5px;
        border-radius: 50%;
        background: #d97757;
        opacity: 0.6;
    }

    /* ── Sidebar title ── */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 16px 0;
    }
    .sidebar-brand .brand-name {
        font-size: 17px;
        font-weight: 600;
        color: #ececec !important;
        letter-spacing: -0.01em;
    }
    .sidebar-brand .brand-tag {
        font-size: 10px;
        font-weight: 500;
        color: #d97757 !important;
        background: #2a1a14;
        border: 1px solid #4a2a1a;
        border-radius: 4px;
        padding: 1px 6px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks: list[str] = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "full_text" not in st.session_state:
    st.session_state.full_text: str = ""

if "page_count" not in st.session_state:
    st.session_state.page_count: int = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if "doc_name" not in st.session_state:
    st.session_state.doc_name: str = ""


# ---------------------------------------------------------------------------
# Helper: reset all document-related state when a new file is uploaded
# ---------------------------------------------------------------------------
def _reset_document_state() -> None:
    """Clear all session-state keys that are tied to the current document.

    Called every time the user uploads a new PDF so that stale index data
    from a previous document cannot contaminate the new session.
    """
    st.session_state.chunks = []
    st.session_state.faiss_index = None
    st.session_state.full_text = ""
    st.session_state.page_count = 0
    st.session_state.chat_history = []
    st.session_state.doc_name = ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # Brand header
    st.markdown(
        """
        <div class="sidebar-brand">
            <span style="font-size:22px;">📄</span>
            <div>
                <div class="brand-name">DocuMind</div>
            </div>
            <span class="brand-tag">RAG</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:0 0 16px 0'>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:11px;font-weight:500;color:#666;text-transform:uppercase;"
        "letter-spacing:0.06em;margin-bottom:8px;'>Document</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="PDF only. Processed entirely in memory.",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.doc_name:
            _reset_document_state()
            st.session_state.doc_name = uploaded_file.name

            pdf_bytes = uploaded_file.read()

            # Stage 1 — extraction & chunking
            with st.spinner("Extracting text…"):
                full_text, page_count = extract_text_from_pdf(pdf_bytes)
                chunks = chunk_text(full_text, chunk_size=500, overlap=50)

            st.session_state.full_text = full_text
            st.session_state.page_count = page_count
            st.session_state.chunks = chunks

            # Stage 2 — embedding & indexing
            with st.spinner(f"Indexing {len(chunks)} chunks…"):
                embeddings = embed_chunks(chunks)
                faiss_index = build_faiss_index(embeddings)

            st.session_state.faiss_index = faiss_index
            st.success("Ready to chat!")

    # Document stats cards
    if st.session_state.faiss_index is not None:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:11px;font-weight:500;color:#666;text-transform:uppercase;"
            "letter-spacing:0.06em;margin-bottom:8px;'>Document Info</p>",
            unsafe_allow_html=True,
        )

        # Truncate long filenames for display
        display_name = st.session_state.doc_name
        if len(display_name) > 28:
            display_name = display_name[:25] + "…"

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">File</div>
                <div class="stat-value" title="{st.session_state.doc_name}">{display_name}</div>
            </div>
            <div style="display:flex;gap:8px;">
                <div class="stat-card" style="flex:1">
                    <div class="stat-label">Pages</div>
                    <div class="stat-value">{st.session_state.page_count}</div>
                </div>
                <div class="stat-card" style="flex:1">
                    <div class="stat-label">Chunks</div>
                    <div class="stat-value">{len(st.session_state.chunks)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Summarize button — injects result into the main chat
        if st.button("Summarize Document", use_container_width=True):
            with st.spinner("Generating summary…"):
                try:
                    summary = summarize_document(st.session_state.full_text)
                    ts_now = datetime.datetime.now().strftime("%I:%M %p")
                    st.session_state.chat_history.append(
                        {
                            "question": "Summarize Document",
                            "answer": summary,
                            "sources": [],
                            "timestamp": ts_now,
                        }
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Summarization failed: {exc}")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <span style="font-size:24px;">💬</span>
        <div>
            <h1>Document Q&A</h1>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.faiss_index is None:
    # Empty state placeholder
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">📂</div>
            <h3>No document loaded</h3>
            <p>Upload a PDF from the sidebar to begin asking questions about its content.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Render existing chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
            ts = entry.get("timestamp", "")
            wc = len(entry["question"].split())
            st.markdown(
                f'<div class="msg-meta">{ts}'
                f'<span class="badge">{wc} words</span></div>',
                unsafe_allow_html=True,
            )

        with st.chat_message("assistant"):
            st.write(entry["answer"])
            answer_wc = len(entry["answer"].split())
            src_count = len(entry["sources"])
            st.markdown(
                f'<div class="msg-meta">{ts}'
                f'<span class="badge">{answer_wc} words</span>'
                f'<span class="badge">{src_count} sources</span></div>',
                unsafe_allow_html=True,
            )
            if entry["sources"]:
                with st.expander("View source passages", expanded=False):
                    for i, src in enumerate(entry["sources"], start=1):
                        st.markdown(f"**Passage {i}**")
                        st.caption(src)
                        if i < len(entry["sources"]):
                            st.markdown("---")

    # Chat input
    user_query = st.chat_input("Ask anything about your document…")

    if user_query:
        stripped_query = user_query.strip()
        if not stripped_query:
            st.warning("Please enter a question.")
        else:
            ts_now = datetime.datetime.now().strftime("%I:%M %p")

            with st.chat_message("user"):
                st.write(stripped_query)
                wc = len(stripped_query.split())
                st.markdown(
                    f'<div class="msg-meta">{ts_now}'
                    f'<span class="badge">{wc} words</span></div>',
                    unsafe_allow_html=True,
                )

            # Stage 3 — retrieval
            relevant_chunks = retrieve_top_k(
                stripped_query,
                st.session_state.faiss_index,
                st.session_state.chunks,
                k=5,
            )

            # Stage 4 — generation
            with st.chat_message("assistant"):
                # Animated thinking indicator while waiting for the API
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown(
                    '<div style="padding:4px 0">'
                    '<span class="thinking-dot"></span>'
                    '<span class="thinking-dot"></span>'
                    '<span class="thinking-dot"></span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                try:
                    answer = answer_from_context(stripped_query, relevant_chunks)
                except Exception as exc:
                    answer = f"An error occurred: {exc}"
                    relevant_chunks = []

                # Replace thinking indicator with the real answer
                thinking_placeholder.empty()
                st.write(answer)

                answer_wc = len(answer.split())
                src_count = len(relevant_chunks)
                st.markdown(
                    f'<div class="msg-meta">{ts_now}'
                    f'<span class="badge">{answer_wc} words</span>'
                    f'<span class="badge">{src_count} sources</span></div>',
                    unsafe_allow_html=True,
                )

                if relevant_chunks:
                    with st.expander("View source passages", expanded=False):
                        for i, src in enumerate(relevant_chunks, start=1):
                            st.markdown(f"**Passage {i}**")
                            st.caption(src)
                            if i < len(relevant_chunks):
                                st.markdown("---")

            st.session_state.chat_history.append(
                {
                    "question": stripped_query,
                    "answer": answer,
                    "sources": relevant_chunks,
                    "timestamp": ts_now,
                }
            )

# Fixed footer — model + version info
st.markdown(
    """
    <div class="app-footer">
        <div class="dot"></div>
        DocuMind &nbsp;·&nbsp; claude-sonnet-4-20250514 &nbsp;·&nbsp; all-MiniLM-L6-v2 &nbsp;·&nbsp; © Korey Dillon 2026
    </div>
    """,
    unsafe_allow_html=True,
)
