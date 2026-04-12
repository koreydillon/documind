"""
app.py
------
InferLens — RAG-powered document intelligence for regulated industries.

Architecture:
    PDF Upload → Page-aware extraction → Overlapping chunking
        → Sentence-BERT embeddings → FAISS IndexFlatL2
        → Query embedding → Top-k retrieval
        → Anthropic Claude with inline [N] citations tied to page numbers
"""

from __future__ import annotations

import datetime
import re

import streamlit as st
from dotenv import load_dotenv

from ingestion import Chunk, extract_pages_from_pdf, extract_text_from_pdf, chunk_pages
from embeddings import embed_chunks, build_faiss_index
from retrieval import retrieve_top_k
from generation import answer_from_context, summarize_document, MODEL_ID
from samples import SAMPLES, get_sample, get_sample_pdf_bytes

load_dotenv()

st.set_page_config(
    page_title="InferLens — Document Intelligence",
    page_icon="https://www.latentaxis.io/favicon.ico",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Inline SVG icon helpers — replaces emoji throughout the UI
# ---------------------------------------------------------------------------
def svg(path: str, size: int = 16, color: str = "#d97757") -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round" style="display:inline-block;vertical-align:middle">'
        f"{path}</svg>"
    )


ICON_DOC = svg(
    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
    '<polyline points="14 2 14 8 20 8"/>'
    '<line x1="8" y1="13" x2="16" y2="13"/>'
    '<line x1="8" y1="17" x2="14" y2="17"/>'
)
ICON_CHAT = svg(
    '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
)
ICON_SEARCH = svg(
    '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'
)
ICON_UPLOAD = svg(
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
    '<polyline points="17 8 12 3 7 8"/>'
    '<line x1="12" y1="3" x2="12" y2="15"/>'
)
ICON_SPARK = svg(
    '<polygon points="12 2 15 9 22 9 17 14 19 22 12 17 5 22 7 14 2 9 9 9 12 2"/>'
)
ICON_BOOK = svg(
    '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>'
    '<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>'
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
        background-color: #12161f;
        border: 1px dashed #2a2f3d;
        border-radius: 10px;
        padding: 8px;
    }

    .stat-card {
        background: #12161f;
        border: 1px solid #1f2330;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
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
        background: #d97757 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        padding: 9px 14px !important;
        transition: background 0.15s ease, transform 0.1s ease !important;
    }
    .stButton > button:hover {
        background: #c4623f !important;
        transform: translateY(-1px) !important;
    }

    .main-header {
        display: flex; align-items: center; gap: 12px;
        padding: 28px 0 10px 0;
        border-bottom: 1px solid #1f2330;
        margin-bottom: 20px;
    }
    .main-header h1 {
        font-size: 20px; font-weight: 600; color: #ececec;
        margin: 0; letter-spacing: -0.01em;
    }
    .main-header .sub {
        font-size: 12px; color: #6b7280;
    }

    /* Scenario-framed empty state */
    .hero-card {
        background: linear-gradient(180deg, #12161f 0%, #0d1119 100%);
        border: 1px solid #1f2330;
        border-radius: 16px;
        padding: 32px 36px;
        margin: 20px 0 28px 0;
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

    /* Sample cards grid */
    .samples-label {
        font-size: 10px; font-weight: 600; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin: 28px 0 14px 0;
    }
    .sample-card {
        background: #12161f;
        border: 1px solid #1f2330;
        border-radius: 12px;
        padding: 20px 22px;
        height: 100%;
        transition: border-color 0.15s, transform 0.15s;
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

    /* Suggested question chips */
    .chips-label {
        font-size: 10px; font-weight: 600; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin: 24px 0 10px 0;
    }

    /* Chat messages */
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

    /* Inline citation markers in assistant answers */
    .cite {
        display: inline-block;
        font-size: 10px; font-weight: 700;
        color: #d97757;
        background: #2a1a14;
        border: 1px solid #4a2a1a;
        border-radius: 4px;
        padding: 0 5px;
        margin: 0 2px;
        vertical-align: super;
        line-height: 1.2;
        cursor: help;
    }

    /* Sources block */
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

    /* Perf meta on answers */
    .msg-meta {
        font-size: 11px; color: #6b7280;
        margin-top: 6px; display: flex; align-items: center; gap: 10px;
    }
    .msg-meta .badge {
        background: #141821; border: 1px solid #1f2330;
        border-radius: 4px; padding: 1px 7px;
        font-size: 10px; font-weight: 500;
        color: #9ca3af !important; letter-spacing: 0.03em;
    }

    /* Thinking dots */
    @keyframes blink { 0% { opacity: 0.2; } 20% { opacity: 1; } 100% { opacity: 0.2; } }
    .thinking-dot {
        display: inline-block; width: 6px; height: 6px;
        background: #d97757; border-radius: 50%;
        margin: 0 2px; animation: blink 1.4s infinite both;
    }
    .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(3) { animation-delay: 0.4s; }

    /* Footer */
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

    /* Sidebar brand */
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

    /* Scrollbar */
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
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "chunks": [],
        "faiss_index": None,
        "full_text": "",
        "page_count": 0,
        "chat_history": [],
        "doc_name": "",
        "suggested_questions": [],
        "pending_query": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _reset_document_state() -> None:
    st.session_state.chunks = []
    st.session_state.faiss_index = None
    st.session_state.full_text = ""
    st.session_state.page_count = 0
    st.session_state.chat_history = []
    st.session_state.doc_name = ""
    st.session_state.suggested_questions = []
    st.session_state.pending_query = None


def _ingest_pdf_bytes(pdf_bytes: bytes, doc_name: str) -> None:
    """Run the full ingestion pipeline on a PDF supplied as bytes."""
    _reset_document_state()
    st.session_state.doc_name = doc_name

    with st.spinner("Extracting text..."):
        pages, page_count = extract_pages_from_pdf(pdf_bytes)
        full_text, _ = extract_text_from_pdf(pdf_bytes)
        chunks = chunk_pages(pages, chunk_size=500, overlap=50)

    st.session_state.full_text = full_text
    st.session_state.page_count = page_count
    st.session_state.chunks = chunks

    with st.spinner(f"Indexing {len(chunks)} chunks..."):
        embeddings = embed_chunks(chunks)
        faiss_index = build_faiss_index(embeddings)

    st.session_state.faiss_index = faiss_index


def _load_sample(slug: str) -> None:
    sample = get_sample(slug)
    pdf_bytes = get_sample_pdf_bytes(slug)
    _ingest_pdf_bytes(pdf_bytes, f"{sample.title} (sample)")
    st.session_state.suggested_questions = sample.suggested_questions


# ---------------------------------------------------------------------------
# Citation rendering — replace [N] in answer text with styled superscript
# ---------------------------------------------------------------------------
_CITE_PATTERN = re.compile(r"\[(\d+)\]")


def _render_answer_with_citations(answer: str, sources: list[Chunk]) -> str:
    """Return an HTML-safe string where [N] markers are wrapped as badges."""
    # Escape < and > in the answer to avoid breaking the HTML wrapper. We do
    # not run full HTML escape because we want Claude's markdown to survive.
    safe = answer.replace("<", "&lt;").replace(">", "&gt;")

    def _sub(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= len(sources):
            page = sources[n - 1].page
            return f'<span class="cite" title="Passage {n} · page {page}">[{n}]</span>'
        return m.group(0)

    return _CITE_PATTERN.sub(_sub, safe)


def _render_sources(sources: list[Chunk]) -> str:
    """Return the HTML block listing numbered source passages."""
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
# Sidebar
# ---------------------------------------------------------------------------
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

    st.markdown(
        "<p style='font-size:10px;font-weight:600;color:#6b7280;text-transform:uppercase;"
        "letter-spacing:0.12em;margin-bottom:10px;'>Document</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="PDFs up to 200 MB. Processed entirely in memory — nothing is stored.",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.doc_name:
            _ingest_pdf_bytes(uploaded_file.read(), uploaded_file.name)
            st.session_state.suggested_questions = []
            st.success("Ready.")
            st.rerun()

    # Document stats
    if st.session_state.faiss_index is not None:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:10px;font-weight:600;color:#6b7280;text-transform:uppercase;"
            "letter-spacing:0.12em;margin-bottom:10px;'>Document Info</p>",
            unsafe_allow_html=True,
        )

        display_name = st.session_state.doc_name
        if len(display_name) > 28:
            display_name = display_name[:25] + "..."

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

        if st.button("Summarize Document", use_container_width=True):
            with st.spinner("Generating summary..."):
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

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("Clear document", use_container_width=True):
            _reset_document_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="main-header">
        {ICON_CHAT}
        <div>
            <h1>Document Q&amp;A</h1>
            <div class="sub">Ask anything. Every answer is grounded in your document with page-level citations.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Empty state — scenario hero + sample document cards
# ---------------------------------------------------------------------------
def _render_empty_state():
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">{ICON_SPARK}&nbsp;&nbsp;Try it in 30 seconds</div>
            <h2 class="hero-title">Turn a 100-page document into a conversation.</h2>
            <p class="hero-subtitle">
                Upload a PDF from the sidebar, or pick one of the sample documents below to see InferLens
                extract precise answers with inline page citations. Built for teams that can't afford
                hallucinations — every claim traces back to the exact passage it came from.
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
            if st.button(f"Try this sample", key=f"sample_{sample.slug}", use_container_width=True):
                with st.spinner(f"Loading {sample.title}..."):
                    _load_sample(sample.slug)
                st.rerun()


def _render_suggested_questions():
    if not st.session_state.suggested_questions or st.session_state.chat_history:
        return
    st.markdown('<div class="chips-label">Suggested questions</div>', unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.suggested_questions))
    for col, q in zip(cols, st.session_state.suggested_questions):
        with col:
            if st.button(q, key=f"sq_{hash(q)}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()


def _run_query(query: str):
    ts_now = datetime.datetime.now().strftime("%I:%M %p")
    started = datetime.datetime.now()

    try:
        relevant_chunks = retrieve_top_k(
            query,
            st.session_state.faiss_index,
            st.session_state.chunks,
            k=5,
        )
        answer = answer_from_context(query, relevant_chunks)
    except Exception as exc:
        answer = f"An error occurred: {exc}"
        relevant_chunks = []

    elapsed_ms = int((datetime.datetime.now() - started).total_seconds() * 1000)

    st.session_state.chat_history.append(
        {
            "question": query,
            "answer": answer,
            "sources": relevant_chunks,
            "timestamp": ts_now,
            "elapsed_ms": elapsed_ms,
        }
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
if st.session_state.faiss_index is None:
    _render_empty_state()
else:
    # Render chat history
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

    _render_suggested_questions()

    # Process pending query (from suggested-question button click)
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        _run_query(query)
        st.rerun()

    # Chat input
    user_query = st.chat_input("Ask anything about your document...")
    if user_query and user_query.strip():
        _run_query(user_query.strip())
        st.rerun()


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
