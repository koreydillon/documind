# AI-Powered Document Intelligence & Semantic Search

A Retrieval-Augmented Generation (RAG) application built as an academic capstone project.  Upload any PDF, ask natural-language questions, and receive grounded answers backed by semantic search over the document.

## Architecture

```
PDF Upload
   │
   ▼
Text Extraction (PyMuPDF)
   │
   ▼
Chunking (~500-word overlapping windows)
   │
   ▼
Embedding (all-MiniLM-L6-v2 via sentence-transformers)
   │
   ▼
FAISS IndexFlatL2  ◄──── User Query (embedded with same model)
   │                              │
   └──── Top-5 Nearest Chunks ────┘
                │
                ▼
       Anthropic Claude (RAG prompt)
                │
                ▼
         Grounded Answer + Source passages
```

## Project Structure

```
capstone/
├── app.py          # Streamlit UI and pipeline orchestration
├── ingestion.py    # PDF extraction and text chunking
├── embeddings.py   # Sentence-BERT model and FAISS index construction
├── retrieval.py    # Query embedding and nearest-neighbour search
├── generation.py   # Anthropic API calls (Q&A and summarisation)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Clone / enter the project directory

```bash
cd capstone
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
# Open .env and replace 'your_key_here' with your actual Anthropic API key
```

### 5. Run the application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## Usage

1. **Upload a PDF** using the sidebar uploader.  The document will be extracted, chunked, and indexed automatically (a spinner shows progress).
2. **Check document stats** — page count and chunk count are shown in the sidebar once indexing completes.
3. **Ask questions** in the main chat input.  Each answer includes a collapsed "Source passage(s)" expander showing the exact chunks retrieved.
4. **Summarize** the entire document by clicking the "Summarize Document" button in the sidebar.

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pymupdf` | PDF text extraction |
| `sentence-transformers` | all-MiniLM-L6-v2 embedding model |
| `faiss-cpu` | Approximate/exact nearest-neighbour search |
| `anthropic` | Claude API client |
| `python-dotenv` | `.env` file loading |

## Notes

- The FAISS index and chat history are stored in `st.session_state` and persist across reruns within the same browser session.
- Very large PDFs are truncated to ~10 000 words for the summarisation feature to control token costs; Q&A is unaffected as it only sends the top-5 retrieved chunks.
- The model used for generation is `claude-sonnet-4-20250514`.
