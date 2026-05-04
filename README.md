# InferLens

> Ask any PDF a question. Get a grounded answer with the source passages cited.

**Live demo:** [inferlens.latentaxis.io](https://inferlens.latentaxis.io)

A Retrieval-Augmented Generation (RAG) app that started as an academic capstone and grew into a deployed product. Upload a PDF, ask natural-language questions, and InferLens returns answers backed by semantic search over the document, with the exact retrieved chunks shown beneath each answer.

## Try it in 30 seconds

The live demo at [inferlens.latentaxis.io](https://inferlens.latentaxis.io) ships with three pre-loaded sample documents (a lease, a research paper, a contract) so you can ask questions immediately without uploading anything. Each sample comes with three suggested questions a real evaluator would ask.

## Architecture

```
PDF Upload
   │
   ▼
Text Extraction  (PyMuPDF)
   │
   ▼
Chunking         (~500-word overlapping windows)
   │
   ▼
Embedding        (all-MiniLM-L6-v2 via sentence-transformers)
   │
   ▼
FAISS IndexFlatL2  ◄──── User Query (embedded with same model)
   │                              │
   └──── Top-5 Nearest Chunks ────┘
                │
                ▼
       Anthropic Claude (RAG prompt with citations)
                │
                ▼
   Grounded Answer + Source passages + Share link
```

Embeddings stay in memory per session. SQLite handles analytics, rate limiting, and 30-day shared-document storage. Deployed on Render with a persistent disk mounted at `/var/data` for the SQLite database.

## What's in here

| File | What it does |
|---|---|
| `app.py` | Streamlit UI, session state, pipeline orchestration |
| `ingestion.py` | PDF text extraction, chunking |
| `embeddings.py` | Sentence-BERT model + FAISS index construction |
| `retrieval.py` | Query embedding + top-K nearest-chunk search |
| `generation.py` | Claude API calls (Q&A + document summarization) |
| `analytics.py` | SQLite query log, session table, shared-doc store, monthly rate limit |
| `samples.py` | Three pre-loaded demo PDFs generated at runtime via PyMuPDF |
| `sharing.py` | Encode/decode shareable answer links (base64-JSON tokens) |
| `Dockerfile` / `render.yaml` | Container build + Render deployment config |

## Run it locally

```bash
git clone https://github.com/koreydillon/inferlens.git
cd inferlens
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
streamlit run app.py
```

Open http://localhost:8501.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | Streamlit | Fastest path from notebook to deployable web app |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) | Tiny (~80MB), fast on CPU, great quality-per-byte for English |
| Vector index | FAISS `IndexFlatL2` | Exact search; per-document indexes are small enough to skip approximation |
| Generator | Claude (`claude-sonnet-4-5-20250929`) | Strong instruction-following + grounded answer behavior |
| Storage | SQLite | Single-file, zero-ops, plenty for analytics + share-link TTL |
| Hosting | Render (Docker) | Persistent disk for SQLite, predictable cold starts |

## What I learned

- **Chunking dominates.** 500-word overlapping windows beat both smaller (200) and larger (1000) windows on retrieval quality for legal/research PDFs. Smaller chunks fragmented multi-paragraph reasoning; larger chunks diluted relevance.
- **Show the receipts.** Surfacing the retrieved chunks under each answer turned the app from "trust me" to "verify it." Completely changed how testers used it.
- **Ship the samples.** A blank "upload your PDF" screen has bounce. Three pre-loaded documents with suggested questions cut time-to-aha to under a minute.
- **Rate limit by email, not IP.** Cloud IPs are shared (everyone behind a corporate NAT looks like one user). Email-cookie limits gave a much better signal at minimal friction.

## Roadmap

- Hybrid retrieval (BM25 + dense) for queries that lean on rare terms
- Per-document index persistence so re-uploads don't re-embed
- Streaming answers (currently buffered)
- Multi-PDF cross-document Q&A

## License

MIT. See [LICENSE](LICENSE).

---

Built by [Korey Dillon](https://latentaxis.io) at [LatentAxis](https://latentaxis.io).
