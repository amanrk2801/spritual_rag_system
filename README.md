# Dharma Sahayak — Spiritual Scripture RAG

Ask questions in **Hindi, English or Hinglish** and get answers grounded in your scripture library
(Ramayana, Vedas, Upanishads, Puranas, Gita, …), with page-level citations.

- **Backend:** FastAPI (async, SSE streaming), Gemini, Qdrant hybrid search
- **Frontend:** Streamlit chat UI

For the full design, history and runbook, see [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md).

## Architecture

```
            ┌───────────── Ingestion (once per book) ─────────────┐
 PDF/TXT ─► text layer ─► quality check ─► Gemini OCR (if garbled) ─► normalise ─► section-aware chunks
                                             (cached per page)                       │
                                                       Gemini embeddings + BM25 sparse ▼
                                                                                   Qdrant
            ┌───────────── Query (per request) ───────────────────┐                 ▲
 Question ─► query rewrite (follow-ups / cross-lingual, skipped    ─► dense+BM25 ───┘ RRF fusion
             for self-contained Hindi questions)                     multi-query
          ─► grounded prompt with numbered passages ─► Gemini stream ─► SSE ─► Streamlit
```

### Design decisions

| Concern | Choice |
|---|---|
| Broken Hindi PDFs | Many Hindi PDFs (including this one) embed fonts with broken Unicode maps, so extracted text is garbled (`चक` instead of `कि`). `ingestion/quality.py` detects this per page and per document, and `gemini-3.8-flash` OCRs those pages. OCR output is cached on disk, so re-indexing never re-runs OCR. |
| Chunking | Sentence-aware (`।`, `॥`, `?`, `.`), never crosses a chapter heading, can span page breaks, ~1100 chars with ~220 chars overlap. Each chunk keeps `page_start`/`page_end`/`section`. |
| Contextual embeddings | Each chunk is embedded as `"<book> \| <chapter>\n<text>"`, so a passage that never names its subject still matches. |
| Retrieval | Hybrid: `gemini-embedding-001` (768-d, cross-lingual) **+** BM25 sparse vectors (Qdrant applies IDF server-side). Multi-query (original + Hindi rewrites), fused with **RRF in a single Qdrant call**. |
| Indexing | Qdrant HNSW (`m=32`) + int8 scalar quantisation kept in RAM, with originals on disk: search time is O(log N), and memory stays low as the library grows to millions of chunks. Payload indexes on `doc_id` / `category` / `language` speed up filtered search. |
| Model overload | If `gemini-3.8-flash` returns 503/429 or hasn't started answering within 5 s, the answer automatically fails over to `GENERATION_FALLBACK_MODELS` (`gemini-3.6-flash`, then `gemini-3.5-flash`). |
| Latency | Fast path skips the rewrite for self-contained Hindi questions. The raw-question embedding runs in parallel with the rewrite. LRU query-embedding cache, TTL answer cache (cache hit ≈ 3 ms), streaming tokens. |
| Accuracy / safety | Answers come only from retrieved passages with `[n]` citations. The model admits when the sources lack an answer, doesn't invent shlokas, and treats passages as data (prompt-injection guard). It replies in the user's language and script. |
| Ops | Typed config (`.env`), structured/JSON logs with request IDs, `/health` + `/ready`, retries with jittered backoff for Gemini, rate limiting, optional API keys, admin ingest jobs, Docker + Compose. |

## Quick start (local, Windows/Linux/macOS)

```bash
# 1. Install
pip install -r backend/requirements.txt -r frontend/requirements.txt

# 2. Configure: copy .env.example to .env and set GEMINI_API_KEY (done already for this machine)

# 3. Ingest a book (stop the API first when using embedded Qdrant — see note below)
cd backend
python -m app.ingestion.cli "../data/raw/राम राम.pdf" --title "राम राम" --category ramayana --language hi

# 4. Run the API            -> http://127.0.0.1:8000/docs
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 5. Run the UI (new terminal) -> http://localhost:8501
cd frontend
streamlit run streamlit_app.py
```

> **Embedded Qdrant note:** with no `QDRANT_URL` set, the index lives in `data/qdrant/` and only **one
> process** can open it. Either stop the API before running the CLI, or ingest through the running API:
>
> ```bash
> curl -X POST http://127.0.0.1:8000/v1/admin/ingest -H "X-API-Key: $ADMIN_API_KEY" \
>   -F "file=@data/raw/kathopanishad.pdf" -F "title=कठोपनिषद्" -F "category=upanishad" -F "language=sa"
> curl http://127.0.0.1:8000/v1/admin/jobs/<job_id> -H "X-API-Key: $ADMIN_API_KEY"
> ```

## Adding more books (Vedas, Upanishads, Puranas…)

- Supported formats: PDF (text or scanned), Word `.docx`, and UTF-8 `.txt` / `.md` (use `## Heading` lines for chapters).
  For `.docx`, page numbers come from the page breaks Word saved, and chapter titles are detected from
  Heading styles or short ALL-CAPS lines ("CHAPTER III" + "PRANA" → `CHAPTER III — PRANA`).
- Ingestion is **idempotent**: `doc_id` is the file's SHA-256 hash, so re-running skips a book that's already indexed; `--force` re-indexes it.
- Use `category` (`veda`, `upanishad`, `purana`, `gita`, `ramayana`, …) and `language` (`hi`, `sa`, `en`) consistently. The UI can limit search to selected books.
- For scanned or garbled books, OCR runs automatically (`OCR_MODE=auto`). Use `always` or `never` to override.

## Production deployment

```bash
docker compose up -d --build   # qdrant + api (2 workers) + ui
```

- Set `QDRANT_URL` to use a Qdrant server or cluster (needed for multiple API workers/replicas).
  Moving from embedded mode means re-ingesting, which is cheap: OCR is cached, so only embeddings are recomputed.
- Set `ADMIN_API_KEY` to a long random value; set `PUBLIC_API_KEY` to require keys on chat and search.
- Caches and the rate limiter are per process. For many replicas, move them to Redis.
- Put a reverse proxy (nginx, Caddy) in front for TLS and gzip, with **buffering disabled** for `/v1/chat/stream`.
- `ENVIRONMENT=prod` disables `/docs`; `LOG_JSON=true` emits JSON logs for your log stack.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/ready` | liveness / readiness (chunk + book counts) |
| GET | `/v1/books` | library catalogue |
| POST | `/v1/chat/stream` | SSE: `meta` (sources, rewritten queries) → `token`* → `done` (timings) |
| POST | `/v1/chat` | same, non-streaming JSON |
| POST | `/v1/search` | retrieval only (debugging / evaluation) |
| POST | `/v1/admin/ingest` | upload & index a book (background job) |
| GET | `/v1/admin/jobs/{id}` | ingestion progress |
| DELETE | `/v1/admin/books/{doc_id}` | remove a book |
| POST | `/v1/admin/cache/clear` | clear answer cache |

Request body: `{"question": "...", "history": [{"role":"user|assistant","content":"..."}], "doc_ids": null, "top_k": 6}`

## Tests

```bash
cd backend && python -m pytest -q
```

## Project layout

```
backend/app/
  config.py          typed settings (.env)
  text.py            Indic normalisation, sentence split, BM25 tokeniser
  gemini.py          Gemini client: retries, batch embeddings, cache, streaming, OCR
  vectorstore.py     Qdrant hybrid store (dense + sparse, RRF)
  registry.py        book catalogue (data/books.json)
  ingestion/         quality detection, loader+OCR, chunker, pipeline, CLI
  rag/               prompts, engine (plan → retrieve → generate)
  api/               routes, schemas, security (API keys, rate limit)
frontend/streamlit_app.py
data/raw/            source books   data/cache/ocr/  OCR cache   data/processed/  clean text (audit)
```
