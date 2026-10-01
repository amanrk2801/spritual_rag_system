"""End-to-end, idempotent ingestion: load -> OCR(if needed) -> chunk -> embed -> index."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import models as qm

from ..config import Settings
from ..gemini import GeminiClient
from ..registry import Book, BookRegistry
from ..text import bm25_document_vector
from ..vectorstore import DENSE, SPARSE, VectorStore, point_id
from .chunker import chunk_pages
from .loader import ProgressFn, file_sha256, load_document

log = logging.getLogger(__name__)


async def ingest_file(
    path: Path,
    *,
    title: str,
    category: str,
    language: str,
    gemini: GeminiClient,
    store: VectorStore,
    registry: BookRegistry,
    settings: Settings,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> Book:
    t0 = time.perf_counter()
    doc_id = file_sha256(path)[:16]
    existing = registry.get(doc_id)
    if existing and not force:
        log.info("Already ingested %s (%s); skipping", title, doc_id)
        return existing

    pages = await load_document(path, doc_id, gemini, settings, progress)
    if not pages:
        raise ValueError("No extractable text found")

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    with (settings.processed_dir / f"{doc_id}.jsonl").open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps({"page": p.number, "ocr": p.ocr, "text": p.text}, ensure_ascii=False) + "\n")

    chunks = chunk_pages(
        ((p.number, p.text) for p in pages), settings.chunk_size_chars, settings.chunk_overlap_chars
    )
    if progress:
        progress("embed", 0, len(chunks))

    # Contextual embedding: prefix book + section so isolated chunks stay disambiguated.
    embed_inputs = [
        f"{title}{' | ' + c.section if c.section else ''}\n{c.text}" for c in chunks
    ]
    vectors = await gemini.embed_documents(embed_inputs)
    if progress:
        progress("embed", len(chunks), len(chunks))

    points = []
    for c, vec in zip(chunks, vectors):
        idx, val = bm25_document_vector(f"{c.section} {c.text}", settings.bm25_avg_doc_len)
        points.append(
            qm.PointStruct(
                id=point_id(doc_id, c.index),
                vector={DENSE: vec, SPARSE: qm.SparseVector(indices=idx, values=val)},
                payload={
                    "doc_id": doc_id,
                    "title": title,
                    "category": category,
                    "language": language,
                    "chunk_index": c.index,
                    "section": c.section,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "text": c.text,
                },
            )
        )

    await store.ensure_collection()
    await store.delete_doc(doc_id)  # idempotent re-ingest
    await store.upsert(points)

    book = Book(
        doc_id=doc_id,
        title=title,
        category=category,
        language=language,
        source_file=path.name,
        pages=len(pages),
        chunks=len(chunks),
        ocr_pages=sum(p.ocr for p in pages),
        ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    registry.upsert(book)
    log.info(
        "Ingested %s: %d pages, %d chunks in %.1fs", title, len(pages), len(chunks), time.perf_counter() - t0
    )
    return book
