"""CLI ingestion.

    python -m app.ingestion.cli ../data/raw/book.pdf --title "Book" --category ramayana --language hi

Note: embedded (local-path) Qdrant allows one process at a time. Stop the API first,
or run Qdrant as a server (QDRANT_URL) / use the POST /v1/admin/ingest endpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from ..config import get_settings
from ..gemini import GeminiClient
from ..logging_setup import setup_logging
from ..registry import BookRegistry
from ..vectorstore import VectorStore
from .pipeline import ingest_file


def _progress(stage: str, done: int, total: int) -> None:
    sys.stdout.write(f"\r  {stage}: {done}/{total}   ")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


async def _main(args: argparse.Namespace) -> None:
    s = get_settings()
    setup_logging(s.log_level, s.log_json)
    gemini, store = GeminiClient(s), VectorStore(s)
    try:
        book = await ingest_file(
            Path(args.path),
            title=args.title or Path(args.path).stem,
            category=args.category,
            language=args.language,
            gemini=gemini,
            store=store,
            registry=BookRegistry(s.registry_path),
            settings=s,
            force=args.force,
            progress=_progress,
        )
        print(book.model_dump_json(indent=2))
    finally:
        await store.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest a scripture into the RAG index")
    p.add_argument("path")
    p.add_argument("--title")
    p.add_argument("--category", default="general", help="e.g. ramayana, veda, upanishad, purana, gita")
    p.add_argument("--language", default="hi")
    p.add_argument("--force", action="store_true", help="re-index even if already ingested")
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
