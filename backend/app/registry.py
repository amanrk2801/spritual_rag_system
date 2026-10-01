"""Book catalogue (small JSON manifest, atomically written)."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from pydantic import BaseModel


class Book(BaseModel):
    doc_id: str
    title: str
    category: str = "general"
    language: str = "hi"
    source_file: str
    pages: int
    chunks: int
    ocr_pages: int = 0
    ingested_at: str


class BookRegistry:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Book]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {k: Book(**v) for k, v in raw.items()}

    def _write(self, books: dict[str, Book]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({k: v.model_dump() for k, v in books.items()}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def all(self) -> list[Book]:
        with self._lock:
            return sorted(self._read().values(), key=lambda b: b.title)

    def get(self, doc_id: str) -> Book | None:
        with self._lock:
            return self._read().get(doc_id)

    def upsert(self, book: Book) -> None:
        with self._lock:
            books = self._read()
            books[book.doc_id] = book
            self._write(books)

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            books = self._read()
            if books.pop(doc_id, None) is None:
                return False
            self._write(books)
            return True
