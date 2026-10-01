"""Document loading: text-layer extraction with automatic, cached Gemini OCR fallback."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..config import Settings
from ..gemini import GeminiClient
from ..text import normalize
from .quality import assess

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]

OCR_PROMPT = """You are a precise OCR engine for Indian spiritual and religious texts.
Transcribe ALL text on this page image exactly as written, in correct Unicode
(Devanagari for Hindi/Sanskrit/Marathi, proper script for other languages).
Rules:
- Preserve the original wording and spelling. Do NOT translate, summarise, explain or fix grammar.
- Keep paragraph breaks as blank lines. Keep verse / shloka line breaks and verse numbers.
- Prefix chapter or section titles with "## ".
- Omit running headers, footers and standalone page numbers.
- Output only the transcribed text. If the page has no text, output nothing."""

# If more than this share of pages is garbled, the font mapping is broken document-wide.
_DOC_LEVEL_OCR_THRESHOLD = 0.3


@dataclass(slots=True)
class Page:
    number: int  # 1-based
    text: str
    ocr: bool


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


async def load_document(
    path: Path,
    doc_id: str,
    gemini: GeminiClient,
    settings: Settings,
    progress: ProgressFn | None = None,
) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        raw = path.read_text(encoding="utf-8")
        parts = raw.split("\f") if "\f" in raw else [raw]
        return [Page(i + 1, normalize(p), False) for i, p in enumerate(parts) if p.strip()]
    if suffix != ".pdf":
        raise ValueError(f"Unsupported file type: {suffix}")
    return await _load_pdf(path, doc_id, gemini, settings, progress)


async def _load_pdf(
    path: Path, doc_id: str, gemini: GeminiClient, s: Settings, progress: ProgressFn | None
) -> list[Page]:
    doc = await asyncio.to_thread(pymupdf.open, path)
    try:
        texts = await asyncio.to_thread(lambda: [p.get_text("text", sort=True) for p in doc])
        flags = [assess(t).needs_ocr for t in texts]

        if s.ocr_mode == "always":
            ocr_idx = list(range(len(texts)))
        elif s.ocr_mode == "never":
            ocr_idx = []
        elif sum(flags) / max(len(flags), 1) > _DOC_LEVEL_OCR_THRESHOLD:
            ocr_idx = list(range(len(texts)))
        else:
            ocr_idx = [i for i, f in enumerate(flags) if f]
        log.info("%s: %d pages, %d need OCR (mode=%s)", path.name, len(texts), len(ocr_idx), s.ocr_mode)

        cache_dir = s.ocr_cache_dir / doc_id / s.ocr_model
        cache_dir.mkdir(parents=True, exist_ok=True)
        render_lock = asyncio.Lock()  # PyMuPDF documents are not thread-safe
        sem = asyncio.Semaphore(s.ocr_concurrency)
        done = 0

        async def ocr(i: int) -> None:
            nonlocal done
            cache = cache_dir / f"p{i + 1:04d}.txt"
            if cache.exists():
                texts[i] = cache.read_text(encoding="utf-8")
            else:
                async with sem:
                    async with render_lock:
                        png = await asyncio.to_thread(
                            lambda: doc[i].get_pixmap(dpi=s.ocr_dpi).tobytes("png")
                        )
                    texts[i] = await gemini.ocr_page(png, OCR_PROMPT)
                cache.write_text(texts[i], encoding="utf-8")
            done += 1
            if progress:
                progress("ocr", done, len(ocr_idx))

        await asyncio.gather(*(ocr(i) for i in ocr_idx))
    finally:
        doc.close()

    ocr_set = set(ocr_idx)
    return [
        Page(i + 1, normalize(t), i in ocr_set) for i, t in enumerate(texts) if t.strip()
    ]
