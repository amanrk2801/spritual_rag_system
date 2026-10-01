"""Section-aware, sentence-boundary chunker with overlap and page provenance.

Chunks never split a sentence (unless a single sentence exceeds the budget), never
cross a section heading, and may span page breaks so narrative context is kept.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..text import split_sentences


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    page_start: int
    page_end: int
    section: str


@dataclass(slots=True)
class _Sent:
    text: str
    page: int


def _hard_wrap(sentence: str, size: int) -> list[str]:
    words, out, cur = sentence.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        out.append(cur)
    return out


def _sections(pages: Iterable[tuple[int, str]], size: int) -> Iterable[tuple[str, list[_Sent]]]:
    section, sents = "", []
    for page_no, text in pages:
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if para.startswith("#"):
                heading = para.lstrip("#").strip().split("\n")[0]
                if sents:
                    yield section, sents
                section, sents = heading, []
                rest = para.split("\n", 1)[1].strip() if "\n" in para else ""
                if not rest:
                    continue
                para = rest
            # Single newlines inside a paragraph are layout wraps, except in verses.
            joined = para.replace("\n", " ") if "॥" not in para else para
            for s in split_sentences(joined):
                for piece in _hard_wrap(s, size) if len(s) > size else [s]:
                    sents.append(_Sent(piece, page_no))
    if sents:
        yield section, sents


def chunk_pages(pages: Iterable[tuple[int, str]], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, sents in _sections(pages, size):
        start = 0
        while start < len(sents):
            end, length = start, 0
            while end < len(sents) and (length == 0 or length + len(sents[end].text) + 1 <= size):
                length += len(sents[end].text) + 1
                end += 1
            window = sents[start:end]
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=" ".join(s.text for s in window),
                    page_start=window[0].page,
                    page_end=window[-1].page,
                    section=section,
                )
            )
            if end >= len(sents):
                break
            # Step back over trailing sentences to create overlap (always advance >= 1).
            back, ov = end, 0
            while back - 1 > start and ov + len(sents[back - 1].text) <= overlap:
                back -= 1
                ov += len(sents[back].text) + 1
            start = back
    return chunks
