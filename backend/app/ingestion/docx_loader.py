"""Word (.docx) loading with page numbers and chapter detection.

Page numbers come from the page breaks Word records when it saves a file
(`w:lastRenderedPageBreak`) plus explicit page breaks. Many books typeset chapter
titles as plain short ALL-CAPS paragraphs rather than Heading styles, so headings are
detected from formatting as well; consecutive heading lines ("CHAPTER III" / "PRANA")
are merged into one section title.
"""
from __future__ import annotations

from pathlib import Path

import docx
from docx.oxml.ns import qn

from ..text import normalize

_LAST_RENDERED_BREAK = qn("w:lastRenderedPageBreak")
_BR = qn("w:br")
_BR_TYPE = qn("w:type")
_MAX_HEADING_CHARS = 90


def _page_breaks(paragraph) -> int:
    el = paragraph._p
    n = sum(1 for _ in el.iter(_LAST_RENDERED_BREAK))
    return n + sum(1 for br in el.iter(_BR) if br.get(_BR_TYPE) == "page")


def _is_heading(text: str, style: str) -> bool:
    if style.startswith(("heading", "title")):
        return True
    if len(text) > _MAX_HEADING_CHARS or text.endswith((".", ",", ";", ":")):
        return False
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 2:
        return False
    if sum(c.isupper() for c in letters) / len(letters) >= 0.9:
        return True  # ALL-CAPS titles may be questions, e.g. "WHAT IS DUTY?"
    return style == "center" and len(text) <= 60 and not text.endswith(("?", "!"))


def load_docx(path: Path) -> list[tuple[int, str]]:
    """Returns (page_number, text) pairs; chapter titles are emitted as '## ' paragraphs."""
    document = docx.Document(str(path))
    pages: dict[int, list[str]] = {}
    page = 1
    pending_heading: list[str] = []
    pending_page = 1

    for p in document.paragraphs:
        breaks = _page_breaks(p)
        text = p.text.strip()
        if text and breaks and p._p.find(f".//{_LAST_RENDERED_BREAK}") is not None:
            # A rendered break inside a paragraph usually precedes its first line on the new page.
            page += breaks
            breaks = 0
        if text:
            if _is_heading(text, p.style.name.lower()):
                if not pending_heading:
                    pending_page = page
                pending_heading.append(text)
            else:
                if pending_heading:
                    pages.setdefault(pending_page, []).append("## " + " — ".join(pending_heading))
                    pending_heading = []
                pages.setdefault(page, []).append(text)
        page += breaks

    if pending_heading:
        pages.setdefault(pending_page, []).append("## " + " — ".join(pending_heading))
    return [(n, normalize("\n\n".join(paras))) for n, paras in sorted(pages.items())]
