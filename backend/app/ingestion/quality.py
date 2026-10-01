"""Detects pages whose embedded text layer is unusable (broken font ToUnicode maps,
scanned images, legacy non-Unicode Hindi fonts like Kruti Dev) so they get OCR'd."""
from __future__ import annotations

import re
from dataclasses import dataclass

_DEV_WORD = re.compile(r"[ऀ-ॿ]+")
# A word may never begin with a dependent vowel sign / virama / modifier.
_ORPHAN_SIGN = re.compile(r"^[ँ-ःा-्ॢॣ]")
# Virama directly followed by a dependent vowel sign is invalid Unicode Hindi.
_VIRAMA_MATRA = re.compile(r"्[ा-ौ]")
# Ultra-frequent Hindi words containing the i-matra; reordered-matra garbling destroys them.
_I_MATRA_WORDS = frozenset({"कि", "लिए", "किया", "दिया", "किसी", "लिया", "जिस", "दिन", "फिर", "लिये"})
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


@dataclass(slots=True)
class PageQuality:
    chars: int
    orphan_ratio: float
    virama_matra: int
    i_matra_ratio: float
    needs_ocr: bool


def assess(text: str) -> PageQuality:
    stripped = text.strip()
    dev_words = _DEV_WORD.findall(stripped)
    n = len(dev_words)
    latin = len(_LATIN_WORD.findall(stripped))

    if len(stripped) < 40:  # empty / image-only page
        return PageQuality(len(stripped), 0.0, 0, 0.0, needs_ocr=True)
    if n < 0.2 * (n + latin):  # predominantly Latin-script text: trust the text layer
        return PageQuality(len(stripped), 0.0, 0, 0.0, needs_ocr=False)

    orphan = sum(1 for w in dev_words if _ORPHAN_SIGN.match(w)) / n
    vm = len(_VIRAMA_MATRA.findall(stripped))
    im = sum(1 for w in dev_words if w in _I_MATRA_WORDS) / n
    needs = orphan > 0.01 or vm > 0 or (n > 80 and im < 0.004)
    return PageQuality(len(stripped), round(orphan, 4), vm, round(im, 4), needs)
