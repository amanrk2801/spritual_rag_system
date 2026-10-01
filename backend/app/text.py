"""Language-aware text utilities: normalisation, sentence splitting, lexical tokenisation.

Designed for Devanagari (Hindi / Sanskrit) and Latin scripts so the same pipeline
works for Ramayana prose, Vedic mantras, Upanishads, Puranas, Gita translations, etc.
"""
from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter

_WS = re.compile(r"[ \t ​]+")
_MULTI_NL = re.compile(r"\n{3,}")
# Pipe used as danda in many Hindi typesettings -> real danda.
_PIPE_DOUBLE_DANDA = re.compile(r"(?<=[\u0900-\u097F\s])\|\|(?=\s|$)")
_PIPE_DANDA = re.compile(r"(?<=[\u0900-\u097F\s])\|(?=\s|$)")
_DOUBLE_DANDA = re.compile(r"।\s*।")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Sentence boundary: danda / double danda / ?! / latin full stop followed by space.
_SENT_SPLIT = re.compile(r"(?<=[।॥?!])\s+|(?<=[.])\s+(?=[A-Z\u0900-\u097F\"“'])")

# Lexical tokens: Latin alnum or Devanagari letters + combining marks (danda excluded).
_TOKEN = re.compile(r"[a-z0-9]+|[\u0900-\u0963\u0966-\u097F]+")

_STOPWORDS = frozenset(
    """
    है हैं था थी थे हो होता होती होते का के की को में से पर ने और भी तो यह वह ये वे इस उस इन उन एक
    कि कर करके किया दिया जो जा जब तब तक लिए लिये ही नहीं न कुछ क्या क्यों कैसे अपने अपना अपनी मैं
    हम तुम आप उसे उसने उनके उनकी उनका सब बहुत अब फिर साथ बाद रहा रही रहे गया गई गए
    च वा हि तु एव अपि इति स सा तत् ते
    a an the of to in on for and or is are was were be been it this that with as by at from
    what who whom which why how do does did i you he she we they me my your his her our their
    """.split()
)


def normalize(text: str) -> str:
    """Canonical Unicode form + whitespace / punctuation cleanup (idempotent)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _PIPE_DOUBLE_DANDA.sub("॥", text)
    text = _PIPE_DANDA.sub("।", text)
    text = _DOUBLE_DANDA.sub("॥", text)
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MULTI_NL.sub("\n\n", text).strip()


def is_devanagari(text: str, threshold: float = 0.3) -> bool:
    letters = [c for c in text if c.isalpha() or unicodedata.category(c).startswith("M")]
    if not letters:
        return False
    return sum(1 for c in letters if _DEVANAGARI.match(c)) / len(letters) >= threshold


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(paragraph) if s.strip()]


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    # Strip nukta / chandrabindu variance so spelling variants match lexically.
    text = text.replace("़", "").replace("ँ", "ं")
    return [t for t in _TOKEN.findall(text) if t not in _STOPWORDS and len(t) > 1]


def _token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


def bm25_document_vector(
    text: str, avg_len: float, k1: float = 1.2, b: float = 0.75
) -> tuple[list[int], list[float]]:
    """BM25 term-frequency component; IDF is applied server-side by Qdrant (Modifier.IDF)."""
    tokens = tokenize(text)
    if not tokens:
        return [], []
    counts = Counter(_token_id(t) for t in tokens)
    norm = k1 * (1 - b + b * len(tokens) / avg_len)
    indices = list(counts.keys())
    values = [tf * (k1 + 1) / (tf + norm) for tf in counts.values()]
    return indices, values


def bm25_query_vector(text: str) -> tuple[list[int], list[float]]:
    ids = sorted({_token_id(t) for t in tokenize(text)})
    return ids, [1.0] * len(ids)
