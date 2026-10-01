"""RAG orchestration: query understanding -> hybrid retrieval -> grounded streaming answer."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from cachetools import TTLCache
from google.genai import types

from ..config import Settings
from ..gemini import GeminiClient
from ..text import bm25_query_vector, is_devanagari
from ..vectorstore import VectorStore
from .prompts import (
    NO_CONTEXT_REPLY,
    REWRITE_PROMPT,
    REWRITE_SCHEMA,
    SYSTEM_PROMPT,
    build_user_turn,
    format_context,
)

log = logging.getLogger(__name__)

_MAX_HISTORY_CHARS = 1500


@dataclass(slots=True)
class QueryPlan:
    standalone: str
    queries: list[str]
    answer_language: str
    rewritten: bool


@dataclass(slots=True)
class Retrieval:
    plan: QueryPlan
    sources: list[dict[str, Any]]
    timings: dict[str, float] = field(default_factory=dict)


class RAGEngine:
    def __init__(self, settings: Settings, gemini: GeminiClient, store: VectorStore):
        self.s = settings
        self.gemini = gemini
        self.store = store
        self._answers: TTLCache[str, dict[str, Any]] = TTLCache(
            settings.answer_cache_size, settings.answer_cache_ttl_s
        )

    # ------------------------------------------------------------ query understanding
    async def plan(self, question: str, history: list[dict[str, str]]) -> QueryPlan:
        question = question.strip()
        devanagari = is_devanagari(question)
        if not history and devanagari:  # fast path: already in the corpus language, self-contained
            return QueryPlan(question, [question], "Hindi", rewritten=False)

        hist = "\n".join(
            f"{m['role']}: {m['content'][:_MAX_HISTORY_CHARS]}" for m in history[-4:]
        ) or "(none)"
        try:
            out = await self.gemini.generate_json(
                REWRITE_PROMPT.format(history=hist, question=question), REWRITE_SCHEMA
            )
            if isinstance(out, str):
                out = json.loads(out)
            standalone = (out.get("standalone_question") or question).strip()
            queries = [standalone] + [q.strip() for q in out.get("search_queries", []) if q.strip()]
            lang = out.get("answer_language") or ("Hindi" if devanagari else "English")
            return QueryPlan(standalone, list(dict.fromkeys(queries))[:4], lang, rewritten=True)
        except Exception as exc:  # degrade gracefully: retrieval still works cross-lingually
            log.warning("Query rewrite failed (%s); using raw question", exc)
            return QueryPlan(question, [question], "Hindi" if devanagari else "English", False)

    # ------------------------------------------------------------ retrieval
    async def retrieve(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> Retrieval:
        timings: dict[str, float] = {}
        t = time.perf_counter()
        # Embed the raw question while the rewrite runs; it is usually one of the final queries.
        prefetch = asyncio.create_task(self.gemini.embed_queries([question.strip()]))
        plan = await self.plan(question, history or [])
        timings["plan_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        try:
            await prefetch
        except Exception as exc:
            log.warning("Query prefetch embedding failed: %s", exc)
        dense = await self.gemini.embed_queries(plan.queries)  # cache hit for the prefetched one
        timings["embed_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        hits = await self.store.hybrid_search(
            dense_queries=dense,
            sparse_queries=[bm25_query_vector(q) for q in plan.queries],
            limit=top_k or self.s.top_k,
            prefetch_k=self.s.prefetch_k,
            doc_ids=doc_ids,
        )
        timings["search_ms"] = (time.perf_counter() - t) * 1000

        sources = [
            {
                "n": i,
                "doc_id": h.payload.get("doc_id"),
                "title": h.payload.get("title"),
                "section": h.payload.get("section", ""),
                "page_start": h.payload.get("page_start"),
                "page_end": h.payload.get("page_end"),
                "score": round(h.score, 4),
                "text": h.payload.get("text", ""),
            }
            for i, h in enumerate(hits, 1)
        ]
        return Retrieval(plan, sources, timings)

    # ------------------------------------------------------------ generation
    def _cache_key(self, question: str, doc_ids: list[str] | None, top_k: int | None) -> str:
        raw = json.dumps(
            [" ".join(question.lower().split()), sorted(doc_ids or []), top_k or self.s.top_k],
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _contents(
        self, history: list[dict[str, str]], question: str, plan: QueryPlan, sources: list[dict]
    ) -> list[types.Content]:
        turns = history[-self.s.max_history_turns * 2 :]
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"][:_MAX_HISTORY_CHARS])],
            )
            for m in turns
        ]
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=build_user_turn(
                            plan.standalone, format_context(sources), question, plan.answer_language
                        )
                    )
                ],
            )
        )
        return contents

    async def stream(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yields events: meta -> token* -> done."""
        t0 = time.perf_counter()
        history = history or []
        cache_key = None if history else self._cache_key(question, doc_ids, top_k)

        if cache_key and (cached := self._answers.get(cache_key)):
            yield {"type": "meta", **cached["meta"], "cached": True}
            yield {"type": "token", "text": cached["answer"]}
            yield {"type": "done", "latency_ms": round((time.perf_counter() - t0) * 1000), "cached": True}
            return

        r = await self.retrieve(question, history, doc_ids, top_k)
        meta = {
            "standalone_question": r.plan.standalone,
            "search_queries": r.plan.queries,
            "answer_language": r.plan.answer_language,
            "sources": r.sources,
        }
        yield {"type": "meta", **meta, "cached": False}

        if not r.sources:
            lang = "Hindi" if r.plan.answer_language.lower().startswith("hindi") else "English"
            yield {"type": "token", "text": NO_CONTEXT_REPLY[lang]}
            yield {"type": "done", "latency_ms": round((time.perf_counter() - t0) * 1000), "cached": False}
            return

        system = SYSTEM_PROMPT.format(answer_language=r.plan.answer_language)
        contents = self._contents(history, question, r.plan, r.sources)
        t_gen = time.perf_counter()
        ttft = None
        parts: list[str] = []
        async for text in self.gemini.stream_answer(system, contents):
            if ttft is None:
                ttft = (time.perf_counter() - t_gen) * 1000
            parts.append(text)
            yield {"type": "token", "text": text}

        answer = "".join(parts)
        timings = {**r.timings, "ttft_ms": ttft or 0.0, "generate_ms": (time.perf_counter() - t_gen) * 1000}
        total = (time.perf_counter() - t0) * 1000
        log.info(
            "answered rewritten=%s hits=%d %s total=%.0fms",
            r.plan.rewritten,
            len(r.sources),
            " ".join(f"{k}={v:.0f}" for k, v in timings.items()),
            total,
        )
        if cache_key and answer.strip():
            self._answers[cache_key] = {"meta": meta, "answer": answer}
        yield {
            "type": "done",
            "latency_ms": round(total),
            "timings": {k: round(v) for k, v in timings.items()},
            "cached": False,
        }

    async def answer(self, **kw: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"answer": ""}
        async for ev in self.stream(**kw):
            if ev["type"] == "token":
                result["answer"] += ev["text"]
            else:
                result.update({k: v for k, v in ev.items() if k != "type"})
        return result

    def clear_cache(self) -> None:
        self._answers.clear()


async def warmup(engine: RAGEngine) -> None:
    """Prime connections so the first user request is not slow."""
    try:
        await asyncio.wait_for(engine.gemini.embed_queries(["राम"]), timeout=10)
    except Exception as exc:
        log.warning("Warmup failed: %s", exc)
