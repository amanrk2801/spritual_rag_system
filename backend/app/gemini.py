"""Thin async wrapper over google-genai with retries, batching and caching."""
from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator, Sequence
from typing import Any

from cachetools import LRUCache
from google import genai
from google.genai import errors, types
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import Settings

log = logging.getLogger(__name__)

_NO_AFC = types.AutomaticFunctionCallingConfig(disable=True)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in (408, 429, 500, 502, 503, 504)
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError))


def _retrying(attempts: int = 5, max_wait: float = 30) -> AsyncRetrying:
    return AsyncRetrying(
        retry=retry_if_exception(_is_transient),
        wait=wait_exponential_jitter(initial=min(1, max_wait), max=max_wait),
        stop=stop_after_attempt(attempts),
        reraise=True,
    )


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _thinking(level: str | None) -> types.ThinkingConfig | None:
    return types.ThinkingConfig(thinking_level=level) if level else None


class GeminiClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self._client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(),
            http_options=types.HttpOptions(timeout=int(settings.llm_timeout_s * 1000)),
        )
        self.aio = self._client.aio
        self._query_cache: LRUCache[str, list[float]] = LRUCache(settings.embedding_cache_size)

    # ------------------------------------------------------------------ embeddings
    async def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        async for attempt in _retrying():
            with attempt:
                resp = await self.aio.models.embed_content(
                    model=self.s.embedding_model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type, output_dimensionality=self.s.embedding_dim
                    ),
                )
        embeddings = resp.embeddings or []
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} != {len(texts)}")
        return [_l2_normalize(e.values or []) for e in embeddings]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        bs = self.s.embed_batch_size
        batches = [texts[i : i + bs] for i in range(0, len(texts), bs)]
        sem = asyncio.Semaphore(self.s.embed_concurrency)

        async def run(batch: list[str]) -> list[list[float]]:
            async with sem:
                return await self._embed_batch(batch, "RETRIEVAL_DOCUMENT")

        results = await asyncio.gather(*(run(b) for b in batches))
        return [v for batch in results for v in batch]

    async def embed_queries(self, queries: list[str]) -> list[list[float]]:
        missing = [q for q in dict.fromkeys(queries) if q not in self._query_cache]
        if missing:
            for q, v in zip(missing, await self._embed_batch(missing, "RETRIEVAL_QUERY")):
                self._query_cache[q] = v
        return [self._query_cache[q] for q in queries]

    # ------------------------------------------------------------------ generation
    def _gen_config(self, system: str | None, **kw: Any) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            automatic_function_calling=_NO_AFC,
            **kw,
        )

    async def generate_json(
        self, prompt: str, schema: dict, model: str | None = None, timeout: float | None = None
    ) -> Any:
        cfg = self._gen_config(
            None,
            temperature=0.0,
            response_mime_type="application/json",
            response_json_schema=schema,
            thinking_config=_thinking(self.s.rewrite_thinking_level),
        )
        resp = await asyncio.wait_for(
            self.aio.models.generate_content(
                model=model or self.s.rewrite_model, contents=prompt, config=cfg
            ),
            timeout=timeout or self.s.rewrite_timeout_s,
        )
        return resp.parsed if resp.parsed is not None else resp.text

    async def stream_answer(
        self, system: str, contents: list[types.Content]
    ) -> AsyncIterator[str]:
        cfg = self._gen_config(
            system,
            temperature=self.s.generation_temperature,
            max_output_tokens=self.s.max_output_tokens,
            thinking_config=_thinking(self.s.generation_thinking_level),
        )
        async def open_stream(model: str):
            async for attempt in _retrying(2, max_wait=2):
                with attempt:
                    stream = await self.aio.models.generate_content_stream(
                        model=model, contents=contents, config=cfg
                    )
                    return stream, await anext(stream, None)

        # Interactive path: fail over to the next model when one is overloaded (503/429)
        # or too slow to start answering. Only possible before the first token is sent.
        models = [self.s.generation_model, *self.s.generation_fallback_models]
        for i, model in enumerate(models):
            last = i == len(models) - 1
            try:
                stream, first = await asyncio.wait_for(
                    open_stream(model), None if last else self.s.first_token_timeout_s
                )
                break
            except Exception as exc:
                if last or not _is_transient(exc):
                    raise
                reason = "no first token in %.0fs" % self.s.first_token_timeout_s if isinstance(
                    exc, asyncio.TimeoutError
                ) else str(exc)[:120]
                log.warning("%s unavailable (%s); falling back to %s", model, reason, models[i + 1])
        if model != self.s.generation_model:
            log.info("answer served by fallback model %s", model)

        if first is not None and first.text:
            yield first.text
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    # ------------------------------------------------------------------ OCR
    async def ocr_page(self, png: bytes, prompt: str) -> str:
        cfg = self._gen_config(
            None, temperature=0.0, thinking_config=_thinking(self.s.ocr_thinking_level)
        )
        async for attempt in _retrying(6):
            with attempt:
                resp = await self.aio.models.generate_content(
                    model=self.s.ocr_model,
                    contents=[types.Part.from_bytes(data=png, mime_type="image/png"), prompt],
                    config=cfg,
                )
        return resp.text or ""
