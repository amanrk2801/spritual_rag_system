"""FastAPI entrypoint.  Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import admin, public, router
from .api.security import RateLimiter
from .config import get_settings
from .gemini import GeminiClient
from .logging_setup import request_id_var, setup_logging
from .rag.engine import RAGEngine, warmup
from .registry import BookRegistry
from .vectorstore import VectorStore

settings = get_settings()
setup_logging(settings.log_level, settings.log_json)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    st = app.state
    st.settings = settings
    st.gemini = GeminiClient(settings)
    st.store = VectorStore(settings)
    await st.store.ensure_collection()
    st.registry = BookRegistry(settings.registry_path)
    st.engine = RAGEngine(settings, st.gemini, st.store)
    st.rate_limiter = RateLimiter(settings.rate_limit_per_minute)
    st.jobs, st.tasks, st.ingest_lock = {}, set(), asyncio.Lock()
    await warmup(st.engine)
    log.info("Ready: %d chunks indexed, %d books", await st.store.count(), len(st.registry.all()))
    yield
    for t in st.tasks:
        t.cancel()
    await st.store.close()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "prod" else None,
)
# No GZip middleware: it buffers SSE token streams. Let the reverse proxy compress JSON.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id_var.set(rid)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    log.info(
        "%s %s -> %s %.0fms [%s]",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - t0) * 1000,
        rid,
    )
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(router)
app.include_router(public)
app.include_router(admin)
