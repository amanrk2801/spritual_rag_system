from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from ..ingestion.pipeline import ingest_file
from ..registry import Book
from .schemas import ChatRequest, ChatResponse, JobStatus, SearchRequest, SearchResponse
from .security import require_admin_key, require_public_key

log = logging.getLogger(__name__)

router = APIRouter()
public = APIRouter(prefix="/v1", dependencies=[Depends(require_public_key)])
admin = APIRouter(prefix="/v1/admin", dependencies=[Depends(require_admin_key)], tags=["admin"])

_SAFE_NAME = re.compile(r"[^\w.\-ऀ-ॿ ]+")


def _state(request: Request):
    return request.app.state


def _rate_limit(request: Request) -> None:
    request.app.state.rate_limiter(request)


# ---------------------------------------------------------------- health
@router.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready", tags=["ops"])
async def ready(request: Request) -> dict:
    try:
        count = await _state(request).store.count()
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"vector store unavailable: {exc}")
    return {"status": "ready", "chunks": count, "books": len(_state(request).registry.all())}


# ---------------------------------------------------------------- public API
@public.get("/books", response_model=list[Book], tags=["library"])
async def list_books(request: Request) -> list[Book]:
    return _state(request).registry.all()


@public.post("/search", response_model=SearchResponse, tags=["rag"], dependencies=[Depends(_rate_limit)])
async def search(req: SearchRequest, request: Request) -> SearchResponse:
    r = await _state(request).engine.retrieve(req.query, None, req.doc_ids, req.top_k)
    return SearchResponse(
        standalone_question=r.plan.standalone,
        search_queries=r.plan.queries,
        sources=r.sources,
        timings={k: round(v) for k, v in r.timings.items()},
    )


@public.post("/chat", response_model=ChatResponse, tags=["rag"], dependencies=[Depends(_rate_limit)])
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    result = await _state(request).engine.answer(
        question=req.question,
        history=[m.model_dump() for m in req.history],
        doc_ids=req.doc_ids,
        top_k=req.top_k,
    )
    return ChatResponse(**result)


@public.post("/chat/stream", tags=["rag"], dependencies=[Depends(_rate_limit)])
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events: each `data:` line is JSON with type meta | token | done | error."""
    engine = _state(request).engine

    async def events():
        try:
            async for ev in engine.stream(
                question=req.question,
                history=[m.model_dump() for m in req.history],
                doc_ids=req.doc_ids,
                top_k=req.top_k,
            ):
                if await request.is_disconnected():
                    log.info("client disconnected; aborting generation")
                    return
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception:
            log.exception("stream failed")
            err = {"type": "error", "message": "The answer could not be generated. Please try again."}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------- admin API
@admin.post("/ingest", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(..., min_length=1, max_length=200),
    category: str = Form("general", max_length=50),
    language: str = Form("hi", max_length=20),
    force: bool = Form(False),
) -> JobStatus:
    st = _state(request)
    s = st.settings
    name = _SAFE_NAME.sub("_", Path(file.filename or "upload").name)
    if Path(name).suffix.lower() not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF, TXT and MD are supported")

    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = s.uploads_dir / name
    size, limit = 0, s.max_upload_mb * 1024 * 1024
    with dest.open("wb") as f:
        while block := await file.read(1 << 20):
            size += len(block)
            if size > limit:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")
            f.write(block)

    job = JobStatus(job_id=uuid.uuid4().hex, status="queued")
    st.jobs[job.job_id] = job

    def progress(stage: str, done: int, total: int) -> None:
        job.stage, job.done, job.total = stage, done, total

    async def run() -> None:
        async with st.ingest_lock:  # one ingestion at a time protects API quota
            job.status = "running"
            try:
                book = await ingest_file(
                    dest,
                    title=title,
                    category=category,
                    language=language,
                    gemini=st.gemini,
                    store=st.store,
                    registry=st.registry,
                    settings=s,
                    force=force,
                    progress=progress,
                )
                job.status, job.doc_id = "completed", book.doc_id
                st.engine.clear_cache()
            except Exception as exc:
                log.exception("ingestion failed")
                job.status, job.error = "failed", str(exc)

    st.tasks.add(task := asyncio.create_task(run()))
    task.add_done_callback(st.tasks.discard)
    return job


@admin.get("/jobs/{job_id}", response_model=JobStatus)
async def job_status(job_id: str, request: Request) -> JobStatus:
    job = _state(request).jobs.get(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown job")
    return job


@admin.delete("/books/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_book(doc_id: str, request: Request) -> None:
    st = _state(request)
    if not st.registry.get(doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown book")
    await st.store.delete_doc(doc_id)
    st.registry.remove(doc_id)
    st.engine.clear_cache()


@admin.post("/cache/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_cache(request: Request) -> None:
    _state(request).engine.clear_cache()
