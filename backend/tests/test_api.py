"""API contract tests with the engine stubbed out (no network, no vector store)."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import admin, public, router
from app.api.security import RateLimiter
from app.registry import BookRegistry


class FakeEngine:
    async def stream(self, question, history=None, doc_ids=None, top_k=None):
        yield {"type": "meta", "sources": [], "search_queries": [question], "cached": False}
        yield {"type": "token", "text": "नमस्ते"}
        yield {"type": "done", "latency_ms": 1, "cached": False}

    async def answer(self, **kw):
        out = {"answer": ""}
        async for ev in self.stream(**kw):
            if ev["type"] == "token":
                out["answer"] += ev["text"]
        return out


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.include_router(public)
    app.include_router(admin)
    app.state.engine = FakeEngine()
    app.state.registry = BookRegistry(tmp_path / "books.json")
    app.state.rate_limiter = RateLimiter(3)
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_stream_sse(client):
    with client.stream("POST", "/v1/chat/stream", json={"question": "राम कौन हैं?"}) as r:
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    assert [e["type"] for e in events] == ["meta", "token", "done"]


def test_validation_rejects_blank_question(client):
    assert client.post("/v1/chat", json={"question": "   "}).status_code == 422


def test_rate_limit(client):
    codes = [client.post("/v1/chat", json={"question": "q"}).status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429


def test_admin_requires_key(client):
    r = client.post("/v1/admin/cache/clear")
    assert r.status_code in (401, 403)
