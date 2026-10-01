from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=40)
    doc_ids: list[str] | None = Field(default=None, description="Restrict to these books")
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    doc_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class Source(BaseModel):
    n: int
    doc_id: str | None
    title: str | None
    section: str = ""
    page_start: int | None
    page_end: int | None
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    standalone_question: str | None = None
    search_queries: list[str] = []
    answer_language: str | None = None
    latency_ms: int | None = None
    timings: dict[str, int] | None = None
    cached: bool = False


class SearchResponse(BaseModel):
    standalone_question: str
    search_queries: list[str]
    sources: list[Source]
    timings: dict[str, int]


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str = ""
    done: int = 0
    total: int = 0
    error: str | None = None
    doc_id: str | None = None
