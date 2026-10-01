"""Qdrant-backed hybrid (dense + BM25 sparse) store with server-side RRF fusion.

Local embedded mode (QDRANT_PATH) for development; set QDRANT_URL for a Qdrant
server/cluster in production (HNSW index, payload indexes, horizontal scaling).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models as qm

from .config import Settings

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"
_NS = uuid.UUID("5b0f6c1e-8b8e-4c9a-9a51-0a6f3d1f2e77")


def point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NS, f"{doc_id}:{chunk_index}"))


@dataclass(slots=True)
class Hit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, settings: Settings):
        self.s = settings
        self.collection = settings.collection_name
        if settings.qdrant_url:
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
                prefer_grpc=True,
            )
            self.remote = True
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = AsyncQdrantClient(path=str(settings.qdrant_path))
            self.remote = False

    async def ensure_collection(self) -> None:
        if await self.client.collection_exists(self.collection):
            return
        log.info("Creating collection %s", self.collection)
        await self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: qm.VectorParams(
                    size=self.s.embedding_dim,
                    distance=qm.Distance.COSINE,
                    on_disk=True,
                    hnsw_config=qm.HnswConfigDiff(m=32, ef_construct=256),
                )
            },
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            quantization_config=qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, always_ram=True)
            ),
        )
        if self.remote:  # payload indexes are a no-op in embedded mode
            for field in ("doc_id", "category", "language"):
                await self.client.create_payload_index(
                    self.collection, field, qm.PayloadSchemaType.KEYWORD
                )

    async def upsert(self, points: list[qm.PointStruct], batch: int = 128) -> None:
        for i in range(0, len(points), batch):
            await self.client.upsert(self.collection, points[i : i + batch], wait=True)

    async def delete_doc(self, doc_id: str) -> None:
        await self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(filter=_doc_filter([doc_id])),
            wait=True,
        )

    async def count(self) -> int:
        return (await self.client.count(self.collection, exact=False)).count

    async def hybrid_search(
        self,
        dense_queries: list[list[float]],
        sparse_queries: list[tuple[list[int], list[float]]],
        limit: int,
        prefetch_k: int,
        doc_ids: list[str] | None = None,
    ) -> list[Hit]:
        flt = _doc_filter(doc_ids) if doc_ids else None
        prefetch = [
            qm.Prefetch(query=v, using=DENSE, limit=prefetch_k, filter=flt) for v in dense_queries
        ] + [
            qm.Prefetch(
                query=qm.SparseVector(indices=idx, values=val),
                using=SPARSE,
                limit=prefetch_k,
                filter=flt,
            )
            for idx, val in sparse_queries
            if idx
        ]
        res = await self.client.query_points(
            self.collection,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
            query_filter=flt,
        )
        return [Hit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in res.points]

    async def close(self) -> None:
        await self.client.close()


def _doc_filter(doc_ids: list[str]) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids))])
