from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from ..config import Settings, get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _matches(provided: str | None, expected: str) -> bool:
    return provided is not None and hmac.compare_digest(provided.encode(), expected.encode())


def require_public_key(
    key: str | None = Depends(_api_key_header), s: Settings = Depends(get_settings)
) -> None:
    if s.public_api_key and not _matches(key, s.public_api_key.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")


def require_admin_key(
    key: str | None = Depends(_api_key_header), s: Settings = Depends(get_settings)
) -> None:
    if not s.admin_api_key:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin API disabled (ADMIN_API_KEY not set)")
    if not _matches(key, s.admin_api_key.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid admin API key")


class RateLimiter:
    """Sliding-window limiter, per client. In-process; use Redis for multi-instance deployments."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def __call__(self, request: Request) -> None:
        if self.per_minute <= 0:
            return
        fwd = request.headers.get("x-forwarded-for")
        client = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")
        now = time.monotonic()
        q = self._hits[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Rate limit exceeded, please slow down",
                headers={"Retry-After": str(int(60 - (now - q[0])) + 1)},
            )
        q.append(now)
        if len(self._hits) > 50_000:  # bound memory under IP churn
            for k in [k for k, v in self._hits.items() if not v or now - v[-1] > 60]:
                del self._hits[k]
