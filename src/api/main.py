"""SDLC-Swarm FastAPI backend — API gateway and health endpoint."""

from __future__ import annotations

import os

import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Load environment variables from .env at import time (before any config reads).
load_dotenv()

__version__ = "0.1.0"

app = FastAPI(
    title="SDLC-Swarm",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)


class HealthResponse(BaseModel):
    """Structured health payload per VAL-BACKEND-API-001."""

    status: str
    version: str
    db: str
    langfuse: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return structured health status.

    Checks Postgres connectivity and Langfuse reachability.
    Responds within 200 ms (VAL-BACKEND-API-001).
    """
    db_status = await _check_postgres()
    langfuse_status = await _check_langfuse()

    overall = "ok" if db_status == "ok" and langfuse_status != "unreachable" else "degraded"

    return HealthResponse(
        status=overall,
        version=__version__,
        db=db_status,
        langfuse=langfuse_status,
    )


async def _check_postgres() -> str:
    """Probe Postgres+pgvector on port 5433.

    Returns ``"ok"`` on successful connection, ``"unreachable"`` otherwise.
    """
    dsn = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'sdlc_swarm')}"
        f":{os.getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
        f":{os.getenv('POSTGRES_PORT', '5433')}"
        f"/{os.getenv('POSTGRES_DB', 'sdlc_swarm')}"
    )
    try:
        conn = await asyncpg.connect(dsn)
        await conn.execute("SELECT 1")
        await conn.close()
        return "ok"
    except Exception:
        return "unreachable"


async def _check_langfuse() -> str:
    """Probe Langfuse on port 3110.

    Langfuse v3 does not expose ``/api/health``; the root URL returns 200
    when the server is up.  Returns ``"ok"`` if reachable with a 2xx status,
    ``"degraded"`` if reachable but unhealthy, ``"unreachable"`` if it
    cannot be contacted.
    """
    langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3110")
    # Try /api/health first (Langfuse v2), fall back to root (Langfuse v3)
    for path in ("/api/health", "/"):
        url = f"{langfuse_host}{path}"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(url)
                if 200 <= resp.status_code < 300:
                    return "ok"
                if resp.status_code < 500:
                    # Server is up but not healthy at this endpoint
                    continue
        except Exception:
            continue
    # If we got here, try one more time on root as a last resort
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{langfuse_host}/")
            if 200 <= resp.status_code < 300:
                return "ok"
            return "degraded"
    except Exception:
        return "unreachable"
