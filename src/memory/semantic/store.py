"""pgvector Semantic Store — async Postgres-backed RAG operations.

Manages the ``repo_chunks`` table for semantic retrieval over
indexed repositories.  Provides:

- ``index_repo``: Walk a cloned repo directory, filter files by
  extension, chunk via token-aware splitter, embed via OpenAI
  ``text-embedding-3-small``, and write rows.  Idempotent (deletes
  existing rows for repo_url first).

- ``retrieve``: Cosine similarity top-k=8 scoped by repo_url.

- ``is_indexed``: Check if a repo_url already has chunks.

Architecture reference: §2.7 pgvector Semantic Store.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import asyncpg
from openai import AsyncOpenAI

from src.github_client.client import canonicalize_repo_url
from src.memory.semantic.chunker import chunk_text
from src.memory.semantic.models import IndexerConfig, RetrievalResult
from src.memory.semantic.schema import SEMANTIC_SCHEMA_SQL

logger = logging.getLogger(__name__)


def _dsn() -> str:
    """Build Postgres DSN from environment variables."""
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'sdlc_swarm')}"
        f":{os.getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
        f":{os.getenv('POSTGRES_PORT', '5433')}"
        f"/{os.getenv('POSTGRES_DB', 'sdlc_swarm')}"
    )


class SemanticStore:
    """Async semantic store backed by Postgres + pgvector.

    Usage::

        store = SemanticStore()
        await store.connect()
        # ... use store ...
        await store.close()

    Or as an async context manager::

        async with SemanticStore() as store:
            chunks = await store.retrieve(query="...", repo_url="...")
    """

    def __init__(
        self,
        dsn: str | None = None,
        config: IndexerConfig | None = None,
    ) -> None:
        self._dsn = dsn or _dsn()
        self._config = config or IndexerConfig()
        self._pool: asyncpg.Pool | None = None
        self._openai_client: AsyncOpenAI | None = None

    # ── Connection lifecycle ──────────────────────────────────────

    async def connect(self) -> None:
        """Open a connection pool and ensure schema exists."""
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        await self._ensure_schema()
        self._openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    async def close(self) -> None:
        """Close the connection pool and OpenAI client."""
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> SemanticStore:
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ── Schema ────────────────────────────────────────────────────

    async def _ensure_schema(self) -> None:
        """Create semantic tables and extension if they don't exist."""
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(SEMANTIC_SCHEMA_SQL)

    @property
    def pool(self) -> asyncpg.Pool:
        """Return the connection pool; raises if not connected."""
        if self._pool is None:
            msg = "SemanticStore not connected. Call await store.connect() first."
            raise RuntimeError(msg)
        return self._pool

    # ── Indexing ──────────────────────────────────────────────────

    async def index_repo(
        self,
        repo_url: str,
        repo_path: str,
        *,
        config: IndexerConfig | None = None,
    ) -> int:
        """Index a cloned repository into repo_chunks.

        Walks the repo directory, filters files by extension and size,
        chunks via token-aware splitter, embeds via OpenAI, and writes
        rows.  Idempotent: deletes existing rows for repo_url first.

        Args:
            repo_url: The canonical repository URL (for scoping).
            repo_path: Local filesystem path to the cloned repo root.
            config: Optional override for indexer configuration.

        Returns:
            Number of chunks written.
        """
        cfg = config or self._config
        canon_url = canonicalize_repo_url(repo_url)

        # Delete existing rows for this repo_url (idempotent, VAL-RAG-007)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM repo_chunks WHERE repo_url = $1",
                canon_url,
            )

        # Walk the repo and collect eligible files
        file_chunks: list[tuple[str, int, str]] = []  # (file_path, chunk_index, chunk_text)
        for file_path, chunk_index, chunk_text_content in self._walk_and_chunk(
            repo_path, cfg
        ):
            file_chunks.append((file_path, chunk_index, chunk_text_content))

        if not file_chunks:
            logger.info("No chunks produced for repo %s", canon_url)
            return 0

        # Embed all chunks in batches
        texts = [chunk_text for _, _, chunk_text in file_chunks]
        embeddings = await self._embed_texts(texts, cfg)

        # Write rows — asyncpg doesn't natively support VECTOR type,
        # so we cast the embedding string to vector in SQL.
        now = datetime.now(UTC)
        async with self.pool.acquire() as conn:
            for (file_path, chunk_index, chunk_text_content), embedding in zip(
                file_chunks, embeddings, strict=True
            ):
                chunk_id = uuid4()
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
                await conn.execute(
                    """
                    INSERT INTO repo_chunks
                        (id, repo_url, file_path, chunk_index, chunk_text, embedding, indexed_at)
                    VALUES ($1, $2, $3, $4, $5, $6::vector, $7)
                    """,
                    chunk_id,
                    canon_url,
                    file_path,
                    chunk_index,
                    chunk_text_content,
                    embedding_str,
                    now,
                )

        logger.info(
            "Indexed %d chunks for repo %s",
            len(file_chunks),
            canon_url,
        )
        return len(file_chunks)

    def _walk_and_chunk(
        self,
        repo_path: str,
        config: IndexerConfig,
    ) -> list[tuple[str, int, str]]:
        """Walk the repo directory and produce (file_path, chunk_index, chunk_text) tuples.

        Filters by extension, skips binary files and oversized files
        (VAL-RAG-009).
        """
        import os as _os

        results: list[tuple[str, int, str]] = []

        for dirpath, dirnames, filenames in _os.walk(repo_path):
            # Skip .git directory
            if ".git" in dirpath.split(_os.sep):
                continue

            # Skip common non-code directories
            dirnames[:] = [
                d
                for d in dirnames
                if d not in {".git", "__pycache__", "node_modules", ".venv", ".mypy_cache"}
            ]

            for filename in filenames:
                file_full_path = _os.path.join(dirpath, filename)

                # Get relative path from repo root
                rel_path = _os.path.relpath(file_full_path, repo_path)

                # Check extension
                _, ext = _os.path.splitext(filename)

                # Skip blocked extensions (binary files) — VAL-RAG-009
                if ext.lower() in config.blocked_extensions:
                    logger.debug("Skipping blocked extension: %s", rel_path)
                    continue

                # Skip non-allowed extensions
                if ext.lower() not in config.allowed_extensions:
                    logger.debug("Skipping non-allowed extension: %s", rel_path)
                    continue

                # Skip oversized files — VAL-RAG-009
                try:
                    file_size = _os.path.getsize(file_full_path)
                except OSError:
                    continue
                if file_size > config.max_file_bytes:
                    logger.debug(
                        "Skipping oversized file (%d bytes): %s",
                        file_size,
                        rel_path,
                    )
                    continue

                # Read file content
                try:
                    with open(file_full_path, encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except OSError:
                    continue

                if not content.strip():
                    continue

                # Chunk the content
                chunks = chunk_text(
                    content,
                    chunk_size_tokens=config.chunk_size_tokens,
                    chunk_overlap_tokens=config.chunk_overlap_tokens,
                )

                for chunk in chunks:
                    results.append((rel_path, chunk.chunk_index, chunk.text))

        return results

    async def _embed_texts(
        self,
        texts: list[str],
        config: IndexerConfig,
    ) -> list[list[float]]:
        """Embed a list of texts using OpenAI text-embedding-3-small.

        Batches the API calls to stay within rate limits.
        Returns a list of 1536-dim float vectors.
        """
        assert self._openai_client is not None

        all_embeddings: list[list[float]] = []
        batch_size = 100  # OpenAI allows up to 2048 inputs per call

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._openai_client.embeddings.create(
                model=config.embedding_model,
                input=batch,
                dimensions=config.embedding_dimensions,
            )

            # Sort by index to ensure correct ordering
            sorted_data = sorted(response.data, key=lambda x: x.index)
            for item in sorted_data:
                all_embeddings.append(item.embedding)

        return all_embeddings

    # ── Retrieval ─────────────────────────────────────────────────

    async def retrieve(
        self,
        *,
        query: str,
        repo_url: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-k chunks by cosine similarity scoped by repo_url.

        Embeds the query, then performs a cosine similarity search
        in the repo_chunks table scoped to the given repo_url
        (VAL-RAG-002, VAL-RAG-004).

        Args:
            query: Natural language search query.
            repo_url: Repository URL to scope the search.
            top_k: Number of results to return (default from config).

        Returns:
            List of RetrievalResult sorted by descending similarity.
        """
        k = top_k or self._config.top_k
        canon_url = canonicalize_repo_url(repo_url)

        # Embed the query
        assert self._openai_client is not None
        response = await self._openai_client.embeddings.create(
            model=self._config.embedding_model,
            input=[query],
            dimensions=self._config.embedding_dimensions,
        )
        query_embedding = response.data[0].embedding
        query_vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        # Cosine similarity search scoped by repo_url
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT file_path, chunk_index, chunk_text,
                       1 - (embedding <=> $1::vector) AS score
                FROM repo_chunks
                WHERE repo_url = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                query_vec_str,
                canon_url,
                k,
            )

        results: list[RetrievalResult] = []
        for row in rows:
            results.append(
                RetrievalResult(
                    file_path=row["file_path"],
                    chunk_index=row["chunk_index"],
                    chunk_text=row["chunk_text"],
                    score=float(row["score"]),
                )
            )

        return results

    # ── RAGRetriever protocol compatibility ──────────────────────

    async def retrieve_dicts(
        self,
        *,
        query: str,
        repo_url: str,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """Retrieve results as dicts, matching the RAGRetriever protocol.

        This is the protocol-compatible method used by the Planner agent
        (VAL-PLANNER-003).  Returns list[dict] with keys:
        file_path, chunk_text, score, chunk_index.
        """
        results = await self.retrieve(
            query=query,
            repo_url=repo_url,
            top_k=top_k,
        )
        return [
            {
                "file_path": r.file_path,
                "chunk_text": r.chunk_text,
                "score": r.score,
                "chunk_index": r.chunk_index,
            }
            for r in results
        ]

    # ── Query helpers ────────────────────────────────────────────

    async def is_indexed(self, repo_url: str) -> bool:
        """Check if a repo_url has any chunks in the store."""
        canon_url = canonicalize_repo_url(repo_url)
        async with self.pool.acquire() as conn:
            count: int = await conn.fetchval(
                "SELECT COUNT(*) FROM repo_chunks WHERE repo_url = $1",
                canon_url,
            )
        return count > 0

    async def get_chunk_count(self, repo_url: str) -> int:
        """Return the number of chunks for a repo_url."""
        canon_url = canonicalize_repo_url(repo_url)
        async with self.pool.acquire() as conn:
            count: int = await conn.fetchval(
                "SELECT COUNT(*) FROM repo_chunks WHERE repo_url = $1",
                canon_url,
            )
        return count or 0

    async def get_chunk_ids(self, repo_url: str) -> set[str]:
        """Return the set of chunk IDs for a repo_url."""
        canon_url = canonicalize_repo_url(repo_url)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id FROM repo_chunks WHERE repo_url = $1",
                canon_url,
            )
        return {str(row["id"]) for row in rows}

    async def get_vector_dims(self, repo_url: str) -> int | None:
        """Return the vector dimensions for chunks of a repo_url.

        Queries pgvector metadata to verify 1536-dim embeddings.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT vector_dims(embedding) AS dims
                FROM repo_chunks
                WHERE repo_url = $1
                LIMIT 1
                """,
                canonicalize_repo_url(repo_url),
            )
        if row is None:
            return None
        return int(row["dims"])

    async def verify_index_exists(self) -> bool:
        """Check that the ivfflat index exists on repo_chunks (VAL-RAG-006)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT indexdef
                FROM pg_indexes
                WHERE indexname = 'repo_chunks_embedding_idx'
                """
            )
        return row is not None

    async def get_index_definition(self) -> str | None:
        """Return the index definition for the ivfflat index (VAL-RAG-006)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT indexdef
                FROM pg_indexes
                WHERE indexname = 'repo_chunks_embedding_idx'
                """
            )
        if row is None:
            return None
        return str(row["indexdef"])
