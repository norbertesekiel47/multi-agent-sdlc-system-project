"""SQL DDL for the pgvector semantic store.

Creates the ``repo_chunks`` table with pgvector support
per architecture.md §2.7.

Schema:
  - repo_chunks: (id, repo_url, file_path, chunk_index, chunk_text,
                  embedding VECTOR(1536), indexed_at)
  - ivfflat index on embedding with vector_cosine_ops
  - b-tree index on repo_url for scoped retrieval
"""

from __future__ import annotations

SEMANTIC_SCHEMA_SQL: str = """\
-- ────────────────────────────────────────────────────────────────
-- pgvector Semantic Store — architecture.md §2.7
-- ────────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS repo_chunks (
  id              UUID PRIMARY KEY,
  repo_url        TEXT NOT NULL,
  file_path       TEXT NOT NULL,
  chunk_index     INTEGER NOT NULL,
  chunk_text      TEXT NOT NULL,
  embedding       VECTOR(1536) NOT NULL,
  indexed_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS repo_chunks_repo_url_idx ON repo_chunks(repo_url);
CREATE INDEX IF NOT EXISTS repo_chunks_embedding_idx ON repo_chunks
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""
