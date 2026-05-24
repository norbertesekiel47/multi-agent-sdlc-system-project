"""Pydantic models for the pgvector semantic store.

Models map to the repo_chunks table and retrieval results
per architecture.md §2.7.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RepoChunkRow(BaseModel):
    """Maps to the ``repo_chunks`` table."""

    id: UUID
    repo_url: str
    file_path: str
    chunk_index: int
    chunk_text: str
    embedding: list[float] = Field(
        ..., description="1536-dim vector from OpenAI text-embedding-3-small"
    )
    indexed_at: datetime


class RetrievalResult(BaseModel):
    """A single RAG retrieval result from pgvector cosine similarity.

    Includes the file path, chunk text, similarity score, and
    chunk_index.  Results are scoped by repo_url in the retrieval
    query (VAL-RAG-002, VAL-RAG-004).
    """

    file_path: str = Field(..., min_length=1, description="Source file path")
    chunk_index: int = Field(..., ge=0, description="Chunk position in file")
    chunk_text: str = Field(..., min_length=1, description="Chunk content")
    score: float = Field(
        default=0.0, description="Cosine similarity score (1 - cosine distance)"
    )


class IndexerConfig(BaseModel):
    """Configuration for the repo indexer.

    Controls chunking parameters, file filtering, and embedding.
    """

    chunk_size_tokens: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Target chunk size in tokens (~512 per spec)",
    )
    chunk_overlap_tokens: int = Field(
        default=64,
        ge=0,
        le=512,
        description="Overlap between consecutive chunks (~64 per spec)",
    )
    max_file_bytes: int = Field(
        default=1_048_576,  # 1 MiB
        ge=1,
        description="Maximum file size in bytes (default 1 MiB)",
    )
    allowed_extensions: frozenset[str] = Field(
        default=frozenset(
            {".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".txt", ".yaml", ".yml",
             ".toml", ".json", ".cfg", ".ini", ".sh", ".bash", ".zsh", ".rst",
             ".html", ".css", ".scss", ".go", ".rs", ".java", ".c", ".cpp", ".h"}
        ),
        description="File extensions to index",
    )
    blocked_extensions: frozenset[str] = Field(
        default=frozenset(
            {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico",
             ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
             ".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a",
             ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4",
             ".avi", ".mov", ".wav", ".flac", ".webp"}
        ),
        description="Binary/file extensions to skip",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name",
    )
    embedding_dimensions: int = Field(
        default=1536,
        ge=1,
        description="Embedding vector dimensions",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=100,
        description="Default number of retrieval results",
    )

    model_config = {"frozen": True}
