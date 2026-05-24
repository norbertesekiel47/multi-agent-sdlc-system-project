"""Token-aware text splitter for repo chunking.

Splits text into chunks of ~512 tokens with ~64-token overlap
using tiktoken for accurate token counting (VAL-RAG-001).

Architecture reference: §2.7 pgvector Semantic Store.
"""

from __future__ import annotations

import logging

import tiktoken

logger = logging.getLogger(__name__)

# Use the cl100k_base encoding which is what text-embedding-3-small uses.
_ENCODING_NAME = "cl100k_base"


def _get_encoder() -> tiktoken.Encoding:
    """Get the tiktoken encoder for the embedding model."""
    return tiktoken.get_encoding(_ENCODING_NAME)


class Chunk:
    """A single text chunk with metadata."""

    __slots__ = ("text", "token_count", "start_offset", "chunk_index")

    def __init__(
        self,
        text: str,
        token_count: int,
        start_offset: int,
        chunk_index: int,
    ) -> None:
        self.text = text
        self.token_count = token_count
        self.start_offset = start_offset
        self.chunk_index = chunk_index

    def __repr__(self) -> str:
        return (
            f"Chunk(index={self.chunk_index}, tokens={self.token_count}, "
            f"chars={len(self.text)})"
        )


def chunk_text(
    text: str,
    *,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
) -> list[Chunk]:
    """Split text into token-aware chunks with overlap.

    Uses tiktoken to count tokens accurately.  Each chunk targets
    ``chunk_size_tokens`` tokens with ``chunk_overlap_tokens`` overlap
    between consecutive chunks (VAL-RAG-001).

    Args:
        text: The text to split.
        chunk_size_tokens: Target chunk size in tokens.
        chunk_overlap_tokens: Overlap between consecutive chunks.

    Returns:
        List of Chunk objects with metadata.
    """
    if not text.strip():
        return []

    encoder = _get_encoder()
    tokens = encoder.encode(text)

    if len(tokens) == 0:
        return []

    # If the text fits in one chunk, return it as-is
    if len(tokens) <= chunk_size_tokens:
        chunk_text_content = encoder.decode(tokens)
        return [
            Chunk(
                text=chunk_text_content,
                token_count=len(tokens),
                start_offset=0,
                chunk_index=0,
            )
        ]

    # Split into overlapping chunks
    chunks: list[Chunk] = []
    step = chunk_size_tokens - chunk_overlap_tokens
    if step <= 0:
        step = chunk_size_tokens  # fallback: no overlap

    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_content = encoder.decode(chunk_tokens)

        chunks.append(
            Chunk(
                text=chunk_text_content,
                token_count=len(chunk_tokens),
                start_offset=start,
                chunk_index=chunk_index,
            )
        )

        # Move to next chunk
        start += step
        chunk_index += 1

        # If we've reached the end, stop
        if end >= len(tokens):
            break

    return chunks


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string using tiktoken."""
    encoder = _get_encoder()
    return len(encoder.encode(text))
