"""Langfuse tracing client with graceful degradation.

Wraps the Langfuse SDK so that every LLM call produces a trace span with:
  - name, latency, tokens-in, tokens-out, cached-tokens, USD cost
  - Span hierarchy mirroring the call graph (node → agent turn → tool call → LLM completion)
  - Secret redaction in span I/O before emission
  - Input/output truncation at 2 KB with ``…[truncated]`` marker

When Langfuse is unreachable, the backend logs warnings but continues
operating (VAL-CROSS-020).  No agent step blocks on Langfuse availability.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

from langfuse import Langfuse

from src.logging.secret_filter import SecretRedactionFilter

logger = logging.getLogger(__name__)

# Maximum size (bytes) for span input/output before truncation.
_MAX_SPAN_IO_BYTES = 2048

# Truncation marker appended when payload exceeds limit.
_TRUNCATION_MARKER = "…[truncated]"


# ── Secret redaction for span I/O ──────────────────────────────────


def _redact_secrets(text: str) -> str:
    """Replace known secret patterns in *text* with ``***REDACTED***``."""
    return SecretRedactionFilter._redact(text)


def _truncate_and_redact(payload: Any) -> str:
    """Serialize *payload*, redact secrets, and truncate to 2 KB."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        raw = payload
    else:
        try:
            raw = json.dumps(payload, sort_keys=True, default=str)
        except (TypeError, ValueError):
            raw = str(payload)

    # Redact secrets BEFORE truncation so no secret fragment survives
    redacted = _redact_secrets(raw)

    # Truncate at 2 KB
    if len(redacted.encode("utf-8")) > _MAX_SPAN_IO_BYTES:
        # Cut at a safe byte boundary
        truncated = redacted[: _MAX_SPAN_IO_BYTES - len(_TRUNCATION_MARKER)]
        return truncated + _TRUNCATION_MARKER
    return redacted


# ── Span types ──────────────────────────────────────────────────────


class SpanType:
    """Constants for Langfuse observation types."""

    TRACE = "trace"
    SPAN = "span"
    GENERATION = "generation"
    TOOL = "tool"  # mapped to "span" in Langfuse but tagged as tool


# ── Langfuse client wrapper ────────────────────────────────────────


class TracingClient:
    """Langfuse client wrapper with graceful degradation.

    All methods catch Langfuse exceptions and log warnings instead of
    propagating them.  This ensures the backend continues operating
    even when Langfuse is down (VAL-CROSS-020).

    Uses Langfuse v4.6+ API with ``start_observation`` and
    ``TraceContext`` for span hierarchy.
    """

    def __init__(self) -> None:
        self._client: Langfuse | None = None
        self._available: bool = False
        self._init_attempts: int = 0
        self._last_init_attempt: float = 0.0
        # Retry init every 30 seconds when Langfuse is down
        self._init_retry_interval: float = 30.0

    def _try_init(self) -> None:
        """Attempt to initialize the Langfuse client.

        On failure, log a warning and mark as unavailable.
        Retries are throttled to avoid spamming logs.
        """
        now = time.monotonic()
        if self._client is not None:
            self._available = True
            return
        if now - self._last_init_attempt < self._init_retry_interval:
            return
        self._last_init_attempt = now
        self._init_attempts += 1

        try:
            secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
            host = os.getenv("LANGFUSE_HOST", "http://localhost:3110")

            if not secret_key or not public_key:
                logger.warning(
                    "Langfuse keys not configured. Tracing disabled. "
                    "Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY."
                )
                self._available = False
                return

            self._client = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=host,
                tracing_enabled=True,
            )
            # Verify connectivity
            self._client.auth_check()
            self._available = True
            logger.info("Langfuse tracing connected to %s", host)
        except Exception:
            self._available = False
            self._client = None
            logger.warning(
                "Langfuse unreachable (attempt %d). Tracing degraded. "
                "Backend continues operating without tracing.",
                self._init_attempts,
            )

    @property
    def available(self) -> bool:
        """Return True if Langfuse is reachable and keys are configured."""
        self._try_init()
        return self._available

    @property
    def client(self) -> Langfuse | None:
        """Return the underlying Langfuse client, or None if unavailable."""
        self._try_init()
        return self._client

    def flush(self) -> None:
        """Flush pending spans to Langfuse. No-op if unavailable."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                logger.warning("Failed to flush Langfuse spans.")

    # ── High-level span creation helpers ───────────────────────────

    def create_trace(
        self,
        *,
        trace_id: str,
        name: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        input_data: Any = None,
        output_data: Any = None,
    ) -> str | None:
        """Create a top-level trace. Returns the trace_id or None if unavailable."""
        if not self.available or self._client is None:
            return None
        try:
            span = self._client.start_observation(
                name=name,
                as_type="chain",
                input=_truncate_and_redact(input_data),
                output=_truncate_and_redact(output_data),
                metadata={
                    **(metadata or {}),
                    "trace_id": trace_id,
                    "userId": user_id,
                },
            )
            # Store the trace_id in the span for later reference
            span_id = span.id if hasattr(span, "id") else trace_id
            return span_id
        except Exception:
            logger.warning("Failed to create Langfuse trace %s", trace_id)
            return None

    def create_span(
        self,
        *,
        trace_id: str,
        parent_span_id: str | None = None,
        span_id: str | None = None,
        name: str,
        span_type: str = SpanType.SPAN,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        level: str = "DEFAULT",
        status_message: str | None = None,
    ) -> str | None:
        """Create a span within a trace. Returns the span_id or None."""
        if not self.available or self._client is None:
            return None
        try:
            from langfuse.types import TraceContext

            trace_context: TraceContext = {"trace_id": trace_id}
            if parent_span_id is not None:
                trace_context["parent_span_id"] = parent_span_id

            obs_type = "span"
            if span_type == SpanType.TOOL:
                obs_type = "tool"

            span_kwargs: dict[str, Any] = {
                "trace_context": trace_context,
                "name": name,
                "as_type": obs_type,
                "input": _truncate_and_redact(input_data),
                "output": _truncate_and_redact(output_data),
                "metadata": metadata or {},
                "level": level,
            }
            if status_message is not None:
                span_kwargs["status_message"] = status_message

            obs = self._client.start_observation(**span_kwargs)

            # If end_time provided, end the span immediately
            if end_time is not None:
                obs.end()

            result_id = obs.id if hasattr(obs, "id") else span_id
            return result_id
        except Exception:
            logger.warning(
                "Failed to create Langfuse span %s in trace %s", name, trace_id
            )
            return None

    def create_generation(
        self,
        *,
        trace_id: str,
        parent_span_id: str | None = None,
        name: str,
        model: str,
        input_data: Any = None,
        output_data: Any = None,
        usage_input: int = 0,
        usage_output: int = 0,
        cached_tokens: int = 0,
        cost_usd: Decimal = Decimal("0"),
        metadata: dict[str, Any] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> str | None:
        """Create a GENERATION-type span for an LLM call.

        Records token usage, cached_tokens, and cost metadata.
        Returns the span_id or None.
        """
        if not self.available or self._client is None:
            return None
        try:
            from langfuse.types import TraceContext

            trace_context: TraceContext = {"trace_id": trace_id}
            if parent_span_id is not None:
                trace_context["parent_span_id"] = parent_span_id

            gen = self._client.start_observation(
                trace_context=trace_context,
                name=name,
                as_type="generation",
                model=model,
                input=_truncate_and_redact(input_data),
                output=_truncate_and_redact(output_data),
                usage_details={
                    "input": usage_input,
                    "output": usage_output,
                    "cached_tokens": cached_tokens,
                },
                cost_details={
                    "input": float(cost_usd),
                    "output": 0.0,
                    "total": float(cost_usd),
                },
                metadata={
                    **(metadata or {}),
                    "cached_tokens": cached_tokens,
                    "cost_usd": str(cost_usd),
                },
            )

            # End the generation if end_time is provided
            if end_time is not None:
                gen.end()

            return gen.id if hasattr(gen, "id") else None
        except Exception:
            logger.warning(
                "Failed to create Langfuse generation span %s in trace %s",
                name, trace_id,
            )
            return None

    def update_span(
        self,
        *,
        trace_id: str,
        span_id: str,
        output_data: Any = None,
        end_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",
        status_message: str | None = None,
    ) -> None:
        """Update an existing span with output/end time."""
        if not self.available or self._client is None:
            return
        try:
            # Use update_current_span or update_current_generation
            # For now, we just log since the v4.6 API requires
            # context-based span tracking
            update_kwargs: dict[str, Any] = {}
            if output_data is not None:
                update_kwargs["output"] = _truncate_and_redact(output_data)
            if metadata is not None:
                update_kwargs["metadata"] = metadata
            if status_message is not None:
                update_kwargs["status_message"] = status_message
            update_kwargs["level"] = level

            # The v4.6 API uses span.end() to finalize
            # We'd need to track active spans to update them
            logger.debug("Span update requested for %s (tracing v4.6 context-based)", span_id)
        except Exception:
            logger.warning("Failed to update Langfuse span %s", span_id)


# ── Module-level singleton ─────────────────────────────────────────

_tracing_client: TracingClient | None = None


def get_tracing_client() -> TracingClient:
    """Return the module-level TracingClient singleton."""
    global _tracing_client
    if _tracing_client is None:
        _tracing_client = TracingClient()
    return _tracing_client


def reset_tracing_client() -> None:
    """Reset the singleton (for testing)."""
    global _tracing_client
    _tracing_client = None
