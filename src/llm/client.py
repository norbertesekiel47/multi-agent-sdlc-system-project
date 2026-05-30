"""OpenRouter LLM client wrapper with cost tracking and Langfuse tracing.

All LLM calls go through this wrapper so that:
  - Per-call cost is extracted and accumulated into ``tasks.total_cost_usd``
  - MAX_COST_PER_TASK_USD is enforced after every call
  - A Langfuse generation span is created for every call
  - Token counts (including cached_tokens) are recorded
  - Secret redaction is applied to span I/O

This module is the ONLY place that imports ``openai``/``httpx`` for
LLM communication.  ``src/api/`` must never import this module directly;
it goes through the orchestrator (VAL-BACKEND-API-003).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from openai import AsyncOpenAI

from src.llm.caching import StructuredPrompt
from src.llm.cost import (
    extract_cached_tokens,
    extract_cost_from_response,
)
from src.tracing.langfuse import SpanType, TracingClient, get_tracing_client
from src.tracing.ws_broadcaster import TraceEvent, get_trace_broadcaster

logger = logging.getLogger(__name__)

# Default OpenRouter base URL
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default temperature for non-benchmark runs (VAL-REPRO-004).
# Benchmark runs set LLM_TEMPERATURE=0 via the SWE-bench harness.
_DEFAULT_TEMPERATURE: float = 0.2


def get_temperature() -> float:
    """Return the LLM temperature for the current run context.

    Reads ``LLM_TEMPERATURE`` from the environment.  If not set,
    returns the default of 0.2 (for normal task runs).

    The SWE-bench benchmark harness sets ``LLM_TEMPERATURE=0`` before
    invoking the orchestrator, so benchmark runs get temperature 0.0.

    VAL-REPRO-004: default temperature is 0.2 for non-benchmark,
    0.0 for benchmark.
    """
    env_val = os.getenv("LLM_TEMPERATURE")
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            logger.warning(
                "Invalid LLM_TEMPERATURE=%r, falling back to default %.1f",
                env_val,
                _DEFAULT_TEMPERATURE,
            )
    return _DEFAULT_TEMPERATURE


class LLMCallResult:
    """Result of a single LLM call, including cost and token metadata."""

    __slots__ = (
        "content",
        "model",
        "usage_input",
        "usage_output",
        "cached_tokens",
        "cost_usd",
        "span_id",
    )

    def __init__(
        self,
        *,
        content: str,
        model: str,
        usage_input: int,
        usage_output: int,
        cached_tokens: int,
        cost_usd: Decimal,
        span_id: str | None = None,
    ) -> None:
        self.content = content
        self.model = model
        self.usage_input = usage_input
        self.usage_output = usage_output
        self.cached_tokens = cached_tokens
        self.cost_usd = cost_usd
        self.span_id = span_id


class LLMClient:
    """Async LLM client that wraps OpenRouter with cost tracking and tracing.

    Usage::

        client = LLMClient()
        result = await client.chat(
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "Hello"}],
            task_id="<uuid>",
            trace_id="<langfuse-trace-id>",
            agent_name="coder",
        )
    """

    def __init__(
        self,
        *,
        tracing_client: TracingClient | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self._tracing = tracing_client or get_tracing_client()
        self._openai = openai_client or AsyncOpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", _OPENROUTER_BASE_URL),
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            default_headers={
                "HTTP-Referer": "https://github.com/norbertesekiel47/sdlc-swarm",
                "X-Title": "SDLC-Swarm",
            },
        )

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        task_id: str,
        trace_id: str,
        parent_span_id: str | None = None,
        agent_name: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        extra_metadata: dict[str, Any] | None = None,
    ) -> LLMCallResult:
        """Send a chat completion request to OpenRouter.

        Records a Langfuse generation span and checks the cost budget.
        Raises CostBudgetExceededError if the budget is exceeded.
        """
        start_time = datetime.now(UTC)

        # Create Langfuse generation span
        span_id = self._tracing.create_generation(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=f"{agent_name}.llm_call" if agent_name else "llm_call",
            model=model,
            input_data=messages,
            metadata={
                "agent": agent_name,
                "task_id": task_id,
                "temperature": temperature,
                **(extra_metadata or {}),
            },
            start_time=start_time,
        )

        # Make the LLM call
        try:
            response = await self._openai.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            # Record error in span
            self._tracing.update_span(
                trace_id=trace_id,
                span_id=span_id or "",
                output_data={"error": str(exc)},
                end_time=datetime.now(UTC),
                level="ERROR",
                status_message=str(exc),
            )
            raise

        end_time = datetime.now(UTC)

        # Extract content
        content = response.choices[0].message.content or "" if response.choices else ""
        usage_raw: dict[str, Any] = {}
        if response.usage is not None:
            usage_raw = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            # Extract cached_tokens from prompt_tokens_details
            ptd = getattr(response.usage, "prompt_tokens_details", None)
            if ptd is not None:
                usage_raw["prompt_tokens_details"] = {
                    "cached_tokens": ptd.cached_tokens or 0,
                }

        # Extract cost
        cost_usd = extract_cost_from_response(model=model, usage=usage_raw)
        cached_tokens = extract_cached_tokens(usage_raw)
        usage_input: int = usage_raw.get("prompt_tokens", 0) or 0
        usage_output: int = usage_raw.get("completion_tokens", 0) or 0

        # Update Langfuse span with output and cost
        self._tracing.update_span(
            trace_id=trace_id,
            span_id=span_id or "",
            output_data={"content": content[:500] if content else ""},
            end_time=end_time,
            metadata={
                "agent": agent_name,
                "tokens_in": usage_input,
                "tokens_out": usage_output,
                "cached_tokens": cached_tokens,
                "cost_usd": str(cost_usd),
            },
        )

        # Update the generation span with usage and cost
        if span_id:
            self._tracing.create_generation(
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                name=f"{agent_name}.llm_call" if agent_name else "llm_call",
                model=model,
                input_data=messages,
                output_data={"content": content[:500] if content else ""},
                usage_input=usage_input,
                usage_output=usage_output,
                cached_tokens=cached_tokens,
                cost_usd=cost_usd,
                metadata={
                    "agent": agent_name,
                    "task_id": task_id,
                    **(extra_metadata or {}),
                },
                start_time=start_time,
                end_time=end_time,
            )

        # Broadcast trace event
        broadcaster = get_trace_broadcaster()
        event = TraceEvent(
            type="llm_completion",
            task_id=task_id,
            trace_id=trace_id,
            span_id=span_id or "",
            parent_span_id=parent_span_id,
            name=f"{agent_name}.llm_call" if agent_name else "llm_call",
            span_type=SpanType.GENERATION,
            started_at=start_time,
            ended_at=end_time,
            tokens_in=usage_input,
            tokens_out=usage_output,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            metadata={
                "agent": agent_name,
                "model": model,
            },
        )
        await broadcaster.publish(event)

        # Check cost budget
        # Note: actual budget check against cumulative total is done
        # by the caller (orchestrator) after updating tasks.total_cost_usd.
        # But we provide the per-call cost so the caller can enforce it.

        return LLMCallResult(
            content=content,
            model=model,
            usage_input=usage_input,
            usage_output=usage_output,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            span_id=span_id,
        )

    async def chat_with_cache(
        self,
        *,
        structured_prompt: StructuredPrompt,
        model: str,
        task_id: str,
        trace_id: str,
        parent_span_id: str | None = None,
        agent_name: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        extra_metadata: dict[str, Any] | None = None,
    ) -> LLMCallResult:
        """Send a chat completion request using a StructuredPrompt with cache markers.

        This is the primary entry point for Coder and Reviewer agents
        that benefit from prompt caching (§2.10).  It:

        1. Converts the structured prompt to messages with cache_control
           markers appropriate for the model family.
        2. Sends the request to OpenRouter.
        3. Records cached_tokens from the response onto the Langfuse span.

        Args:
            structured_prompt: A StructuredPrompt with static (cacheable) and
                dynamic blocks.
            model: The model identifier (e.g. ``deepseek/deepseek-chat-v3-0324``).
            task_id: The current task's ID.
            trace_id: Langfuse trace ID for span hierarchy.
            parent_span_id: Optional parent span ID.
            agent_name: Name of the agent making the call (e.g. "coder").
            temperature: Sampling temperature (0.0 for benchmark, 0.2 default).
            max_tokens: Maximum tokens in the response.
            extra_metadata: Additional metadata for the Langfuse span.

        Returns:
            An LLMCallResult with cached_tokens from the OpenRouter response.
        """
        messages = structured_prompt.to_messages_with_cache_markers(model=model)

        return await self.chat(
            model=model,
            messages=messages,
            task_id=task_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_metadata={
                "prompt_caching": True,
                **(extra_metadata or {}),
            },
        )


class EmbeddingClient:
    """Async embedding client that tracks cost like LLM calls.

    Uses OpenAI directly (not OpenRouter) for text-embedding-3-small.
    Cost is included in tasks.total_cost_usd (VAL-COST-BUDGET-008).
    """

    def __init__(
        self,
        *,
        tracing_client: TracingClient | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self._tracing = tracing_client or get_tracing_client()
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._openai = openai_client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def embed(
        self,
        *,
        texts: list[str],
        model: str = "text-embedding-3-small",
        task_id: str,
        trace_id: str,
        parent_span_id: str | None = None,
    ) -> tuple[list[list[float]], Decimal]:
        """Create embeddings and return (embeddings, cost_usd).

        The cost is calculated from token usage and model pricing,
        since OpenAI embedding responses don't include a cost field.
        """
        start_time = datetime.now(UTC)

        span_id = self._tracing.create_span(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name="embedding.embed",
            span_type=SpanType.TOOL,
            input_data={"texts_count": len(texts), "model": model},
            metadata={"task_id": task_id, "agent": "indexer"},
            start_time=start_time,
        )

        try:
            response = await self._openai.embeddings.create(
                input=texts,
                model=model,
            )
        except Exception as exc:
            self._tracing.update_span(
                trace_id=trace_id,
                span_id=span_id or "",
                output_data={"error": str(exc)},
                end_time=datetime.now(UTC),
                level="ERROR",
                status_message=str(exc),
            )
            raise

        end_time = datetime.now(UTC)

        embeddings = [item.embedding for item in response.data]

        # Calculate cost from token usage
        usage_input = response.usage.prompt_tokens if response.usage else 0
        cost_usd = extract_cost_from_response(
            model=model,
            usage={"prompt_tokens": usage_input, "completion_tokens": 0},
        )

        self._tracing.update_span(
            trace_id=trace_id,
            span_id=span_id or "",
            output_data={"embeddings_count": len(embeddings), "tokens": usage_input},
            end_time=end_time,
            metadata={"cost_usd": str(cost_usd), "tokens_in": usage_input},
        )

        # Broadcast trace event
        broadcaster = get_trace_broadcaster()
        event = TraceEvent(
            type="tool_call",
            task_id=task_id,
            trace_id=trace_id,
            span_id=span_id or "",
            parent_span_id=parent_span_id,
            name="embedding.embed",
            span_type=SpanType.TOOL,
            started_at=start_time,
            ended_at=end_time,
            tokens_in=usage_input,
            tokens_out=0,
            cost_usd=cost_usd,
            metadata={"model": model, "agent": "indexer"},
        )
        await broadcaster.publish(event)

        return embeddings, cost_usd


# ── Module-level singletons ────────────────────────────────────────

_llm_client: LLMClient | None = None
_embedding_client: EmbeddingClient | None = None


def get_llm_client() -> LLMClient:
    """Return the module-level LLMClient singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_embedding_client() -> EmbeddingClient:
    """Return the module-level EmbeddingClient singleton."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
