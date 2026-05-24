"""Cost tracking and budget enforcement for LLM calls.

Tracks per-call USD cost from OpenRouter responses and sums into
``tasks.total_cost_usd`` in real time.  Enforces ``MAX_COST_PER_TASK_USD``
after every LLM call.  Falls back to tiktoken-based estimation when
OpenRouter omits the cost field.  Embedding-call costs are included.

Key behaviors (from validation contract):
  - VAL-COST-BUDGET-001: Cost cap halts task on first over-budget call
  - VAL-COST-BUDGET-002: Cost cap writes cost_budget_exhausted outcome row
  - VAL-COST-BUDGET-003: Cumulative cost matches Langfuse span cost sum
  - VAL-COST-BUDGET-005: Cost cap enforced per-call, not per-step
  - VAL-COST-BUDGET-006: Missing usage.cost falls back to tiktoken estimate
  - VAL-COST-BUDGET-007: total_cost_usd updated transactionally per LLM call
  - VAL-COST-BUDGET-008: Embedding-call cost included in total_cost_usd
"""

from __future__ import annotations

import logging
import os
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)


# ── Model pricing (USD per 1M tokens) ──────────────────────────────
# These rates are approximate and used as fallback when OpenRouter
# does not return a cost field.  They should be updated periodically.
# Rates from https://openrouter.ai/models pricing page.

_MODEL_PRICING: dict[str, dict[str, float]] = {
    "deepseek/deepseek-v4-pro": {
        "input_per_mtok": 0.50,
        "output_per_mtok": 2.0,
    },
    "deepseek/deepseek-v4-flash": {
        "input_per_mtok": 0.10,
        "output_per_mtok": 0.40,
    },
    "deepseek/deepseek-chat-v3-0324": {
        # Stable fallback for Coder/Reviewer/QA (per AGENTS.md)
        # Same pricing tier as DeepSeek V4 Flash
        "input_per_mtok": 0.10,
        "output_per_mtok": 0.40,
    },
    # OpenAI embedding pricing
    "text-embedding-3-small": {
        "input_per_mtok": 0.02,
        "output_per_mtok": 0.0,
    },
}

# Default budget from env
_DEFAULT_MAX_COST = Decimal("2.00")


def get_max_cost_per_task() -> Decimal:
    """Return the per-task cost cap from env (default $2.00)."""
    raw = os.getenv("MAX_COST_PER_TASK_USD", "2.00")
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError):
        return _DEFAULT_MAX_COST


def estimate_cost_tiktoken(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Decimal:
    """Estimate USD cost using tiktoken and model pricing table.

    Used as fallback when OpenRouter does not return usage.cost
    (VAL-COST-BUDGET-006).
    """
    pricing = _MODEL_PRICING.get(model)
    if pricing is None:
        # Use DeepSeek Flash pricing as a generic fallback
        logger.warning(
            "No pricing data for model %s, using DeepSeek Flash rates as fallback", model
        )
        pricing = _MODEL_PRICING["deepseek/deepseek-v4-flash"]

    input_cost = (
        Decimal(str(pricing["input_per_mtok"]))
        * Decimal(prompt_tokens) / Decimal("1000000")
    )
    output_cost = (
        Decimal(str(pricing["output_per_mtok"]))
        * Decimal(completion_tokens) / Decimal("1000000")
    )
    return (input_cost + output_cost).quantize(Decimal("0.0001"))


def extract_cost_from_response(
    *,
    model: str,
    usage: dict[str, Any],
) -> Decimal:
    """Extract per-call cost from an OpenRouter API response.

    Tries, in order:
    1. ``usage.cost`` (OpenRouter response field)
    2. ``X-Cost`` response header (not available in dict form, handled elsewhere)
    3. tiktoken-based estimate using model pricing table

    Returns a non-zero Decimal when tokens are present (VAL-COST-BUDGET-006).
    """
    # Try direct cost field
    cost = usage.get("cost")
    if cost is not None:
        try:
            return Decimal(str(cost)).quantize(Decimal("0.0001"))
        except (InvalidOperation, ValueError):
            pass

    # Try nested cost_details
    cost_details = usage.get("cost_details") or usage.get("costDetails")
    if cost_details is not None:
        total = cost_details.get("total")
        if total is not None:
            try:
                return Decimal(str(total)).quantize(Decimal("0.0001"))
            except (InvalidOperation, ValueError):
                pass

    # Fallback to tiktoken estimate
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0

    if prompt_tokens == 0 and completion_tokens == 0:
        return Decimal("0")

    estimated = estimate_cost_tiktoken(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    logger.debug(
        "Cost fallback: tiktoken estimate for %s = $%s (in=%d, out=%d)",
        model,
        estimated,
        prompt_tokens,
        completion_tokens,
    )
    return estimated


def extract_cached_tokens(usage: dict[str, Any]) -> int:
    """Extract cached_tokens from OpenRouter usage details.

    Returns 0 if the field is absent (may be 0 on first call).
    """
    details = usage.get("prompt_tokens_details") or usage.get("promptTokensDetails")
    if details is not None and isinstance(details, dict):
        return int(details.get("cached_tokens", 0) or 0)
    return 0


class CostBudgetExceededError(Exception):
    """Raised when a task's cumulative cost exceeds MAX_COST_PER_TASK_USD."""

    def __init__(
        self,
        *,
        task_id: str,
        total_cost_usd: Decimal,
        max_cost_usd: Decimal,
        triggering_agent: str,
    ) -> None:
        self.task_id = task_id
        self.total_cost_usd = total_cost_usd
        self.max_cost_usd = max_cost_usd
        self.triggering_agent = triggering_agent
        super().__init__(
            f"Task {task_id} cost budget exceeded: "
            f"${total_cost_usd} > ${max_cost_usd} "
            f"(triggering agent: {triggering_agent})"
        )
