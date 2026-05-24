"""SDLC-Swarm Failure-Mode Mitigations — architecture §2.9.

Three independent mechanisms, each with deterministic triggers:

1. **Retry budget** — per (agent, step) counter, max 3 attempts,
   then halt + escalate with ``outcome='retry_budget_exhausted'``.

2. **Loop detection** — sliding window of last 5 tool calls per agent;
   same ``(tool_name, sha256(canonical_json_args))`` appearing 3+ times
   triggers halt + escalate with ``outcome='loop_detected'``.
   Canonicalization: sorted keys, no whitespace.

3. **Uncertainty escalation** — two deterministic paths, first-trigger-wins.
   Path A: Pydantic structured-output validation fails 3 times in a row.
   Path B: external signals (persistent test failure across 3 retries,
   Reviewer rejects same diff_hash twice, tool-error rate >50% in
   10-call window).  LLM self-confidence is explicitly NOT used.

When a trigger fires: log to Langfuse, write ``outcomes`` row,
raise a LangGraph ``interrupt()`` with the trigger reason,
surface to dashboard.

Validation: VAL-RETRY-001 through VAL-RETRY-004,
VAL-LOOP-DETECT-001 through VAL-LOOP-DETECT-005,
VAL-UNCERTAINTY-001 through VAL-UNCERTAINTY-010,
VAL-CROSS-015, VAL-CROSS-016.
"""

from src.failure_modes.loop_detection import (
    LoopDetector,
    canonical_json,
    compute_args_hash,
)
from src.failure_modes.retry_budget import RetryBudget
from src.failure_modes.uncertainty import (
    UncertaintyEscalation,
    UncertaintyTrigger,
)

__all__ = [
    "LoopDetector",
    "RetryBudget",
    "UncertaintyEscalation",
    "UncertaintyTrigger",
    "canonical_json",
    "compute_args_hash",
]
