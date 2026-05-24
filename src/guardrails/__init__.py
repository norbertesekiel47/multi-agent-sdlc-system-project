"""SDLC-Swarm Invariant Guardrails — runtime tool-call interception.

LangGraph middleware intercepting every tool call before dispatch.
Rules (architecture §2.8):
  (1) Block ``rm -rf`` on paths outside sandbox cwd
  (2) Block ``git push --force`` and non-allowlisted refspecs
  (3) Block subprocess commands containing secret patterns from env
  (4) Block HTTP requests to non-allowlisted hosts at application layer

When a rule fires:
  - Log to Langfuse (span tagged ``guardrail.violation``)
  - Write ``outcomes`` row with ``outcome='guardrail_block'``
  - Halt the agent
  - Escalate to HITL with violation details

Validation: VAL-GUARDRAIL-001 through VAL-GUARDRAIL-011.
"""

from __future__ import annotations

from src.guardrails.errors import GuardrailViolation
from src.guardrails.middleware import GuardrailMiddleware, GuardrailSandboxProxy
from src.guardrails.rules import GuardrailRuleBase, get_all_rules

__all__ = [
    "GuardrailMiddleware",
    "GuardrailRuleBase",
    "GuardrailSandboxProxy",
    "GuardrailViolation",
    "get_all_rules",
]
