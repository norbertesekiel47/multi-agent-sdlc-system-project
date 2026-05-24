"""Guardrail violation error types.

``GuardrailViolation`` is the typed error raised when a guardrail
rule fires.  It carries all the metadata needed for Langfuse logging,
outcomes row creation, and HITL escalation.
"""

from __future__ import annotations

from typing import Any

from src.logging.secret_filter import SecretRedactionFilter


class GuardrailViolation(Exception):  # noqa: N818
    """Raised when a guardrail rule blocks a tool call.

    Carries structured metadata for downstream processing:

    - **Langfuse span** (VAL-GUARDRAIL-007): ``to_langfuse_report()``
    - **Outcomes row** (VAL-GUARDRAIL-008): ``to_outcome_data()``
    - **HITL escalation** (VAL-GUARDRAIL-009): ``to_hitl_details()``

    All outputs are redacted — no secret values appear in any field.
    """

    def __init__(
        self,
        *,
        rule_name: str,
        tool_name: str,
        args_summary: str,
        detail: str,
    ) -> None:
        self.rule_name = rule_name
        self.tool_name = tool_name
        # Redact secrets from the args_summary before storing
        self._args_summary_raw = args_summary
        self.args_summary = SecretRedactionFilter._redact(args_summary)  # noqa: SLF001
        self.detail = detail
        super().__init__(
            f"Guardrail violation: rule={rule_name}, tool={tool_name}: {detail}"
        )

    def to_langfuse_report(self) -> dict[str, Any]:
        """Return a dict suitable for creating a Langfuse span.

        VAL-GUARDRAIL-007: span tagged ``guardrail.violation`` with
        ``rule_name``, ``tool_name``, and a redacted ``args_summary``.
        """
        return {
            "name": "guardrail.violation",
            "rule_name": self.rule_name,
            "tool_name": self.tool_name,
            "args_summary": self.args_summary,
            "detail": self.detail,
        }

    def to_outcome_data(self) -> dict[str, Any]:
        """Return a dict suitable for writing an ``outcomes`` row.

        VAL-GUARDRAIL-008: ``outcome='guardrail_block'`` with
        ``detail`` JSONB containing ``rule_name`` and ``tool_name``.
        """
        return {
            "outcome": "guardrail_block",
            "detail": {
                "rule_name": self.rule_name,
                "tool_name": self.tool_name,
                "args_summary": self.args_summary,
            },
        }

    def to_hitl_details(self) -> dict[str, Any]:
        """Return HITL escalation details for the interrupt payload.

        VAL-GUARDRAIL-009: provides cause, rule name, and a
        human-readable explanation.
        """
        return {
            "cause": "guardrail_block",
            "rule_name": self.rule_name,
            "tool_name": self.tool_name,
            "explanation": self.detail,
            "args_summary": self.args_summary,
        }
