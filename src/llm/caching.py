"""Prompt caching support for Coder/Reviewer agents.

In M1, this module provides the data structures and interface
for prompt caching.  Actual cache-marker injection into prompts
will be implemented in M2 when the Coder and Reviewer agents
are built.

Caching is a cost optimization only — it MUST NOT change agent
semantics (architecture §2.10).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class StructuredPrompt(BaseModel):
    """A prompt split into static (cacheable) and dynamic blocks.

    Static blocks are tagged with cache markers per OpenRouter protocol.
    Dynamic blocks change on every call and are not cached.
    """

    static: str
    dynamic: str

    def to_messages(self, *, system_first: bool = True) -> list[dict[str, Any]]:
        """Convert to OpenRouter-compatible message list.

        Static content is placed first (system message) so that
        OpenRouter can cache it.  Dynamic content follows as the
        user message.
        """
        messages: list[dict[str, Any]] = []
        if system_first:
            messages.append({"role": "system", "content": self.static})
            messages.append({"role": "user", "content": self.dynamic})
        else:
            messages.append({"role": "user", "content": self.dynamic})
            messages.append({"role": "system", "content": self.static})
        return messages


def build_structured_prompt(
    *,
    system_instructions: str,
    repo_context: str,
    current_edit: str = "",
    prior_review: str = "",
) -> StructuredPrompt:
    """Build a structured prompt with static and dynamic blocks.

    Static block = system instructions + repo context (cacheable).
    Dynamic block = current edit + prior review result (changes per call).
    """
    static = f"{system_instructions}\n\n---\nRepo context:\n{repo_context}"
    dynamic_parts: list[str] = []
    if current_edit:
        dynamic_parts.append(f"Current edit:\n{current_edit}")
    if prior_review:
        dynamic_parts.append(f"Prior review:\n{prior_review}")
    dynamic = "\n\n".join(dynamic_parts) if dynamic_parts else "No edit context yet."

    return StructuredPrompt(static=static, dynamic=dynamic)
