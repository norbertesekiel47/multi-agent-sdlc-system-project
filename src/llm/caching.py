"""Prompt caching support for Coder/Reviewer agents (§2.10).

Caching is a cost optimization only — it MUST NOT change agent
semantics.  The prompt builder splits into static (repo context +
system instructions) and dynamic (current edit, prior review result)
blocks.  Static blocks are tagged with cache markers per OpenRouter
protocol.

Cache marker behavior by model family:
  - Anthropic-family: ``cache_control: {"type": "ephemeral"}`` on
    the last content block of the system message.
  - DeepSeek: auto-caches (no ``cache_control`` keys in the request).
  - Google Gemini: ``cache_control: {"type": "ephemeral"}`` on the
    last content block (same as Anthropic).
  - OpenAI: implicit caching (no ``cache_control`` keys needed).

The ``CACHING_ENABLED`` environment variable controls whether
cache markers are injected.  When ``false``, no ``cache_control``
markers appear in any request, regardless of model family.
The text content of messages is identical regardless of this flag,
preserving the semantic invariant (VAL-CACHING-006).

The LLM client wrapper records ``usage.prompt_tokens_details.cached_tokens``
from each OpenRouter response onto the per-call Langfuse span and
accumulates it into ``tasks.total_tokens_cached`` (VAL-CACHING-003,
VAL-CACHING-004).
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

# ── CACHING_ENABLED flag ──────────────────────────────────────────
# Controls whether cache_control markers are injected into prompts.
# Default: True (caching enabled).  Set to "false" to disable.
# When disabled, no cache_control markers appear in any request,
# but the text content of messages is identical to when enabled.

_caching_enabled: bool = os.getenv("CACHING_ENABLED", "true").lower() not in (
    "false",
    "0",
    "no",
    "off",
)


def get_caching_enabled() -> bool:
    """Return whether prompt caching is currently enabled."""
    return _caching_enabled


def set_caching_enabled(enabled: bool) -> None:
    """Set the caching enabled flag at runtime.

    This is useful for testing the semantic invariant that
    caching ON vs OFF produces identical output at temperature=0.
    """
    global _caching_enabled
    _caching_enabled = enabled


# ── Model family detection ─────────────────────────────────────────


def _is_anthropic_model(model: str) -> bool:
    """Return True if the model is an Anthropic-family model."""
    return model.startswith("anthropic/")


def _is_gemini_model(model: str) -> bool:
    """Return True if the model is a Google Gemini model."""
    return model.startswith("google/") or model.startswith("gemini/")


def _needs_cache_markers(model: str) -> bool:
    """Return True if the model family requires explicit cache_control markers.

    Anthropic and Gemini models need explicit ``cache_control`` breakpoints.
    DeepSeek and OpenAI models use auto/implicit caching and do NOT need
    markers in the request body (VAL-CACHING-002).
    """
    return _is_anthropic_model(model) or _is_gemini_model(model)


# ── Structured prompt ─────────────────────────────────────────────


class StructuredPrompt(BaseModel):
    """A prompt split into static (cacheable) and dynamic blocks.

    Static blocks contain system instructions + repo context and are
    tagged with cache markers per OpenRouter protocol so that repeated
    calls on the same repo benefit from prompt caching.

    Dynamic blocks contain the current edit and prior review result,
    which change on every call and are not cached.
    """

    static: str
    dynamic: str

    def to_messages(self, *, system_first: bool = True) -> list[dict[str, Any]]:
        """Convert to OpenRouter-compatible message list (no cache markers).

        Static content is placed first (system message) so that
        OpenRouter can cache it.  Dynamic content follows as the
        user message.

        This method does NOT inject cache markers.  Use
        ``to_messages_with_cache_markers`` for that.
        """
        messages: list[dict[str, Any]] = []
        if system_first:
            messages.append({"role": "system", "content": self.static})
            messages.append({"role": "user", "content": self.dynamic})
        else:
            messages.append({"role": "user", "content": self.dynamic})
            messages.append({"role": "system", "content": self.static})
        return messages

    def to_messages_with_cache_markers(
        self,
        *,
        model: str,
    ) -> list[dict[str, Any]]:
        """Convert to OpenRouter-compatible message list with cache markers.

        For Anthropic/Gemini models, the system message uses content-block
        format with ``cache_control: {"type": "ephemeral"}`` on the last
        static block (VAL-CACHING-002).

        For DeepSeek/OpenAI models, the system message uses simple string
        format (no cache markers needed — auto/implicit caching).

        The text content is identical regardless of whether cache markers
        are present, preserving the semantic invariant (VAL-CACHING-006).

        Args:
            model: The model identifier (e.g. ``deepseek/deepseek-chat-v3-0324``,
                   ``anthropic/claude-3.5-sonnet``).

        Returns:
            A list of message dicts compatible with the OpenRouter chat API.
        """
        messages: list[dict[str, Any]] = []

        if _needs_cache_markers(model) and get_caching_enabled():
            # Anthropic/Gemini: use content-block format with cache_control
            # on the last static content block
            system_content: list[dict[str, Any]] = [
                {"type": "text", "text": self.static},
            ]
            # Tag the last static block with cache_control
            system_content[-1]["cache_control"] = {"type": "ephemeral"}

            messages.append({"role": "system", "content": system_content})
        else:
            # DeepSeek/OpenAI: simple string format (auto-caches)
            messages.append({"role": "system", "content": self.static})

        messages.append({"role": "user", "content": self.dynamic})

        return messages


# ── Prompt builder ──────────────────────────────────────────────────


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

    The static block is the same across multiple calls on the same repo,
    making it eligible for OpenRouter prompt caching.  The dynamic block
    changes with each invocation (current edit state, review feedback).

    Args:
        system_instructions: The agent's system prompt (static across calls).
        repo_context: RAG-retrieved repo context (static across calls on same repo).
        current_edit: The current ChangePlan or edit context (dynamic).
        prior_review: Prior ReviewResult feedback (dynamic).

    Returns:
        A ``StructuredPrompt`` ready for cache-marker injection.
    """
    static = f"{system_instructions}\n\n---\nRepo context:\n{repo_context}"
    dynamic_parts: list[str] = []
    if current_edit:
        dynamic_parts.append(f"Current edit:\n{current_edit}")
    if prior_review:
        dynamic_parts.append(f"Prior review:\n{prior_review}")
    dynamic = "\n\n".join(dynamic_parts) if dynamic_parts else "No edit context yet."

    return StructuredPrompt(static=static, dynamic=dynamic)
