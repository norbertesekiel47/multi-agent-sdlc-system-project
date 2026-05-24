"""Secret redaction filter for Python loggers.

Masks any string matching known secret patterns (PAT tokens, API keys, etc.)
before the log record is emitted.  Installed on the root logger at startup
so ALL loggers benefit from the redaction.

Supported patterns:
  - ``github_pat_*``  (fine-grained PATs)
  - ``gho_*``         (OAuth tokens)
  - ``ghp_*``         (classic PATs)
  - ``ghs_*``         (SSH keys in some contexts)
  - ``sk-or-v1-*``    (OpenRouter keys)
  - ``sk-*``          (OpenAI-style keys)
  - ``hf_*``          (Hugging Face tokens)
"""

from __future__ import annotations

import logging
import re

# Secret prefixes and their regex patterns.
# Order matters: longer/more-specific prefixes MUST come before shorter ones
# so that e.g. "sk-or-v1-..." is matched before the generic "sk-..." pattern.
# The patterns are combined into a single alternation regex for one-pass
# replacement, which avoids the problem of a shorter pattern corrupting
# a longer pattern's replacement in a multi-pass approach.
_SECRET_PREFIXES: list[str] = [
    "github_pat_",
    "sk-or-v1-",
    "gho_",
    "ghp_",
    "ghs_",
    "sk-",
    "hf_",
]

# Build a single combined regex: (github_pat_[A-Za-z0-9_]+|sk-or-v1-[A-Za-z0-9_]+|...)
# Longer prefixes come first so the regex engine matches them greedily.
_COMBINED_PATTERN = re.compile(
    "|".join(re.escape(prefix) + r"[A-Za-z0-9_]+" for prefix in _SECRET_PREFIXES)
)


def _prefix_of_match(match_text: str) -> str:
    """Return the secret prefix of a matched string."""
    for prefix in _SECRET_PREFIXES:
        if match_text.startswith(prefix):
            return prefix
    return ""  # Should never happen if patterns are correct


class SecretRedactionFilter(logging.Filter):
    """Logging filter that redacts secret patterns from log messages.

    Usage::

        import logging
        from src.logging.secret_filter import SecretRedactionFilter, install_secret_filter

        # Option 1: Manual
        handler = logging.StreamHandler()
        handler.addFilter(SecretRedactionFilter())

        # Option 2: Install globally on root logger
        install_secret_filter()
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact secret patterns from the log record's message.

        Returns True (always allow the record through, just redact it).
        """
        original = record.getMessage()
        redacted = self._redact(original)
        if redacted != original:
            record.msg = redacted
            record.args = ()
        return True

    @staticmethod
    def _redact(text: str) -> str:
        """Replace any secret pattern in *text* with ``***REDACTED***``."""
        return _COMBINED_PATTERN.sub(
            lambda m: f"{_prefix_of_match(m.group())}***REDACTED***", text
        )


def install_secret_filter() -> None:
    """Install the SecretRedactionFilter on the root logger.

    Safe to call multiple times — idempotent (won't add duplicate filters).
    """
    root = logging.getLogger()
    # Avoid installing duplicates
    for existing in root.filters:
        if isinstance(existing, SecretRedactionFilter):
            return
    root.addFilter(SecretRedactionFilter())
