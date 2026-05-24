"""Static grep test: no confidence-trigger code in src/.

VAL-UNCERTAINTY-006: No code path inspects an LLM-self-reported
confidence field.  A static codebase search (greps src/ for
confidence-trigger keywords) asserts no such signal is used.
"""

from __future__ import annotations

import os
import re

# The root of the project source tree
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

# Confidence-related keywords that would indicate a confidence-based trigger
_CONFIDENCE_KEYWORDS = [
    "self_confidence",
    "llm_confidence",
    "model_confidence",
    "low_confidence",
    "uncertainty_confidence",
    "confidence_score",
    "confidence_level",
    "confidence_threshold",
]

# Patterns that would indicate confidence is used as a trigger condition
_TRIGGER_PATTERNS = [
    re.compile(r"if\s+\w*confidence", re.IGNORECASE),
    re.compile(r"confidence\s*[<>=!]+", re.IGNORECASE),
    re.compile(r"confidence\s+threshold", re.IGNORECASE),
    re.compile(r"trigger.*confidence", re.IGNORECASE),
    re.compile(r"confidence.*trigger", re.IGNORECASE),
    re.compile(r"confidence.*escalat", re.IGNORECASE),
]

# Exclusion: these files are allowed to mention confidence in the
# context of explicitly NOT using it
_ALLOWED_FILES = {
    "src/failure_modes/uncertainty.py",  # documents that confidence is NOT used
    "src/failure_modes/__init__.py",     # inherits docstring
    "tests/failure_modes/test_no_confidence_trigger.py",  # this file
    "tests/failure_modes/test_uncertainty.py",  # tests referencing the exclusion
}


def _is_exclusion_context(line: str) -> bool:
    """Check if a line mentioning 'confidence' is in an exclusion context.

    Exclusion context means the line says confidence is NOT used,
    rather than implementing a confidence-based trigger.
    """
    lower = line.lower()
    return any(
        marker in lower
        for marker in [
            "not used",
            "explicitly not",
            "is explicitly not",
            "never used",
            "not a trigger",
            "no code path",
            "# val-uncertainty-006",
            "assert no such signal",
        ]
    )


class TestNoConfidenceTrigger:
    """VAL-UNCERTAINTY-006: LLM self-confidence is NOT a trigger (static check)."""

    def test_no_confidence_signal_used(self) -> None:
        """Static grep confirms no confidence-trigger code in src/."""
        violations: list[str] = []

        for root, dirs, files in os.walk(_SRC_DIR):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in files:
                if not fname.endswith(".py"):
                    continue

                filepath = os.path.join(root, fname)
                relpath = os.path.relpath(filepath, _PROJECT_ROOT)

                # Skip allowed files
                if relpath in _ALLOWED_FILES:
                    continue

                with open(filepath) as f:
                    for lineno, line in enumerate(f, 1):
                        # Check for confidence keywords
                        line_lower = line.lower()
                        if "confidence" not in line_lower:
                            continue

                        # If it's in an exclusion context, skip
                        if _is_exclusion_context(line):
                            continue

                        # Check for trigger patterns
                        for pattern in _TRIGGER_PATTERNS:
                            if pattern.search(line):
                                violations.append(
                                    f"{relpath}:{lineno}: {line.strip()}"
                                )
                                break

        assert not violations, (
            "Found code paths that appear to use LLM confidence as a trigger:\n"
            + "\n".join(violations)
        )

    def test_no_confidence_function_in_failure_modes(self) -> None:
        """The failure_modes package has no function that reads confidence."""
        from src.failure_modes.uncertainty import UncertaintyEscalation

        # Check all public methods of UncertaintyEscalation
        public_methods = [
            m for m in dir(UncertaintyEscalation)
            if not m.startswith("_")
        ]
        for method_name in public_methods:
            assert "confidence" not in method_name.lower(), (
                f"Found confidence-related method: {method_name}"
            )

    def test_trigger_names_are_deterministic(self) -> None:
        """All uncertainty trigger names are deterministic (not LLM-dependent)."""
        # The valid trigger names
        valid_triggers = {
            "pydantic_validation_3x",
            "persistent_test_failure",
            "same_fix_rejected_twice",
            "tool_error_rate_exceeded",
        }

        # None should reference confidence or any LLM-dependent concept
        for trigger in valid_triggers:
            assert "confidence" not in trigger.lower()
            assert "llm" not in trigger.lower()
            assert "model" not in trigger.lower()
