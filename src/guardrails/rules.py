"""Declarative guardrail rules as typed Pydantic policies.

VAL-GUARDRAIL-010: Rules live in a single declarative module.
Adding a new rule requires only registering it in the rule list
and writing a matching unit test (no orchestrator changes).

Each rule is a Pydantic model with:
  - ``name``: unique identifier
  - ``description``: human-readable explanation
  - ``check(tool_name, command, sandbox_cwd, **kwargs)``:
    returns a ``GuardrailViolation`` or ``None``
"""

from __future__ import annotations

import abc
import re
from typing import Any

from pydantic import BaseModel

from src.guardrails.errors import GuardrailViolation
from src.sandbox.config import DEFAULT_ALLOWLIST, SANDBOX_REPO_MOUNT_POINT

# Allowlisted git refspec prefix (architecture §2.5, §4.3)
ALLOWLISTED_REF_PREFIX: str = "sdlc-swarm/"


# ── Abstract base for guardrail rules ──────────────────────────────


class GuardrailRuleBase(BaseModel, abc.ABC):
    """Abstract base class for guardrail rules.

    All rules must implement ``check()`` returning a
    ``GuardrailViolation`` or ``None``.
    """

    name: str = ""
    description: str = ""

    @abc.abstractmethod
    def check(
        self,
        tool_name: str,
        command: str,
        sandbox_cwd: str = SANDBOX_REPO_MOUNT_POINT,
        **kwargs: Any,
    ) -> GuardrailViolation | None:
        """Check if the command violates this rule."""
        ...


# ── Rule 1: Block rm -rf outside sandbox cwd ──────────────────────


class RmRfOutsideCwdRule(GuardrailRuleBase):
    """Block ``rm -rf`` on paths outside the sandbox cwd.

    VAL-GUARDRAIL-001: ``rm -rf`` targeting a host path is blocked.
    VAL-GUARDRAIL-002: ``rm -rf`` targeting a path inside /workspace is allowed.

    Matches ``rm -[rRfFdD]*`` patterns that reference absolute paths
    not starting with the sandbox cwd.
    """

    name: str = "rm_rf_outside_cwd"
    description: str = (
        "Block rm -rf on paths outside the sandbox cwd. "
        "rm -rf inside /workspace is allowed (VAL-GUARDRAIL-002)."
    )

    # Regex for rm with recursive/force flags
    _RM_RF_PATTERN: re.Pattern[str] = re.compile(
        r"\brm\s+(-[rRfFdD]+\s*|-[rRfD]+\s+)+", re.IGNORECASE
    )

    def check(
        self,
        tool_name: str,
        command: str,
        sandbox_cwd: str = SANDBOX_REPO_MOUNT_POINT,
        **kwargs: Any,
    ) -> GuardrailViolation | None:
        """Check if the command is an rm -rf targeting a path outside cwd."""
        # Only check commands that contain rm with recursive/force flags
        if not self._is_rm_rf_command(command):
            return None

        # Extract target paths from the command
        paths = self._extract_rm_targets(command)
        if not paths:
            # No explicit paths — relative paths are inside cwd, allowed
            return None

        # Check if any target path is outside the sandbox cwd
        cwd = sandbox_cwd.rstrip("/")
        for path in paths:
            stripped = path.strip()
            if not stripped:
                continue
            # Only check absolute paths (relative paths are inside cwd)
            if stripped.startswith("/"):
                resolved = stripped.rstrip("/")
                if not (resolved == cwd or resolved.startswith(cwd + "/")):
                    return GuardrailViolation(
                        rule_name=self.name,
                        tool_name=tool_name,
                        args_summary=f"rm -rf targeting {stripped}",
                        detail=(
                            f"Destructive 'rm -rf' targets path outside sandbox cwd: "
                            f"{stripped} (cwd={sandbox_cwd}). "
                            f"This operation is blocked to prevent host filesystem damage."
                        ),
                    )
        return None

    def _is_rm_rf_command(self, command: str) -> bool:
        """Check if the command is an rm with recursive/force flags."""
        # Match "rm" followed by flags containing r/R and optionally f/F
        # Patterns: rm -rf, rm -fr, rm -r -f, rm -Rf, rm -r, etc.
        pattern = re.compile(
            r"\brm\s+-[a-zA-Z]*[rRdD][a-zA-Z]*\s",
            re.IGNORECASE,
        )
        return bool(pattern.search(command))

    def _extract_rm_targets(self, command: str) -> list[str]:
        """Extract target paths from an rm command."""
        # Remove the rm command and flags, leaving the path arguments
        # Match: rm [flags] [paths]
        parts = command.split()
        paths: list[str] = []
        skip_flags = True
        for part in parts:
            if part == "rm":
                continue
            if skip_flags and part.startswith("-"):
                continue
            skip_flags = False
            # This is a path argument
            paths.append(part)
        return paths


# ── Rule 2: Block git push --force ────────────────────────────────


class GitPushForceRule(GuardrailRuleBase):
    """Block ``git push --force`` and variants.

    VAL-GUARDRAIL-003: Matches ``--force``, ``-f``, and ``--force-with-lease``.
    """

    name: str = "git_push_force"
    description: str = (
        "Block git push --force, -f, and --force-with-lease. "
        "Force pushes can destroy remote history (VAL-GUARDRAIL-003)."
    )

    # Regex for git push with force flags
    _FORCE_PATTERN: re.Pattern[str] = re.compile(
        r"\bgit\s+push\s+.*(--force(?!-with-lease)|\b-f\b|--force-with-lease)",
        re.IGNORECASE,
    )

    def check(
        self,
        tool_name: str,
        command: str,
        sandbox_cwd: str = SANDBOX_REPO_MOUNT_POINT,
        **kwargs: Any,
    ) -> GuardrailViolation | None:
        """Check if the command is a force push."""
        if not self._is_force_push(command):
            return None

        return GuardrailViolation(
            rule_name=self.name,
            tool_name=tool_name,
            args_summary="git push with force flag",
            detail=(
                "Force push detected (--force, -f, or --force-with-lease). "
                "Force pushes can irreversibly destroy remote history and "
                "are blocked by the guardrail policy."
            ),
        )

    def _is_force_push(self, command: str) -> bool:
        """Check if the command is a force push."""
        # Must start with git push
        if not re.search(r"\bgit\s+push\b", command, re.IGNORECASE):
            return False

        # Check for force flags
        parts = command.split()
        for i, part in enumerate(parts):
            if part == "git" and i + 1 < len(parts) and parts[i + 1] == "push":
                # Look at remaining parts for force flags
                remaining = " ".join(parts[i + 2:])
                if re.search(r"(?:^|\s)--force(?:\s|$|/)", remaining) or \
                   re.search(r"(?:^|\s)-f(?:\s|$)", remaining) or \
                   "--force-with-lease" in remaining:
                    return True
        return False


# ── Rule 3: Block git push to non-allowlisted refspecs ────────────


class GitPushRefspecRule(GuardrailRuleBase):
    """Block ``git push`` to non-allowlisted refspecs.

    VAL-GUARDRAIL-004: Only pushes targeting ``sdlc-swarm/*`` refspecs
    are allowed.  Pushes to ``main``, ``develop``, or any other branch
    without the required prefix are blocked.
    """

    name: str = "git_push_refspec"
    description: str = (
        "Block git push to non-allowlisted refspecs. "
        "Only pushes targeting sdlc-swarm/* branches are allowed "
        "(VAL-GUARDRAIL-004)."
    )

    def check(
        self,
        tool_name: str,
        command: str,
        sandbox_cwd: str = SANDBOX_REPO_MOUNT_POINT,
        **kwargs: Any,
    ) -> GuardrailViolation | None:
        """Check if the push targets a non-allowlisted refspec."""
        if not re.search(r"\bgit\s+push\b", command, re.IGNORECASE):
            return None

        # Skip if this is a force push (handled by GitPushForceRule)
        if re.search(r"--force|-f\b|--force-with-lease", command, re.IGNORECASE):
            return None

        refspecs = self._extract_push_refspecs(command)
        if not refspecs:
            # No explicit refspec — defensive: block bare "git push origin"
            # because the current branch may not be allowlisted
            # Check if there's a remote but no branch
            parts = command.strip().split()
            if len(parts) > 2:
                # "git push origin" with no refspec — blocked
                # Check if there are extra args but none look like a
                # refspec with the allowlisted prefix
                after_push = [p for p in parts[2:] if not p.startswith("-")]
                if after_push:
                    # Check if any arg is an allowlisted refspec
                    allowlisted = any(
                        self._is_allowlisted_ref(arg) for arg in after_push
                    )
                    if not allowlisted:
                        return GuardrailViolation(
                            rule_name=self.name,
                            tool_name=tool_name,
                            args_summary="git push without allowlisted refspec",
                            detail=(
                                "git push without an explicit allowlisted refspec. "
                                f"Only refspecs matching '{ALLOWLISTED_REF_PREFIX}*' "
                                "are allowed (e.g., 'sdlc-swarm/fix-issue-1'). "
                                "This prevents accidental pushes to protected branches."
                            ),
                        )
            return None

        # Check each refspec
        for refspec in refspecs:
            if not self._is_allowlisted_ref(refspec):
                return GuardrailViolation(
                    rule_name=self.name,
                    tool_name=tool_name,
                    args_summary=f"git push to non-allowlisted refspec: {refspec}",
                    detail=(
                        f"git push targets non-allowlisted refspec '{refspec}'. "
                        f"Only refspecs matching '{ALLOWLISTED_REF_PREFIX}*' "
                        "are allowed (e.g., 'sdlc-swarm/fix-issue-1')."
                    ),
                )
        return None

    def _extract_push_refspecs(self, command: str) -> list[str]:
        """Extract refspec arguments from a git push command."""
        parts = command.split()
        refspecs: list[str] = []
        found_push = False
        for part in parts:
            if part == "push":
                found_push = True
                continue
            if not found_push:
                continue
            if part.startswith("-"):
                continue
            # This could be a remote name or a refspec
            # Refspecs contain ':' or are branch names after the remote
            if ":" in part:
                # Full refspec like "main:refs/heads/main" or
                # "HEAD:refs/heads/sdlc-swarm/fix-1"
                refspecs.append(part)
            elif not part.startswith("origin") and not part.startswith("upstream"):
                # Assume it's a branch name if it doesn't look like a remote
                # But we need to handle "git push origin branch-name"
                # where "origin" is the remote and "branch-name" is the refspec
                pass
        return refspecs

    def _is_allowlisted_ref(self, refspec: str) -> bool:
        """Check if a refspec is allowlisted (matches sdlc-swarm/*)."""
        # Extract the destination branch from the refspec
        if ":" in refspec:
            # "src:dst" format — check the dst part
            _, dst = refspec.split(":", 1)
            branch = dst.replace("refs/heads/", "")
        else:
            branch = refspec.replace("refs/heads/", "")

        return branch.startswith(ALLOWLISTED_REF_PREFIX)


# ── Rule 4: Block subprocess commands containing secret values ────


class SecretLeakRule(GuardrailRuleBase):
    """Block subprocess commands containing secret values from env.

    VAL-GUARDRAIL-005: The match is on the exact secret value loaded
    from env at process start, not just the variable name.
    """

    name: str = "secret_leak"
    description: str = (
        "Block subprocess commands containing secret values from env. "
        "Matches exact secret values (not variable names) to prevent "
        "credential exfiltration (VAL-GUARDRAIL-005)."
    )

    # Secret values loaded at init time
    secret_values: list[str] = []

    def check(
        self,
        tool_name: str,
        command: str,
        sandbox_cwd: str = SANDBOX_REPO_MOUNT_POINT,
        **kwargs: Any,
    ) -> GuardrailViolation | None:
        """Check if the command contains any known secret value."""
        for secret in self.secret_values:
            if not secret:
                continue
            if secret in command:
                return GuardrailViolation(
                    rule_name=self.name,
                    tool_name=tool_name,
                    args_summary="command contains secret value (redacted)",
                    detail=(
                        "Command contains a secret value from the environment. "
                        "Exfiltrating credentials through subprocess commands "
                        "is blocked by the guardrail policy."
                    ),
                )
        return None


# ── Rule 5: Block HTTP requests to non-allowlisted hosts ──────────


class HttpHostAllowlistRule(GuardrailRuleBase):
    """Block HTTP requests to non-allowlisted hosts at the application layer.

    VAL-GUARDRAIL-006: Defense-in-depth — even if the network-layer
    proxy were misconfigured, the guardrail blocks non-allowlisted
    egress at the application layer.
    """

    name: str = "http_host_allowlist"
    description: str = (
        "Block HTTP requests to non-allowlisted hosts at the application layer. "
        "Defense-in-depth: even if the proxy is misconfigured, the guardrail "
        "blocks non-allowlisted egress (VAL-GUARDRAIL-006)."
    )

    # Allowlisted hosts (loaded from sandbox config)
    allowed_hosts: list[str] = list(DEFAULT_ALLOWLIST)

    # Patterns that indicate an HTTP request
    _HTTP_URL_PATTERN: re.Pattern[str] = re.compile(
        r"https?://([^/\s:\"']+)",
        re.IGNORECASE,
    )

    def check(
        self,
        tool_name: str,
        command: str,
        sandbox_cwd: str = SANDBOX_REPO_MOUNT_POINT,
        **kwargs: Any,
    ) -> GuardrailViolation | None:
        """Check if the command makes an HTTP request to a non-allowlisted host."""
        urls = self._HTTP_URL_PATTERN.findall(command)
        if not urls:
            return None

        for host in urls:
            # Remove port number if present
            hostname = host.split(":")[0]
            if not self._is_allowlisted_host(hostname):
                return GuardrailViolation(
                    rule_name=self.name,
                    tool_name=tool_name,
                    args_summary=f"HTTP request to non-allowlisted host: {hostname}",
                    detail=(
                        f"HTTP request to non-allowlisted host '{hostname}'. "
                        "Only allowlisted registry hosts are permitted: "
                        f"{', '.join(self.allowed_hosts)}. "
                        "This is a defense-in-depth check at the application layer."
                    ),
                )
        return None

    def _is_allowlisted_host(self, hostname: str) -> bool:
        """Check if a hostname is in the allowlist.

        Supports subdomain matching: if 'pypi.org' is allowlisted,
        then 'anything.pypi.org' is also allowed.
        """
        hostname_lower = hostname.lower()
        for allowed in self.allowed_hosts:
            allowed_lower = allowed.lower()
            if hostname_lower == allowed_lower:
                return True
            # Subdomain matching: files.pythonhosted.org matches pythonhosted.org
            if hostname_lower.endswith("." + allowed_lower):
                return True
        return False


# ── Rule Registry ───────────────────────────────────────────────────


def get_all_rules() -> list[type[GuardrailRuleBase]]:
    """Return all registered guardrail rule classes.

    VAL-GUARDRAIL-010: Adding a new rule requires only adding it
    to this list and writing a matching unit test.
    """
    return [
        RmRfOutsideCwdRule,
        GitPushForceRule,
        GitPushRefspecRule,
        SecretLeakRule,
        HttpHostAllowlistRule,
    ]


def create_default_rules(
    *,
    secret_values: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
) -> list[GuardrailRuleBase]:
    """Create instances of all rules with the given configuration.

    This is the factory used by the GuardrailMiddleware to build
    its rule list.
    """
    rules: list[GuardrailRuleBase] = [
        RmRfOutsideCwdRule(),
        GitPushForceRule(),
        GitPushRefspecRule(),
        SecretLeakRule(secret_values=secret_values or []),
        HttpHostAllowlistRule(allowed_hosts=allowed_hosts or list(DEFAULT_ALLOWLIST)),
    ]
    return rules
