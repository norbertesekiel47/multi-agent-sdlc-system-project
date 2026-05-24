"""Sandbox-specific exception types."""

from __future__ import annotations


class SandboxError(Exception):
    """Base exception for sandbox operations."""


class PathOutsideSandboxError(SandboxError):
    """Raised when a file operation targets a path outside the sandbox cwd."""

    def __init__(self, path: str, cwd: str) -> None:
        self.path = path
        self.cwd = cwd
        super().__init__(f"Path {path!r} is outside sandbox cwd {cwd!r}")


class SandboxTimeoutError(SandboxError):
    """Raised when a sandbox command exceeds the configured timeout."""

    def __init__(self, timeout_seconds: float, command: str = "") -> None:
        self.timeout_seconds = timeout_seconds
        self.command = command
        msg = f"Command timed out after {timeout_seconds}s"
        if command:
            msg += f": {command[:200]}"
        super().__init__(msg)


class SandboxToolError(SandboxError):
    """Raised when a sandbox tool execution fails."""


class SandboxNotRunningError(SandboxError):
    """Raised when trying to use a sandbox that is not running."""


class ProxyError(SandboxError):
    """Raised when the sidecar proxy cannot be started or configured."""
