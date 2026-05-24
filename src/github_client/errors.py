"""GitHub client specific exception types."""

from __future__ import annotations


class GitHubClientError(Exception):
    """Base exception for GitHub client operations."""


class RepoNotFoundError(GitHubClientError):
    """Raised when the target repository does not exist (HTTP 404)."""

    def __init__(self, repo_url: str) -> None:
        self.repo_url = repo_url
        super().__init__(f"Repository not found: {repo_url}")


class InsufficientScopeError(GitHubClientError):
    """Raised when the PAT lacks a required scope (HTTP 403).

    ``missing_scope`` names the scope that was absent.
    """

    def __init__(self, missing_scope: str, detail: str = "") -> None:
        self.missing_scope = missing_scope
        self.detail = detail
        msg = f"Insufficient scope: {missing_scope}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


class BranchPolicyError(GitHubClientError):
    """Raised when a branch name violates the naming policy.

    The default policy requires the ``sdlc-swarm/`` prefix.
    """

    def __init__(self, branch_name: str, prefix: str = "sdlc-swarm/") -> None:
        self.branch_name = branch_name
        self.prefix = prefix
        super().__init__(
            f"Branch name {branch_name!r} does not start with required prefix {prefix!r}"
        )
