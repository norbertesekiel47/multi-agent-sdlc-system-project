"""Hand-rolled GitHub client built on PyGithub.

Provides authenticated clone, branch creation with ``sdlc-swarm/`` prefix,
commit + push, PR open, and issue read.  All secret values (PAT, API keys)
are redacted from logs and error messages before emission.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from github import Github, GithubException
from github.Auth import Token

from src.github_client.config import (
    BRANCH_PREFIX,
    CLONE_TIMEOUT_SECONDS,
    DEFAULT_BASE_BRANCH,
    PUSH_TIMEOUT_SECONDS,
)
from src.github_client.errors import (
    BranchPolicyError,
    GitHubClientError,
    InsufficientScopeError,
    RepoNotFoundError,
)

logger = logging.getLogger(__name__)

# Regex patterns for detecting GitHub PATs and other secrets in log output.
# Order matters: longer/more-specific prefixes MUST come before shorter ones.
_SECRET_PREFIXES: list[str] = [
    "github_pat_",
    "sk-or-v1-",
    "gho_",
    "ghp_",
    "ghs_",
    "sk-",
    "hf_",
]

# Combined single-pass regex to avoid multi-pass corruption issues.
_COMBINED_SECRET_PATTERN = re.compile(
    "|".join(re.escape(prefix) + r"[A-Za-z0-9_]+" for prefix in _SECRET_PREFIXES)
)


def _prefix_of_match(match_text: str) -> str:
    """Return the secret prefix of a matched string."""
    for prefix in _SECRET_PREFIXES:
        if match_text.startswith(prefix):
            return prefix
    return ""  # Should never happen


def _redact(text: str) -> str:
    """Replace any secret pattern in *text* with ***REDACTED***."""
    return _COMBINED_SECRET_PATTERN.sub(
        lambda m: f"{_prefix_of_match(m.group())}***REDACTED***", text
    )


def canonicalize_repo_url(url: str) -> str:
    """Canonicalize a GitHub repo URL.

    - Strip ``.git`` suffix
    - Strip trailing slashes
    - Trim leading/trailing whitespace

    >>> canonicalize_repo_url("  https://github.com/org/repo.git  ")
    'https://github.com/org/repo'
    """
    url = url.strip()
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url


def _strip_url_credentials(url: str) -> str:
    """Remove userinfo from an HTTPS GitHub URL."""
    return re.sub(r"^(https?://)[^/@]+@", r"\1", url.strip())


def _extract_repo_slug(url: str) -> str:
    """Extract ``owner/repo`` from a GitHub URL.

    Accepts both full URLs and ``owner/repo`` slugs.
    """
    url = canonicalize_repo_url(url)
    # If it's already a slug, return it
    if "/" in url and not url.startswith("http"):
        return url
    # Extract from URL
    match = re.match(r"https?://github\.com/([^/]+/[^/]+)", url)
    if match:
        return match.group(1)
    raise GitHubClientError(f"Cannot extract repo slug from URL: {url!r}")


@dataclass
class PullRequestRef:
    """Reference to a created pull request."""

    number: int
    html_url: str
    repo_slug: str


class GitHubClient:
    """Hand-rolled GitHub client built on PyGithub.

    Usage::

        client = GitHubClient(pat=os.getenv("GITHUB_PAT"), username="norbertesekiel47")
        client.clone("https://github.com/org/repo", "/tmp/repo")
        client.create_branch("/tmp/repo", "sdlc-swarm/fix-issue")
        client.commit_and_push("/tmp/repo", "sdlc-swarm/fix-issue", "fix: ...")
        pr = client.open_pull_request("org/repo", "sdlc-swarm/fix-issue", "main", "Fix", "Body")
        issue = client.read_issue("org/repo", 42)
    """

    def __init__(self, pat: str, username: str = "") -> None:
        self._pat = pat
        self._username = username
        self._gh: Github | None = None

    def _get_github(self) -> Github:
        """Return a PyGithub instance (lazy-init)."""
        if self._gh is None:
            self._gh = Github(auth=Token(self._pat))
        return self._gh

    def _authenticated_url(self, repo_url: str) -> str:
        """Return a one-shot authenticated HTTPS URL for git transport."""
        canon_url = canonicalize_repo_url(_strip_url_credentials(repo_url))
        if not self._pat:
            return canon_url
        return canon_url.replace("https://", f"https://{self._pat}@")

    # ── Clone ──────────────────────────────────────────────

    def clone(self, repo_url: str, dest: str) -> None:
        """Clone a GitHub repo using PAT authentication.

        The PAT is embedded in the URL for authentication but never logged.
        Raises RepoNotFoundError if the repo does not exist (404).
        """
        canon_url = canonicalize_repo_url(repo_url)
        # Build authenticated URL (PAT in URL for this git invocation only).
        auth_url = self._authenticated_url(canon_url)

        logger.info(
            "Cloning repo %s to %s",
            _redact(canon_url),
            dest,
        )

        try:
            result = subprocess.run(
                ["git", "clone", auth_url, dest],
                capture_output=True,
                text=True,
                timeout=CLONE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitHubClientError(
                f"Clone timed out after {CLONE_TIMEOUT_SECONDS}s for {canon_url}"
            ) from exc

        if result.returncode != 0:
            stderr = result.stderr
            # Check for 404 / not found
            if "not found" in stderr.lower() or "404" in stderr:
                raise RepoNotFoundError(repo_url)
            # Redact any leaked PAT in the error message
            raise GitHubClientError(f"Clone failed: {_redact(stderr)}")

        # git clone persists the clone URL as origin. Scrub the PAT-bearing URL
        # immediately so .git/config and later diagnostics cannot leak it.
        subprocess.run(
            ["git", "remote", "set-url", "origin", canon_url],
            capture_output=True,
            text=True,
            cwd=dest,
            check=True,
            timeout=10,
        )

        logger.info("Cloned repo %s successfully", _redact(canon_url))

    # ── Branch ──────────────────────────────────────────────

    def create_branch(self, repo_path: str, branch_name: str) -> None:
        """Create a new branch in the local repo at *repo_path*.

        The branch name MUST start with the configured prefix (``sdlc-swarm/``).
        Raises BranchPolicyError if the prefix is missing.
        """
        self._check_branch_prefix(branch_name)

        subprocess.run(
            ["git", "branch", branch_name],
            capture_output=True,
            text=True,
            cwd=repo_path,
            check=True,
            timeout=10,
        )
        subprocess.run(
            ["git", "checkout", branch_name],
            capture_output=True,
            text=True,
            cwd=repo_path,
            check=True,
            timeout=10,
        )
        logger.info("Created and checked out branch %s", branch_name)

    # ── Commit + Push ──────────────────────────────────────

    def commit_and_push(
        self,
        repo_path: str,
        branch_name: str,
        commit_message: str,
    ) -> None:
        """Stage all changes, commit, and push *branch_name* to origin.

        Raises BranchPolicyError if the branch name does not start with
        the required prefix.
        """
        self._check_branch_prefix(branch_name)

        # Stage all changes
        subprocess.run(
            ["git", "add", "."],
            capture_output=True,
            text=True,
            cwd=repo_path,
            check=True,
            timeout=10,
        )

        # Commit
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            cwd=repo_path,
            check=True,
            timeout=10,
        )

        origin_result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            check=True,
            timeout=10,
        )
        push_url = self._authenticated_url(origin_result.stdout.strip())

        # Push using a one-shot URL so the PAT is never written into origin.
        try:
            result = subprocess.run(
                ["git", "push", "-u", push_url, branch_name],
                capture_output=True,
                text=True,
                cwd=repo_path,
                timeout=PUSH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitHubClientError(
                f"Push timed out after {PUSH_TIMEOUT_SECONDS}s for branch {branch_name}"
            ) from exc

        if result.returncode != 0:
            raise GitHubClientError(f"Push failed: {_redact(result.stderr)}")

        logger.info("Pushed branch %s to origin", branch_name)

    # ── PR open ────────────────────────────────────────────

    def open_pull_request(
        self,
        repo: str,
        head_branch: str,
        base_branch: str = DEFAULT_BASE_BRANCH,
        title: str = "",
        body: str = "",
    ) -> PullRequestRef:
        """Open a pull request via PyGithub.

        Returns a PullRequestRef with the PR number, URL, and repo slug.
        Raises InsufficientScopeError if the PAT lacks the required scope.
        """
        self._check_branch_prefix(head_branch)

        gh = self._get_github()
        try:
            gh_repo = gh.get_repo(repo)
        except GithubException as exc:
            if exc.status == 404:
                raise RepoNotFoundError(repo) from exc
            raise

        try:
            pr = gh_repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base_branch,
            )
        except GithubException as exc:
            if exc.status == 403:
                raise InsufficientScopeError(
                    missing_scope="pull_requests:write",
                    detail=str(exc.data) if exc.data else "",
                ) from exc
            raise

        logger.info("Opened PR #%d: %s", pr.number, pr.html_url)

        return PullRequestRef(
            number=pr.number,
            html_url=pr.html_url,
            repo_slug=repo,
        )

    # ── Issue read ─────────────────────────────────────────

    def read_issue(self, repo: str, issue_number: int) -> dict[str, Any]:
        """Read an issue's title and body from GitHub.

        Returns ``{"title": str, "body": str}``.
        """
        gh = self._get_github()
        try:
            gh_repo = gh.get_repo(repo)
            issue = gh_repo.get_issue(issue_number)
        except GithubException as exc:
            if exc.status == 404:
                raise RepoNotFoundError(repo) from exc
            raise

        return {"title": issue.title, "body": issue.body or ""}

    # ── Helpers ────────────────────────────────────────────

    def _check_branch_prefix(self, branch_name: str) -> None:
        """Enforce the branch naming policy: must start with BRANCH_PREFIX."""
        if not branch_name.startswith(BRANCH_PREFIX):
            raise BranchPolicyError(branch_name, BRANCH_PREFIX)
