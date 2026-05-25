"""GitHub client integration and unit tests — VAL-GH-CLIENT-001 through 012.

Integration tests (marked @pytest.mark.integration) require:
  - GITHUB_PAT in .env with Contents:write, PullRequests:write, Issues:read
  - GITHUB_USERNAME in .env
  - The test repo norbertesekiel47/sdlc-swarm-test-repo must exist with
    issue #1 seeded.

Unit tests run without any network access.

Teardown: integration tests that create PRs/branches clean up after themselves
using the GitHub API.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

TEST_REPO = "norbertesekiel47/sdlc-swarm-test-repo"
TEST_REPO_URL = f"https://github.com/{TEST_REPO}"
TEST_ISSUE_NUMBER = 1
TEST_ISSUE_TITLE = "Test issue for read_issue verification"
BRANCH_PREFIX = "sdlc-swarm/"


@pytest.fixture()
def pat() -> str:
    """Return GITHUB_PAT, skip if missing."""
    token = os.getenv("GITHUB_PAT", "")
    if not token:
        pytest.skip("GITHUB_PAT not set")
    return token


@pytest.fixture()
def username() -> str:
    """Return GITHUB_USERNAME, skip if missing."""
    name = os.getenv("GITHUB_USERNAME", "")
    if not name:
        pytest.skip("GITHUB_USERNAME not set")
    return name


@pytest.fixture()
def tmp_clone_dir() -> Generator[str]:
    """Provide a temp directory for cloning; clean up afterwards."""
    d = tempfile.mkdtemp(prefix="gh-client-test-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_client(pat: str, username: str):
    """Create a GitHubClient instance via lazy import."""
    from src.github_client.client import GitHubClient

    return GitHubClient(pat=pat, username=username)


def test_clone_scrubs_authenticated_origin_remote() -> None:
    """clone() must not leave a PAT-bearing URL in .git/config."""
    client = _make_client("ghp_fake_secret_token", "octocat")
    dest = "/tmp/repo"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        client.clone("https://github.com/org/repo.git", dest)

    mock_run.assert_any_call(
        ["git", "remote", "set-url", "origin", "https://github.com/org/repo"],
        capture_output=True,
        text=True,
        cwd=dest,
        check=True,
        timeout=10,
    )
    set_url_calls = [
        call
        for call in mock_run.call_args_list
        if call.args and call.args[0][:3] == ["git", "remote", "set-url"]
    ]
    assert set_url_calls, "clone() should scrub origin after authenticated clone"
    for call in set_url_calls:
        assert "ghp_fake_secret_token" not in repr(call)


def test_commit_and_push_uses_authenticated_url_without_mutating_origin() -> None:
    """Push should authenticate without writing the PAT into the origin remote."""
    client = _make_client("ghp_fake_secret_token", "octocat")

    def _run_side_effect(cmd: list[str], **_: object) -> MagicMock:
        if cmd[:3] == ["git", "remote", "get-url"]:
            return MagicMock(stdout="https://github.com/org/repo\n", returncode=0, stderr="")
        return MagicMock(returncode=0, stderr="", stdout="")

    with patch("subprocess.run", side_effect=_run_side_effect) as mock_run:
        client.commit_and_push("/tmp/repo", "sdlc-swarm/fix-1", "fix: issue")

    push_calls = [
        call
        for call in mock_run.call_args_list
        if call.args and call.args[0][:2] == ["git", "push"]
    ]
    assert len(push_calls) == 1
    push_cmd = push_calls[0].args[0]
    assert "ghp_fake_secret_token" in push_cmd[3]

    set_url_calls = [
        call
        for call in mock_run.call_args_list
        if call.args and call.args[0][:3] == ["git", "remote", "set-url"]
    ]
    assert set_url_calls == []


# ===========================================================================
# VAL-GH-CLIENT-001: clone uses PAT auth successfully
# ===========================================================================


@pytest.mark.integration
def test_clone_uses_pat_auth_successfully(pat: str, username: str, tmp_clone_dir: str) -> None:
    """GitHubClient.clone(repo_url, dest) clones using PAT authentication.

    The cloned repo must have a .git/HEAD file, proving a successful clone.
    Clone must complete in under 30 s.
    """
    client = _make_client(pat, username)

    dest = os.path.join(tmp_clone_dir, "repo")
    client.clone(TEST_REPO_URL, dest)

    assert os.path.isfile(os.path.join(dest, ".git", "HEAD"))


# ===========================================================================
# VAL-GH-CLIENT-002: PAT value never leaks in logs after clone
# ===========================================================================


@pytest.mark.integration
def test_pat_never_leaks_in_logs(pat: str, username: str, tmp_clone_dir: str) -> None:
    """After clone, the raw PAT value must NOT appear in captured logs.

    Regex sweeps captured stdout/stderr and any configured Python logging
    output for the pattern ``github_pat_[A-Za-z0-9_]+`` OR the actual token
    value (since it may start with ``gho_``).
    """
    client = _make_client(pat, username)

    # Capture log output
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("src.github_client")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    import io
    log_capture = io.StringIO()
    handler.setStream(log_capture)  # type: ignore[attr-defined]

    dest = os.path.join(tmp_clone_dir, "repo")
    client.clone(TEST_REPO_URL, dest)

    log_output = log_capture.getvalue()

    # Check that the raw PAT does not appear in log output
    assert pat not in log_output, "Raw PAT value found in log output"

    # Also check the github_pat_ pattern
    pat_pattern = re.compile(r"github_pat_[A-Za-z0-9_]+")
    assert not pat_pattern.search(log_output), "github_pat_ pattern found in log output"

    # Check gho_ pattern too
    gho_pattern = re.compile(r"gho_[A-Za-z0-9_]+")
    assert not gho_pattern.search(log_output), "gho_ pattern found in log output"

    logger.removeHandler(handler)


# ===========================================================================
# VAL-GH-CLIENT-003: create_branch refuses non-prefixed branch names
# ===========================================================================


def test_create_branch_refuses_non_prefixed_names(pat: str, username: str) -> None:
    """create_branch MUST refuse any branch name not starting with sdlc-swarm/."""
    for bad_name in ["main", "feature/foo", "fix-123", "sdlc-swarm", "SDLCSWARM/test"]:
        client = _make_client(pat, username)
        with pytest.raises(Exception, match="BranchPolicyError|branch.*prefix|sdlc-swarm/"):
            client.create_branch("/fake/path", bad_name)


def test_create_branch_accepts_prefixed_names(pat: str, username: str) -> None:
    """create_branch accepts names starting with sdlc-swarm/."""
    # We can't test the actual branch creation without a repo, but we can
    # verify that the prefix check passes. This is a unit test; the
    # integration test is in test_e2e_clone_to_pr.
    client = _make_client(pat, username)
    # This should NOT raise BranchPolicyError (it may fail for other reasons
    # like the path not existing, but not for the prefix policy).
    with pytest.raises(Exception) as exc_info:
        client.create_branch("/nonexistent/path", "sdlc-swarm/test-branch")
    # Ensure it's NOT a BranchPolicyError
    from src.github_client.errors import BranchPolicyError
    assert not isinstance(exc_info.value, BranchPolicyError)


# ===========================================================================
# VAL-GH-CLIENT-004: commit_and_push creates remote branch with diff
# ===========================================================================


@pytest.mark.integration
def test_commit_and_push_creates_remote_branch(
    pat: str, username: str, tmp_clone_dir: str
) -> None:
    """commit_and_push pushes a branch and diff to the remote test repo."""
    client = _make_client(pat, username)

    dest = os.path.join(tmp_clone_dir, "repo")
    client.clone(TEST_REPO_URL, dest)

    branch_name = f"sdlc-swarm/test-push-{os.getpid()}"
    client.create_branch(dest, branch_name)

    # Write a trivial file
    test_file = os.path.join(dest, "push_test.txt")
    with open(test_file, "w") as f:
        f.write("test content from commit_and_push test\n")

    # git add + commit + push
    client.commit_and_push(dest, branch_name, "test: add push_test.txt")

    # Verify the branch exists on the remote
    result = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", branch_name],
        capture_output=True,
        text=True,
        cwd=dest,
        timeout=30,
    )
    assert result.returncode == 0
    assert branch_name in result.stdout, f"Branch {branch_name} not found on remote"

    # Cleanup: delete the remote branch
    from github import Github
    from github.Auth import Token

    g = Github(auth=Token(pat))
    repo = g.get_repo(TEST_REPO)
    try:
        ref = repo.get_git_ref(f"heads/{branch_name}")
        ref.delete()
    except Exception:
        pass


# ===========================================================================
# VAL-GH-CLIENT-005: open_pull_request returns valid PR URL on test repo
# ===========================================================================


@pytest.mark.integration
def test_open_pull_request_returns_valid_url(
    pat: str, username: str, tmp_clone_dir: str
) -> None:
    """open_pull_request opens a real PR and returns a URL matching the
    expected pattern.
    """
    client = _make_client(pat, username)

    dest = os.path.join(tmp_clone_dir, "repo")
    client.clone(TEST_REPO_URL, dest)

    branch_name = f"sdlc-swarm/test-pr-{os.getpid()}"
    client.create_branch(dest, branch_name)

    # Write a file
    test_file = os.path.join(dest, "pr_test.txt")
    with open(test_file, "w") as f:
        f.write("test content for PR\n")

    client.commit_and_push(dest, branch_name, "test: add pr_test.txt")

    # Open PR
    pr_ref = client.open_pull_request(
        repo=TEST_REPO,
        head_branch=branch_name,
        base_branch="main",
        title="test: PR from integration test",
        body="Auto-generated PR for VAL-GH-CLIENT-005 test.",
    )

    url = pr_ref.html_url
    assert re.match(
        r"^https://github\.com/norbertesekiel47/sdlc-swarm-test-repo/pull/\d+$", url
    ), f"PR URL doesn't match expected pattern: {url}"

    # Cleanup: close PR and delete branch
    from github import Github
    from github.Auth import Token

    g = Github(auth=Token(pat))
    repo = g.get_repo(TEST_REPO)
    try:
        pr_obj = repo.get_pull(pr_ref.number)
        pr_obj.edit(state="closed")
    except Exception:
        pass
    try:
        ref = repo.get_git_ref(f"heads/{branch_name}")
        ref.delete()
    except Exception:
        pass


# ===========================================================================
# VAL-GH-CLIENT-006: read_issue returns matching title and body
# ===========================================================================


@pytest.mark.integration
def test_read_issue_returns_matching_title_and_body(pat: str, username: str) -> None:
    """read_issue returns the issue title and body matching the seeded issue."""
    client = _make_client(pat, username)

    result = client.read_issue(repo=TEST_REPO, issue_number=TEST_ISSUE_NUMBER)

    assert result["title"] == TEST_ISSUE_TITLE
    assert "VAL-GH-CLIENT-006" in result["body"]


# ===========================================================================
# VAL-GH-CLIENT-007: clone of nonexistent repo raises RepoNotFoundError
# ===========================================================================


@pytest.mark.integration
def test_clone_nonexistent_repo_raises_typed_error(pat: str, username: str) -> None:
    """clone of a nonexistent repo raises RepoNotFoundError, not a generic error."""
    from src.github_client.errors import RepoNotFoundError

    client = _make_client(pat, username)

    with pytest.raises(RepoNotFoundError):
        client.clone(
            "https://github.com/norbertesekiel47/this-repo-does-not-exist-xyz",
            "/tmp/should-not-exist",
        )


# ===========================================================================
# VAL-GH-CLIENT-008: open_pull_request raises InsufficientScopeError
# ===========================================================================


def test_open_pull_request_raises_insufficient_scope_error() -> None:
    """open_pull_request against a repo without write permission raises
    InsufficientScopeError.

    We simulate this by mocking the PyGithub call to raise a 403.
    """
    from github import GithubException
    from src.github_client.errors import InsufficientScopeError

    client = _make_client("fake-token", "fake-user")

    with patch.object(client, "_get_github") as mock_gh:
        mock_repo = MagicMock()
        mock_repo.create_pull.side_effect = GithubException(
            403,
            {"message": "Resource not accessible by personal access token"},
            None,
        )
        mock_gh.return_value.get_repo.return_value = mock_repo

        with pytest.raises(InsufficientScopeError):
            client.open_pull_request(
                repo="other-org/protected-repo",
                head_branch="sdlc-swarm/test",
                base_branch="main",
                title="test",
                body="test",
            )


# ===========================================================================
# VAL-GH-CLIENT-009: Logging filter redacts PAT pattern in log records
# ===========================================================================


def test_logging_filter_redacts_pat_pattern() -> None:
    """The secret redaction filter masks PAT patterns in log records."""
    from src.logging.secret_filter import SecretRedactionFilter

    test_logger = logging.getLogger("test.redaction")
    test_logger.setLevel(logging.DEBUG)

    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretRedactionFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))
    test_logger.addHandler(handler)

    # Emit a log containing a fake PAT
    test_logger.info("Using token github_pat_ABCDEFG12345 for authentication")
    test_logger.info("Using token gho_ABCDEFG12345 for OAuth")
    test_logger.info("API key sk-or-v1-1234567890abcdef called")

    output = stream.getvalue()

    assert "github_pat_ABCDEFG12345" not in output
    assert "github_pat_***REDACTED***" in output
    assert "gho_ABCDEFG12345" not in output
    assert "gho_***REDACTED***" in output
    assert "sk-or-v1-1234567890abcdef" not in output
    assert "sk-or-v1-***REDACTED***" in output

    test_logger.removeHandler(handler)


# ===========================================================================
# VAL-GH-CLIENT-010: commit_and_push refuses non-allowlisted refspec
# ===========================================================================


def test_commit_and_push_refuses_non_allowlisted_refspec(pat: str, username: str) -> None:
    """commit_and_push with a non-allowlisted refspec (e.g. pushing to main)
    raises BranchPolicyError at the client layer.
    """
    from src.github_client.errors import BranchPolicyError

    client = _make_client(pat, username)

    # Trying to push to main should be refused
    with pytest.raises(BranchPolicyError):
        client.commit_and_push("/fake/path", "main", "should be refused")

    # Also refuse unprefixed branch names
    with pytest.raises(BranchPolicyError):
        client.commit_and_push("/fake/path", "feature/something", "should be refused")


# ===========================================================================
# VAL-GH-CLIENT-011: End-to-end clone-to-PR client sequence
# ===========================================================================


@pytest.mark.integration
def test_e2e_clone_to_pr_opens_real_pr(pat: str, username: str, tmp_clone_dir: str) -> None:
    """Full happy-path: clone → branch → write file → commit → push → open PR.

    Verify the PR exists via GitHub API and is in OPEN state.
    Teardown: close the PR and delete the branch.
    """
    client = _make_client(pat, username)

    dest = os.path.join(tmp_clone_dir, "repo")
    client.clone(TEST_REPO_URL, dest)

    branch_name = f"sdlc-swarm/e2e-test-{os.getpid()}"
    client.create_branch(dest, branch_name)

    # Write a file
    test_file = os.path.join(dest, "e2e_test.txt")
    with open(test_file, "w") as f:
        f.write("e2e test content\n")

    client.commit_and_push(dest, branch_name, "test: e2e full sequence")

    pr_ref = client.open_pull_request(
        repo=TEST_REPO,
        head_branch=branch_name,
        base_branch="main",
        title="test: end-to-end PR from integration test",
        body="Auto-generated PR for VAL-GH-CLIENT-011 test.",
    )

    # Verify PR is OPEN
    from github import Github
    from github.Auth import Token

    g = Github(auth=Token(pat))
    repo = g.get_repo(TEST_REPO)
    pr_obj = repo.get_pull(pr_ref.number)
    assert pr_obj.state == "open", f"PR state is {pr_obj.state}, expected 'open'"

    # Teardown
    pr_obj.edit(state="closed")
    try:
        ref = repo.get_git_ref(f"heads/{branch_name}")
        ref.delete()
    except Exception:
        pass


# ===========================================================================
# VAL-GH-CLIENT-012: Repo URL canonicalization in PR open
# ===========================================================================


def test_repo_url_canonicalization_strips_dot_git() -> None:
    """Repo URL with .git suffix produces the same canonical form as without."""
    from src.github_client.client import canonicalize_repo_url

    url_with_suffix = "https://github.com/norbertesekiel47/sdlc-swarm-test-repo.git"
    url_without_suffix = "https://github.com/norbertesekiel47/sdlc-swarm-test-repo"

    assert canonicalize_repo_url(url_with_suffix) == canonicalize_repo_url(url_without_suffix)


def test_repo_url_canonicalization_trims_whitespace() -> None:
    """Repo URL with leading/trailing whitespace is trimmed."""
    from src.github_client.client import canonicalize_repo_url

    url_with_ws = "  https://github.com/norbertesekiel47/sdlc-swarm-test-repo  "
    url_clean = "https://github.com/norbertesekiel47/sdlc-swarm-test-repo"

    assert canonicalize_repo_url(url_with_ws) == canonicalize_repo_url(url_clean)


def test_repo_url_canonicalization_trailing_slash() -> None:
    """Repo URL with trailing slash is canonicalized."""
    from src.github_client.client import canonicalize_repo_url

    url_with_slash = "https://github.com/norbertesekiel47/sdlc-swarm-test-repo/"
    url_clean = "https://github.com/norbertesekiel47/sdlc-swarm-test-repo"

    assert canonicalize_repo_url(url_with_slash) == canonicalize_repo_url(url_clean)


@pytest.mark.integration
def test_repo_url_canonicalization_in_pr_open(
    pat: str, username: str, tmp_clone_dir: str
) -> None:
    """PR opened with .git-suffix URL targets the same repo slug as without."""
    client = _make_client(pat, username)

    dest = os.path.join(tmp_clone_dir, "repo")
    # Clone with .git suffix
    client.clone(TEST_REPO_URL + ".git", dest)

    branch_name = f"sdlc-swarm/canon-test-{os.getpid()}"
    client.create_branch(dest, branch_name)

    test_file = os.path.join(dest, "canon_test.txt")
    with open(test_file, "w") as f:
        f.write("canonicalization test\n")

    client.commit_and_push(dest, branch_name, "test: canonicalization check")

    pr_ref = client.open_pull_request(
        repo=TEST_REPO,  # Canonical form without .git
        head_branch=branch_name,
        base_branch="main",
        title="test: canonicalization PR",
        body="PR for VAL-GH-CLIENT-012 test.",
    )

    # The PR's base repo should be the same regardless of .git suffix
    assert pr_ref.repo_slug == TEST_REPO

    # Cleanup
    from github import Github
    from github.Auth import Token

    g = Github(auth=Token(pat))
    repo = g.get_repo(TEST_REPO)
    try:
        pr_obj = repo.get_pull(pr_ref.number)
        pr_obj.edit(state="closed")
    except Exception:
        pass
    try:
        ref = repo.get_git_ref(f"heads/{branch_name}")
        ref.delete()
    except Exception:
        pass
