"""GitHub client configuration constants."""

from __future__ import annotations

import os

# Branch name prefix — all branches created by the client must start with this.
BRANCH_PREFIX: str = os.getenv("GITHUB_BRANCH_PREFIX", "sdlc-swarm/")

# Default base branch for PRs.
DEFAULT_BASE_BRANCH: str = os.getenv("GITHUB_DEFAULT_BASE_BRANCH", "main")

# GitHub username for authenticated operations.
GITHUB_USERNAME: str = os.getenv("GITHUB_USERNAME", "")

# Timeout for git clone operations (seconds).
CLONE_TIMEOUT_SECONDS: int = int(os.getenv("GITHUB_CLONE_TIMEOUT", "30"))

# Timeout for git push operations (seconds).
PUSH_TIMEOUT_SECONDS: int = int(os.getenv("GITHUB_PUSH_TIMEOUT", "30"))
