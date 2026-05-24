"""Sandbox configuration constants and defaults."""

from __future__ import annotations

import os

# ── Resource limits ──────────────────────────────────────
SANDBOX_MEMORY_LIMIT: str = os.getenv("SANDBOX_MEMORY_LIMIT", "4g")
SANDBOX_CPU_LIMIT: float = float(os.getenv("SANDBOX_CPU_LIMIT", "2"))
SANDBOX_TIMEOUT_SECONDS: int = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "600"))
SANDBOX_USER_UID: int = int(os.getenv("SANDBOX_USER_UID", "1000"))

# ── Docker naming ────────────────────────────────────────
SANDBOX_CONTAINER_PREFIX: str = "sdlc-swarm-sandbox-"
SANDBOX_NETWORK_PREFIX: str = "sdlc-swarm-net-"
SANDBOX_PROXY_PREFIX: str = "sdlc-swarm-proxy-"

# ── Workspace ────────────────────────────────────────────
SANDBOX_WORKSPACE_ROOT: str = os.getenv(
    "SANDBOX_WORKSPACE_ROOT", os.path.expanduser("~/sdlc-swarm/work")
)
SANDBOX_REPO_MOUNT_POINT: str = "/workspace"

# ── Proxy ────────────────────────────────────────────────
PROXY_PORT: int = 3128
PROXY_IMAGE: str = "sdlc-swarm/sandbox-proxy:latest"

# ── Allowlist registries ─────────────────────────────────
DEFAULT_ALLOWLIST: list[str] = [
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    "crates.io",
    "huggingface.co",
]

# ── Base images for sandbox containers ───────────────────
DEFAULT_SANDBOX_IMAGE: str = os.getenv("SANDBOX_IMAGE", "python:3.12-slim")
NODE_SANDBOX_IMAGE: str = os.getenv("NODE_SANDBOX_IMAGE", "node:20-slim")

# ── Janitor ──────────────────────────────────────────────
JANITOR_MAX_AGE_SECONDS: int = int(os.getenv("JANITOR_MAX_AGE_SECONDS", "600"))
