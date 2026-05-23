"""Tests for project infrastructure — VAL-REPRO-001, VAL-REPRO-002, VAL-BACKEND-API-001."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.api.main import HealthResponse, app

REPO_ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────
# VAL-REPRO-001: Python packages pinned with upper bounds
# ──────────────────────────────────────────────


def _parse_pyproject_deps() -> list[tuple[str, str]]:
    """Extract (name, specifier) pairs for direct dependencies from pyproject.toml."""
    pyproject = REPO_ROOT / "pyproject.toml"
    text = pyproject.read_text()

    in_deps = False
    deps: list[tuple[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps:
            if stripped.startswith("[") or (not stripped and not in_deps):
                break
            if stripped.startswith('"') or stripped.startswith("'"):
                dep = stripped.strip('",').strip("'")
                match = re.match(r"^([A-Za-z0-9_.-]+(?:\[[^\]]+\])?)\s*(.*)$", dep)
                if match:
                    name, spec = match.group(1), match.group(2).strip()
                    if spec:
                        deps.append((name, spec))
    return deps


def test_pyproject_every_dep_has_upper_bound() -> None:
    """VAL-REPRO-001: every direct dep has >=X,<Y or ==X bounds (no floating deps)."""
    deps = _parse_pyproject_deps()
    assert len(deps) > 0, "No dependencies found in pyproject.toml"

    unpinned: list[str] = []
    for name, spec in deps:
        if re.match(r"^>=\d[^,]*,\s*<\d+", spec) or re.match(r"^==\d+", spec):
            continue
        unpinned.append(f"{name}: {spec}")

    assert unpinned == [], (
        f"Dependencies without upper bounds: {unpinned}. "
        "Every dep must use >=X,<Y or ==X pinning per VAL-REPRO-001."
    )


def test_pyproject_no_floating_deps() -> None:
    """VAL-REPRO-001: no dependency uses bare >=X or * without upper bound."""
    deps = _parse_pyproject_deps()
    for name, spec in deps:
        if re.match(r"^>=\d+$", spec):
            pytest.fail(f"Floating dep {name}: {spec} (missing upper bound)")
        if "*" in spec and not spec.startswith("=="):
            pytest.fail(f"Floating dep {name}: {spec}")


# ──────────────────────────────────────────────
# VAL-REPRO-002: Node packages locked by pnpm-lock.yaml
# ──────────────────────────────────────────────


def test_pnpm_workspace_yaml_exists() -> None:
    """pnpm-workspace.yaml must exist at repo root."""
    assert (REPO_ROOT / "pnpm-workspace.yaml").is_file(), "pnpm-workspace.yaml missing"


def test_pnpm_workspace_allows_required_builds() -> None:
    """pnpm-workspace.yaml must seed allowBuilds for sharp, unrs-resolver, msw."""
    text = (REPO_ROOT / "pnpm-workspace.yaml").read_text()
    for pkg in ("sharp", "unrs-resolver", "msw"):
        assert pkg in text, f"pnpm-workspace.yaml missing allowBuilds entry for {pkg}"


def test_pnpm_lock_yaml_exists() -> None:
    """VAL-REPRO-002: pnpm-lock.yaml must exist at workspace root."""
    assert (REPO_ROOT / "pnpm-lock.yaml").is_file(), "pnpm-lock.yaml missing"


# ──────────────────────────────────────────────
# VAL-BACKEND-API-001: GET /health returns 200 with structured payload
# ──────────────────────────────────────────────


def test_health_response_model() -> None:
    """HealthResponse model has all required fields."""
    resp = HealthResponse(status="ok", version="0.1.0", db="ok", langfuse="ok")
    assert resp.status == "ok"
    assert resp.version == "0.1.0"
    assert resp.db == "ok"
    assert resp.langfuse == "ok"


@pytest.mark.asyncio
async def test_health_endpoint_structure() -> None:
    """GET /health returns JSON with {status, version, db, langfuse} keys."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        body = resp.json()
        required_keys = {"status", "version", "db", "langfuse"}
        assert required_keys.issubset(body.keys()), (
            f"Missing keys: {required_keys - body.keys()}"
        )
        assert body["status"] in ("ok", "degraded")
        assert body["db"] in ("ok", "unreachable")
        assert body["langfuse"] in ("ok", "degraded", "unreachable")
        assert re.match(r"^\d+\.\d+\.\d+", body["version"])


# ──────────────────────────────────────────────
# Infrastructure file checks
# ──────────────────────────────────────────────


def test_docker_compose_exists() -> None:
    """infra/docker-compose.yml must exist."""
    assert (REPO_ROOT / "infra" / "docker-compose.yml").is_file(), (
        "infra/docker-compose.yml missing"
    )


def test_docker_compose_postgres_port_5433() -> None:
    """Postgres container must bind to host port 5433."""
    text = (REPO_ROOT / "infra" / "docker-compose.yml").read_text()
    assert "5433:5432" in text, "Postgres not bound to host port 5433"


def test_docker_compose_langfuse_port_3110() -> None:
    """Langfuse server must bind to host port 3110."""
    text = (REPO_ROOT / "infra" / "docker-compose.yml").read_text()
    assert "3110:3000" in text, "Langfuse not bound to host port 3110"


def test_docker_compose_no_forbidden_ports() -> None:
    """Docker compose must not expose forbidden ports (6379, 5432, 5000, 7000)."""
    text = (REPO_ROOT / "infra" / "docker-compose.yml").read_text()
    forbidden_host_ports = {"6379", "5432", "5000", "7000"}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-") and ":" in stripped:
            mapping = stripped.strip("-").strip().strip('"').strip("'")
            host_port = mapping.split(":")[0]
            assert host_port not in forbidden_host_ports, (
                f"Forbidden host port {host_port} in docker-compose.yml"
            )


def test_init_sh_exists_and_executable() -> None:
    """init.sh must exist and be executable."""
    init_sh = REPO_ROOT / "init.sh"
    assert init_sh.is_file(), "init.sh missing"
    assert os.access(init_sh, os.X_OK), "init.sh not executable"


def test_pyproject_toml_exists() -> None:
    """pyproject.toml must exist at repo root."""
    assert (REPO_ROOT / "pyproject.toml").is_file(), "pyproject.toml missing"


def test_gitignore_excludes_env() -> None:
    """.gitignore must exclude .env and secrets."""
    text = (REPO_ROOT / ".gitignore").read_text()
    assert ".env" in text, ".gitignore does not exclude .env"


def test_src_api_main_exists() -> None:
    """src/api/main.py must exist with FastAPI app."""
    assert (REPO_ROOT / "src" / "api" / "main.py").is_file(), "src/api/main.py missing"


def test_dotenv_loading_in_main() -> None:
    """src/api/main.py must load .env via python-dotenv."""
    text = (REPO_ROOT / "src" / "api" / "main.py").read_text()
    assert "load_dotenv" in text, "python-dotenv load_dotenv not called in src/api/main.py"
