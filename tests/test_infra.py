"""Tests for project infrastructure.

Covers: VAL-REPRO-001, VAL-REPRO-002, VAL-REPRO-003,
VAL-REPRO-004, VAL-BACKEND-API-001.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import yaml
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


def test_root_package_json_exposes_common_commands() -> None:
    """Root pnpm scripts must match documented repository commands."""
    package_json = REPO_ROOT / "package.json"
    assert package_json.is_file(), "package.json missing at repo root"
    data = json.loads(package_json.read_text())
    scripts = data.get("scripts", {})
    for script in ("dev", "build", "lint", "typecheck", "test", "test:e2e"):
        assert script in scripts, f"package.json missing {script} script"


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


# ──────────────────────────────────────────────
# VAL-REPRO-003: pinned-models.yaml model IDs match runtime configuration
# ──────────────────────────────────────────────


def _load_pinned_models() -> dict[str, str]:
    """Load agent → model_id mapping from pinned-models.yaml."""
    yaml_path = REPO_ROOT / "pinned-models.yaml"
    assert yaml_path.is_file(), "pinned-models.yaml missing"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    models: dict[str, str] = {}
    for role, info in data.get("models", {}).items():
        models[role] = info["model_id"]
    return models


def _get_runtime_model_ids() -> dict[str, str]:
    """Extract agent → model_id mapping from runtime source code."""
    runtime: dict[str, str] = {}

    # Planner
    planner_src = (REPO_ROOT / "src" / "agents" / "planner.py").read_text()
    match = re.search(r'^_PLANNER_MODEL:\s*str\s*=\s*"([^"]+)"', planner_src, re.MULTILINE)
    assert match, "Could not find _PLANNER_MODEL in planner.py"
    runtime["planner"] = match.group(1)

    # Coder
    coder_src = (REPO_ROOT / "src" / "agents" / "coder.py").read_text()
    match = re.search(r'^_CODER_MODEL:\s*str\s*=\s*"([^"]+)"', coder_src, re.MULTILINE)
    assert match, "Could not find _CODER_MODEL in coder.py"
    runtime["coder"] = match.group(1)

    # Reviewer
    reviewer_src = (REPO_ROOT / "src" / "agents" / "reviewer.py").read_text()
    match = re.search(r'^_REVIEWER_MODEL:\s*str\s*=\s*"([^"]+)"', reviewer_src, re.MULTILINE)
    assert match, "Could not find _REVIEWER_MODEL in reviewer.py"
    runtime["reviewer"] = match.group(1)

    # QA
    qa_src = (REPO_ROOT / "src" / "agents" / "qa.py").read_text()
    match = re.search(r'^_QA_MODEL:\s*str\s*=\s*"([^"]+)"', qa_src, re.MULTILINE)
    assert match, "Could not find _QA_MODEL in qa.py"
    runtime["qa"] = match.group(1)

    return runtime


def test_pinned_models_yaml_exists() -> None:
    """pinned-models.yaml must exist at repo root."""
    assert (REPO_ROOT / "pinned-models.yaml").is_file(), "pinned-models.yaml missing"


def test_pinned_models_match_runtime_planner() -> None:
    """VAL-REPRO-003: Planner model ID in pinned-models.yaml matches runtime."""
    pinned = _load_pinned_models()
    runtime = _get_runtime_model_ids()
    assert pinned["planner"] == runtime["planner"], (
        f"Planner model mismatch: pinned-models.yaml={pinned['planner']}, "
        f"runtime={runtime['planner']}"
    )


def test_pinned_models_match_runtime_coder() -> None:
    """VAL-REPRO-003: Coder model ID in pinned-models.yaml matches runtime."""
    pinned = _load_pinned_models()
    runtime = _get_runtime_model_ids()
    assert pinned["coder"] == runtime["coder"], (
        f"Coder model mismatch: pinned-models.yaml={pinned['coder']}, "
        f"runtime={runtime['coder']}"
    )


def test_pinned_models_match_runtime_reviewer() -> None:
    """VAL-REPRO-003: Reviewer model ID in pinned-models.yaml matches runtime."""
    pinned = _load_pinned_models()
    runtime = _get_runtime_model_ids()
    assert pinned["reviewer"] == runtime["reviewer"], (
        f"Reviewer model mismatch: pinned-models.yaml={pinned['reviewer']}, "
        f"runtime={runtime['reviewer']}"
    )


def test_pinned_models_match_runtime_qa() -> None:
    """VAL-REPRO-003: QA model ID in pinned-models.yaml matches runtime."""
    pinned = _load_pinned_models()
    runtime = _get_runtime_model_ids()
    assert pinned["qa"] == runtime["qa"], (
        f"QA model mismatch: pinned-models.yaml={pinned['qa']}, "
        f"runtime={runtime['qa']}"
    )


def test_pinned_models_all_roles_match() -> None:
    """VAL-REPRO-003: All agent roles in pinned-models.yaml match runtime config."""
    pinned = _load_pinned_models()
    runtime = _get_runtime_model_ids()
    for role in ("planner", "coder", "reviewer", "qa"):
        assert pinned[role] == runtime[role], (
            f"{role} model mismatch: pinned-models.yaml={pinned[role]}, "
            f"runtime={runtime[role]}"
        )


def test_pinned_models_configs_match() -> None:
    """VAL-REPRO-003: model_id in configs/*.yaml matches pinned-models.yaml."""
    pinned = _load_pinned_models()
    for role in ("planner", "coder", "reviewer", "qa"):
        config_path = REPO_ROOT / "configs" / f"{role}.yaml"
        if config_path.is_file():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            config_model = config.get("agent", {}).get("model_id", "")
            assert config_model == pinned[role], (
                f"configs/{role}.yaml model_id={config_model} != "
                f"pinned-models.yaml={pinned[role]}"
            )


# ──────────────────────────────────────────────
# VAL-REPRO-004: Default temperature is 0.2 for non-benchmark, 0 for benchmark
# ──────────────────────────────────────────────


def test_llm_client_default_temperature_is_0_2() -> None:
    """VAL-REPRO-004: LLMClient.chat() default temperature is 0.2."""
    import inspect

    from src.llm.client import LLMClient

    sig = inspect.signature(LLMClient.chat)
    temp_default = sig.parameters["temperature"].default
    assert temp_default == 0.2, (
        f"LLMClient.chat() default temperature is {temp_default}, expected 0.2"
    )


def test_llm_client_chat_with_cache_default_temperature_is_0_2() -> None:
    """VAL-REPRO-004: LLMClient.chat_with_cache() default temperature is 0.2."""
    import inspect

    from src.llm.client import LLMClient

    sig = inspect.signature(LLMClient.chat_with_cache)
    temp_default = sig.parameters["temperature"].default
    assert temp_default == 0.2, (
        f"LLMClient.chat_with_cache() default temperature is {temp_default}, expected 0.2"
    )


def test_get_temperature_default_is_0_2() -> None:
    """VAL-REPRO-004: get_temperature() returns 0.2 when LLM_TEMPERATURE is not set."""
    from src.llm.client import get_temperature

    # Save and clear
    original = os.environ.pop("LLM_TEMPERATURE", None)
    try:
        temp = get_temperature()
        assert temp == 0.2, f"Default temperature is {temp}, expected 0.2"
    finally:
        if original is not None:
            os.environ["LLM_TEMPERATURE"] = original


def test_get_temperature_benchmark_is_0() -> None:
    """VAL-REPRO-004: get_temperature() returns 0.0 when LLM_TEMPERATURE=0 (benchmark)."""
    from src.llm.client import get_temperature

    original = os.environ.get("LLM_TEMPERATURE")
    try:
        os.environ["LLM_TEMPERATURE"] = "0"
        temp = get_temperature()
        assert temp == 0.0, f"Benchmark temperature is {temp}, expected 0.0"
    finally:
        if original is not None:
            os.environ["LLM_TEMPERATURE"] = original
        else:
            os.environ.pop("LLM_TEMPERATURE", None)


def test_swebench_run_config_default_temperature_is_0() -> None:
    """VAL-REPRO-004: SWE-bench RunConfig default temperature is 0.0."""
    from src.benchmarks.swebench.models import RunConfig

    config = RunConfig()
    assert config.temperature == 0.0, (
        f"SWE-bench RunConfig default temperature is {config.temperature}, expected 0.0"
    )


def test_swebench_cli_default_temperature_is_0() -> None:
    """VAL-REPRO-004: SWE-bench CLI --temperature defaults to 0.0."""
    swebench_main = REPO_ROOT / "src" / "benchmarks" / "swebench" / "__main__.py"
    text = swebench_main.read_text()
    # The argparse default for --temperature should be 0.0
    assert '"--temperature"' in text or "'--temperature'" in text, (
        "--temperature argument not found in SWE-bench CLI"
    )
    # Check default=0.0 appears near --temperature
    match = re.search(r"--temperature.*?default\s*=\s*([\d.]+)", text)
    assert match, "Could not find --temperature default in SWE-bench CLI"
    assert float(match.group(1)) == 0.0, (
        f"SWE-bench CLI default temperature is {match.group(1)}, expected 0.0"
    )


def test_agents_use_get_temperature() -> None:
    """VAL-REPRO-004: Agent LLM calls use get_temperature(), not hardcoded 0.2."""
    for agent_file in ("coder.py", "reviewer.py", "qa.py"):
        src = (REPO_ROOT / "src" / "agents" / agent_file).read_text()
        # After wiring, agents should use get_temperature() instead of hardcoded 0.2
        # Check that the chat_with_cache call does NOT have temperature=0.2 hardcoded
        # It should use get_temperature() or a variable
        has_get_temperature = "get_temperature()" in src
        has_hardcoded = bool(
            re.search(r"temperature\s*=\s*0\.2\b", src)
        )
        assert has_get_temperature, (
            f"src/agents/{agent_file} does not import or use get_temperature()"
        )
        # If get_temperature is imported, hardcoded values are fine as fallback defaults
        # but the actual call should use get_temperature()
        if has_get_temperature and has_hardcoded:
            # Verify the actual LLM call uses get_temperature(), not 0.2
            # Pattern: temperature=get_temperature() in the LLM call
            assert re.search(r"temperature\s*=\s*get_temperature\(\)", src), (
                f"src/agents/{agent_file} imports get_temperature() but "
                f"LLM call still uses hardcoded 0.2"
            )
