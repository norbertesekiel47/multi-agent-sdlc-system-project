"""Sandbox isolation and tool tests.

VAL-SANDBOX-ISO-001 through VAL-SANDBOX-ISO-017 + VAL-CROSS-017.

These tests spin up REAL Docker containers and verify actual network
isolation, resource limits, and teardown behavior. Docker mocks are
NOT used for isolation tests per the testing strategy in architecture.md.

Tests are marked with @pytest.mark.sandbox_isolation to allow
selective runs.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from src.sandbox.config import (
    DEFAULT_ALLOWLIST,
    PROXY_PORT,
    SANDBOX_CONTAINER_PREFIX,
    SANDBOX_NETWORK_PREFIX,
    SANDBOX_PROXY_PREFIX,
    SANDBOX_REPO_MOUNT_POINT,
)
from src.sandbox.errors import (
    PathOutsideSandboxError,
    SandboxTimeoutError,
    SandboxToolError,
)
from src.sandbox.manager import SandboxManager

# ──────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────

# Use a shorter timeout for tests to keep them fast
TEST_TIMEOUT = 30
TEST_MEMORY = "4g"
TEST_CPU = 2.0
TEST_UID = 1000


def _unique_task_id() -> str:
    """Generate a unique task ID for test isolation."""
    return f"test-{uuid.uuid4().hex[:12]}"


def _docker_ps_containers(task_id: str) -> list[str]:
    """Return list of container names matching the task_id."""
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=sdlc-swarm-sandbox-{task_id}",
         "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    return result.stdout.strip().splitlines() if result.stdout.strip() else []


def _docker_ps_all_sdlc() -> list[str]:
    """Return all sdlc-swarm container names."""
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=sdlc-swarm-",
         "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    return result.stdout.strip().splitlines() if result.stdout.strip() else []


def _docker_network_ls(task_id: str) -> list[str]:
    """Return networks matching the task_id."""
    result = subprocess.run(
        ["docker", "network", "ls", "--filter", f"name=sdlc-swarm-net-{task_id}",
         "--format", "{{.Name}}"],
        capture_output=True, text=True,
    )
    return result.stdout.strip().splitlines() if result.stdout.strip() else []


def _docker_inspect(task_id: str) -> dict:
    """Return docker inspect for the sandbox container."""
    container_name = f"{SANDBOX_CONTAINER_PREFIX}{task_id}"
    result = subprocess.run(
        ["docker", "inspect", container_name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    import json
    data = json.loads(result.stdout)
    return data[0] if data else {}


def _docker_network_inspect(task_id: str) -> dict:
    """Return docker network inspect for the per-task network."""
    network_name = f"{SANDBOX_NETWORK_PREFIX}{task_id}"
    result = subprocess.run(
        ["docker", "network", "inspect", network_name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    import json
    data = json.loads(result.stdout)
    return data[0] if data else {}


def _cleanup_task(task_id: str) -> None:
    """Force cleanup any leftover Docker resources for a task."""
    for name in [
        f"{SANDBOX_CONTAINER_PREFIX}{task_id}",
        f"{SANDBOX_PROXY_PREFIX}{task_id}",
    ]:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    net_name = f"{SANDBOX_NETWORK_PREFIX}{task_id}"
    subprocess.run(["docker", "network", "rm", net_name], capture_output=True)
    ws_dir = Path(f"/var/sdlc-swarm/work/{task_id}")
    if ws_dir.exists():
        shutil.rmtree(ws_dir, ignore_errors=True)


@pytest.fixture
async def sandbox():
    """Provide a fresh sandbox that is cleaned up after the test."""
    task_id = _unique_task_id()
    sb = SandboxManager(
        task_id=task_id,
        memory_limit=TEST_MEMORY,
        cpu_limit=TEST_CPU,
        timeout_seconds=TEST_TIMEOUT,
        user_uid=TEST_UID,
    )
    try:
        await sb.setup()
        # Give containers a moment to fully start
        await asyncio.sleep(2)
        yield sb
    finally:
        await sb.teardown()
        # Ensure cleanup even if teardown failed
        _cleanup_task(task_id)


@pytest.fixture
def task_id_cleanup():
    """Provide a unique task_id and clean up after the test."""
    task_id = _unique_task_id()
    yield task_id
    _cleanup_task(task_id)


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-001: Sandbox blocks HTTP egress to non-allowlisted host
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_blocks_http_egress_to_non_allowlisted_host(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-001: HTTP egress to non-allowlisted host MUST fail.

    Uses Python's urllib (available in the default Python image) instead of
    curl, since python:3.12-slim doesn't include curl.
    """
    try:
        output = await sandbox.run_command(
            "python3 -c \"import urllib.request; "
            "r = urllib.request.urlopen('http://example.com', timeout=5); "
            "print(r.status)\"",
            timeout=15,
        )
        # If we get here, the request succeeded — that's a failure
        pytest.fail(f"HTTP request to example.com succeeded (status: {output.strip()})")
    except SandboxToolError:
        # Non-zero exit — expected (network unreachable / proxy denied)
        pass


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-002: Sandbox blocks HTTPS egress to non-allowlisted host
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_blocks_https_egress_to_non_allowlisted_host(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-002: HTTPS egress to non-allowlisted host MUST fail."""
    try:
        output = await sandbox.run_command(
            "python3 -c \"import urllib.request; "
            "r = urllib.request.urlopen('https://example.com', timeout=5); "
            "print(r.status)\"",
            timeout=15,
        )
        pytest.fail(f"HTTPS request to example.com succeeded (status: {output.strip()})")
    except SandboxToolError:
        # Non-zero exit — expected
        pass


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-003: Sandbox allows pip install from allowlisted PyPI
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_allows_pip_install_from_allowlisted_pypi(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-003: pip install requests must succeed via proxy."""
    output = await sandbox.run_command(
        "pip install --no-cache-dir requests==2.32.3",
        timeout=60,
    )
    assert "Successfully installed" in output or "Requirement already satisfied" in output, (
        f"pip install failed. Output: {output[:500]}"
    )


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-004: Sandbox allows npm install from allowlisted registry
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_allows_npm_install_from_allowlisted_registry(task_id_cleanup: str) -> None:
    """VAL-SANDBOX-ISO-004: npm install from registry.npmjs.org must succeed.

    This test uses a Node.js image since the default Python image doesn't have npm.
    """
    task_id = task_id_cleanup
    sb = SandboxManager(
        task_id=task_id,
        image="node:20-slim",
        memory_limit=TEST_MEMORY,
        cpu_limit=TEST_CPU,
        timeout_seconds=120,
        user_uid=0,  # Node images need root for npm global installs
    )
    try:
        await sb.setup()
        await asyncio.sleep(2)
        await sb.run_command("npm init -y", timeout=30)
        _output = await sb.run_command(
            "npm install --omit=dev left-pad@1.3.0",
            timeout=60,
        )
        # Check that the package was installed
        check = await sb.run_command("ls node_modules/left-pad/package.json", timeout=10)
        assert check.strip(), "npm install left-pad did not produce node_modules"
    finally:
        await sb.teardown()


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-005: Sandbox DNS lookups for non-allowlisted hosts fail
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_dns_lookups_for_non_allowlisted_hosts_fail(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-005: DNS lookup for non-allowlisted host MUST fail."""
    # Try using Python to resolve DNS (more reliable than getent in slim images)
    try:
        output = await sandbox.run_command(
            "python3 -c \"import socket; print(socket.gethostbyname('evil-host.invalid'))\"",
            timeout=10,
        )
        pytest.fail(f"DNS resolved non-allowlisted host unexpectedly: {output}")
    except SandboxToolError:
        # Expected: socket.gaierror causes non-zero exit
        pass


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-006: Sandbox mounts only the per-task workspace directory
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_mounts_only_workspace_directory(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-006: docker inspect must show only the workspace bind mount."""
    inspect_data = _docker_inspect(sandbox.task_id)
    assert inspect_data, "Container not found for inspection"

    # Docker inspect stores mount info in top-level "Mounts" and/or HostConfig.Mounts
    mounts = inspect_data.get("Mounts", [])
    if not mounts:
        # Fallback to HostConfig.Mounts
        mounts = inspect_data.get("HostConfig", {}).get("Mounts", [])
    assert len(mounts) >= 1, f"Expected at least 1 mount, found {len(mounts)}"

    # Check each mount — must not mount host filesystem paths outside workspace
    forbidden_source_prefixes = ["/etc", "/var/run/docker.sock", "/Users/norbertesekiel"]
    for mount in mounts:
        source = mount.get("Source", "")
        dest = (
            mount.get("Destination", "")
            or mount.get("Target", "")
        )

        # The workspace mount must be present
        if dest == SANDBOX_REPO_MOUNT_POINT:
            assert sandbox.task_id in source, (
                f"Workspace mount source {source!r} "
                f"doesn't match task_id {sandbox.task_id}"
            )
            continue

        # Any other mount must not be a forbidden host filesystem path
        for prefix in forbidden_source_prefixes:
            if source.startswith(prefix) and dest != SANDBOX_REPO_MOUNT_POINT:
                # Allow the workspace mount
                if sandbox.task_id in source:
                    continue
                pytest.fail(f"Forbidden mount: {source} -> {dest}")

    # Verify /workspace is mounted
    workspace_mounts = [
        m for m in mounts
        if (m.get("Destination") or m.get("Target", ""))
        == SANDBOX_REPO_MOUNT_POINT
    ]
    assert len(workspace_mounts) >= 1, (
        f"No mount for {SANDBOX_REPO_MOUNT_POINT} found. "
        f"Mounts: {mounts}"
    )

    # Explicitly check that /etc, /Users, docker.sock are NOT mounted
    for mount in mounts:
        dest = mount.get("Destination") or mount.get("Target", "")
        assert dest not in ("/etc", "/Users", "/var/run/docker.sock"), (
            f"Forbidden mount destination: {dest}"
        )


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-007: Sandbox enforces 4 GiB memory and 2 CPU limits
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_enforces_memory_and_cpu_limits(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-007: docker inspect shows --memory=4g and --cpus=2."""
    inspect_data = _docker_inspect(sandbox.task_id)
    assert inspect_data, "Container not found for inspection"

    host_config = inspect_data.get("HostConfig", {})

    # Memory: 4 GiB = 4 * 1024 * 1024 * 1024 = 4294967296
    memory = host_config.get("Memory", 0)
    expected_memory = 4 * 1024 * 1024 * 1024
    assert memory == expected_memory, (
        f"Memory limit {memory} != expected {expected_memory} (4 GiB)"
    )

    # CPUs: 2 * 1_000_000_000 nanocpus
    nano_cpus = host_config.get("NanoCpus", 0)
    expected_cpus = 2 * 1_000_000_000
    assert nano_cpus == expected_cpus, (
        f"CPU limit {nano_cpus} != expected {expected_cpus} (2 CPUs)"
    )


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-008: Sandbox attached to internal Docker network only
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_attached_to_internal_docker_network(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-008: network must have Internal=true, no bridge."""
    net_data = _docker_network_inspect(sandbox.task_id)
    assert net_data, "Network not found for inspection"

    # Check Internal flag
    assert net_data.get("Internal") is True, (
        f"Network {sandbox.network_name} is not internal: Internal={net_data.get('Internal')}"
    )

    # Container must not be on bridge
    inspect_data = _docker_inspect(sandbox.task_id)
    assert inspect_data, "Container not found"
    networks = inspect_data.get("NetworkSettings", {}).get("Networks", {})
    assert "bridge" not in networks, "Container is on bridge network (not isolated)"


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-009: Sandbox HTTP_PROXY env routes egress through sidecar
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_http_proxy_env_set(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-009: container must have HTTP_PROXY and HTTPS_PROXY env vars."""
    output = await sandbox.run_command("env | grep -i proxy", timeout=10)
    lower = output.lower()

    assert "http_proxy" in lower, f"HTTP_PROXY not set. env output: {output[:500]}"
    assert "https_proxy" in lower, f"HTTPS_PROXY not set. env output: {output[:500]}"

    # Proxy URL must point to the sidecar proxy container
    proxy_name = f"{SANDBOX_PROXY_PREFIX}{sandbox.task_id}"
    assert proxy_name in output, (
        f"Proxy URL doesn't reference proxy container {proxy_name}. Got: {output[:500]}"
    )
    assert str(PROXY_PORT) in output, (
        f"Proxy URL doesn't reference port {PROXY_PORT}. Got: {output[:500]}"
    )


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-010: Sandbox teardown removes containers and network on success
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_teardown_removes_all_resources(task_id_cleanup: str) -> None:
    """VAL-SANDBOX-ISO-010: After teardown, no container/network remains."""
    task_id = task_id_cleanup
    sb = SandboxManager(
        task_id=task_id,
        memory_limit=TEST_MEMORY,
        cpu_limit=TEST_CPU,
        timeout_seconds=TEST_TIMEOUT,
    )
    await sb.setup()
    await asyncio.sleep(1)

    # Verify resources exist
    containers = _docker_ps_containers(task_id)
    assert containers, "Container not found before teardown"

    # Teardown
    await sb.teardown()

    # Wait a moment for cleanup
    await asyncio.sleep(2)

    # Verify resources are gone
    containers_after = _docker_ps_containers(task_id)
    assert not containers_after, f"Containers remain after teardown: {containers_after}"

    networks_after = _docker_network_ls(task_id)
    assert not networks_after, f"Networks remain after teardown: {networks_after}"

    # Check proxy container too
    proxy_result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={SANDBOX_PROXY_PREFIX}{task_id}",
         "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    proxy_names = proxy_result.stdout.strip().splitlines() if proxy_result.stdout.strip() else []
    assert not proxy_names, f"Proxy containers remain after teardown: {proxy_names}"


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-011: Sandbox teardown still occurs on exception path
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_teardown_on_exception_path(task_id_cleanup: str) -> None:
    """VAL-SANDBOX-ISO-011: teardown must clean up even when an exception occurs."""
    task_id = task_id_cleanup

    # Use the async context manager which calls teardown on exception
    try:
        async with SandboxManager(
            task_id=task_id,
            memory_limit=TEST_MEMORY,
            cpu_limit=TEST_CPU,
            timeout_seconds=TEST_TIMEOUT,
        ) as _sb:
            # Verify resources exist
            containers = _docker_ps_containers(task_id)
            assert containers, "Container not created"

            # Simulate an exception mid-run
            raise RuntimeError("Simulated crash mid-task")
    except RuntimeError:
        pass  # Expected

    # Wait for cleanup
    await asyncio.sleep(2)

    # Verify resources are gone
    containers_after = _docker_ps_containers(task_id)
    assert not containers_after, f"Containers remain after exception teardown: {containers_after}"

    networks_after = _docker_network_ls(task_id)
    assert not networks_after, f"Networks remain after exception teardown: {networks_after}"


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-012: Janitor reaps orphaned sandbox containers after SIGKILL
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_janitor_reaps_orphaned_containers(task_id_cleanup: str) -> None:
    """VAL-SANDBOX-ISO-012: After SIGKILL, janitor sweep cleans up within 60s."""
    task_id = task_id_cleanup
    sb = SandboxManager(
        task_id=task_id,
        memory_limit=TEST_MEMORY,
        cpu_limit=TEST_CPU,
        timeout_seconds=TEST_TIMEOUT,
    )
    await sb.setup()
    await asyncio.sleep(1)

    # Verify resources exist
    containers = _docker_ps_containers(task_id)
    assert containers, "Container not created"

    # Simulate SIGKILL: kill the containers without proper teardown
    # (In real scenario, the orchestrator process is killed, leaving containers)
    # We intentionally skip teardown
    sb._container = None  # Break the reference so __aexit__ won't clean up
    sb._proxy_container = None
    sb._network = None

    # Run the janitor
    await SandboxManager.janitor_sweep(max_age_seconds=0)
    # The janitor should remove our containers
    # Note: since containers are freshly created (age < default threshold),
    # we use max_age_seconds=0 to force removal

    await asyncio.sleep(2)

    # Verify all resources are gone
    containers_after = _docker_ps_containers(task_id)
    assert not containers_after, f"Containers remain after janitor: {containers_after}"

    networks_after = _docker_network_ls(task_id)
    assert not networks_after, f"Networks remain after janitor: {networks_after}"


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-013: Sandbox commands are killed at configured timeout
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_commands_killed_at_timeout(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-013: long-running command is killed at timeout, returns TimeoutError."""
    start = time.monotonic()
    with pytest.raises(SandboxTimeoutError) as exc_info:
        await sandbox.run_command("sleep 30", timeout=3)
    elapsed = time.monotonic() - start

    # Must time out faster than the sleep duration
    assert elapsed < 10, f"Timeout took too long: {elapsed:.1f}s"
    assert exc_info.value.timeout_seconds == 3


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-014: Sidecar proxy allowlist contains required registries
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sidecar_proxy_allowlist_contains_required_registries(
    sandbox: SandboxManager,
) -> None:
    """VAL-SANDBOX-ISO-014: proxy allowlist must include required registries."""
    # Check the allowlist config file
    repo_root = Path("/Users/norbertesekiel/Developer/MultiAgenticSystem")
    allowlist_path = repo_root / "infra" / "sandbox-proxy" / "allowlist.yaml"
    assert allowlist_path.exists(), "allowlist.yaml not found"

    content = allowlist_path.read_text()
    required_domains = [
        "pypi.org", "files.pythonhosted.org",
        "registry.npmjs.org", "crates.io",
    ]
    for domain in required_domains:
        assert domain in content, (
            f"Required domain {domain!r} not in allowlist.yaml"
        )

    # Also check huggingface.co is in the default allowlist
    assert "huggingface.co" in content, (
        "huggingface.co missing from allowlist.yaml"
    )

    # Verify that non-allowlisted host returns an error through the proxy
    # Use Python's urllib with explicit proxy setting
    proxy_name = f"{SANDBOX_PROXY_PREFIX}{sandbox.task_id}"
    try:
        await sandbox.run_command(
            f"python3 -c \"import urllib.request; "
            f"proxy = urllib.request.ProxyHandler("
            f"{{'http': 'http://{proxy_name}:{PROXY_PORT}', "
            f"'https': 'http://{proxy_name}:{PROXY_PORT}'}}); "
            f"opener = urllib.request.build_opener(proxy); "
            f"r = opener.open("
            f"urllib.request.Request('http://example.com'), "
            f"timeout=5); print(r.status)\"",
            timeout=15,
        )
        pytest.fail("Proxy allowed non-allowlisted host example.com")
    except SandboxToolError:
        # Expected: proxy denied or connection error
        pass


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-015: Sandbox container runs as non-root UID
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_container_runs_as_non_root_uid(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-015: container runs as non-root UID (default 1000)."""
    output = await sandbox.run_command("id -u", timeout=10)
    uid = int(output.strip())
    assert uid != 0, f"Container is running as root (UID 0), got UID {uid}"
    assert uid == TEST_UID, f"Expected UID {TEST_UID}, got {uid}"


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-016: read_file and run_command tools are scoped to sandbox cwd
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_read_file_scoped_to_sandbox_cwd(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-016: read_file inside cwd succeeds, outside raises error."""
    # Write a file inside the sandbox first
    await sandbox.write_file("test_file.txt", "hello world")

    # Reading inside the cwd must succeed
    content = await sandbox.read_file("test_file.txt")
    assert "hello world" in content, f"read_file inside cwd failed: {content}"

    # Reading outside the cwd must raise PathOutsideSandboxError
    with pytest.raises(PathOutsideSandboxError):
        await sandbox.read_file("/etc/passwd")

    # Traversal attack must also be blocked
    with pytest.raises(PathOutsideSandboxError):
        await sandbox.read_file("../../../etc/passwd")


@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_run_command_cwd_fixed_to_mounted_repo_root(sandbox: SandboxManager) -> None:
    """VAL-SANDBOX-ISO-016: run_command cwd is always /workspace, cannot be overridden."""
    output = await sandbox.run_command("pwd", timeout=10)
    assert "/workspace" in output.strip(), f"run_command cwd is not /workspace: {output}"


# ──────────────────────────────────────────────────────
# VAL-SANDBOX-ISO-017: Sandbox child processes are reaped on container teardown
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_child_processes_reaped_on_teardown(task_id_cleanup: str) -> None:
    """VAL-SANDBOX-ISO-017: No orphaned child processes on host after teardown."""
    task_id = task_id_cleanup
    sb = SandboxManager(
        task_id=task_id,
        memory_limit=TEST_MEMORY,
        cpu_limit=TEST_CPU,
        timeout_seconds=TEST_TIMEOUT,
    )
    await sb.setup()
    await asyncio.sleep(1)

    # Get the container's PID on the host
    inspect_data = _docker_inspect(task_id)
    container_pid = inspect_data.get("State", {}).get("Pid", 0)
    assert container_pid > 0, "Container PID not found"

    # Spawn a long-running subprocess inside the sandbox (using shell)
    await sb.run_command("nohup sleep 300 </dev/null &>/dev/null &", timeout=5)

    # Teardown
    await sb.teardown()
    await asyncio.sleep(3)

    # Verify container is gone
    containers = _docker_ps_containers(task_id)
    assert not containers, f"Container still exists after teardown: {containers}"

    # Verify that the container's PID no longer exists on the host
    # (Container removal kills the entire PID namespace)
    result = subprocess.run(
        ["ps", "-p", str(container_pid)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0, (
        f"Container PID {container_pid} still running on host after teardown"
    )


# ──────────────────────────────────────────────────────
# VAL-CROSS-017: Sandbox teardown on external crash
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_sandbox_teardown_on_external_crash(task_id_cleanup: str) -> None:
    """VAL-CROSS-017: External docker kill of sandbox triggers cleanup."""
    task_id = task_id_cleanup
    sb = SandboxManager(
        task_id=task_id,
        memory_limit=TEST_MEMORY,
        cpu_limit=TEST_CPU,
        timeout_seconds=TEST_TIMEOUT,
    )
    await sb.setup()
    await asyncio.sleep(1)

    # Verify resources exist
    containers = _docker_ps_containers(task_id)
    assert containers, "Container not created"

    # External kill of the sandbox container
    container_name = f"{SANDBOX_CONTAINER_PREFIX}{task_id}"
    subprocess.run(["docker", "kill", container_name], capture_output=True)

    # The SandboxManager should detect the container is gone
    # and be able to clean up the proxy + network
    await sb.teardown()
    await asyncio.sleep(2)

    # Verify all resources are gone
    containers_after = _docker_ps_containers(task_id)
    assert not containers_after, f"Containers remain after external crash: {containers_after}"

    proxy_result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={SANDBOX_PROXY_PREFIX}{task_id}",
         "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    proxy_names = proxy_result.stdout.strip().splitlines() if proxy_result.stdout.strip() else []
    assert not proxy_names, f"Proxy containers remain: {proxy_names}"

    networks_after = _docker_network_ls(task_id)
    assert not networks_after, f"Networks remain after external crash: {networks_after}"


# ──────────────────────────────────────────────────────
# Unit tests (no Docker required)
# ──────────────────────────────────────────────────────

def test_path_validation_inside_cwd() -> None:
    """Path validation accepts paths inside /workspace."""
    sb = SandboxManager(task_id="unit-test")
    # Simple relative path
    result = sb._validate_path("src/main.py")
    assert result == "/workspace/src/main.py"


def test_path_validation_rejects_escape() -> None:
    """Path validation rejects paths outside /workspace."""
    sb = SandboxManager(task_id="unit-test")
    with pytest.raises(PathOutsideSandboxError) as exc_info:
        sb._validate_path("/etc/passwd")
    assert "/etc/passwd" in str(exc_info.value)
    assert "/workspace" in str(exc_info.value)


def test_path_validation_rejects_traversal() -> None:
    """Path validation rejects directory traversal attacks."""
    sb = SandboxManager(task_id="unit-test")
    with pytest.raises(PathOutsideSandboxError):
        sb._validate_path("../../../etc/passwd")


def test_path_validation_rejects_absolute_host_path() -> None:
    """Path validation rejects absolute host paths."""
    sb = SandboxManager(task_id="unit-test")
    with pytest.raises(PathOutsideSandboxError):
        sb._validate_path("/Users/norbertesekiel/secrets")


def test_memory_parse_gibibytes() -> None:
    """Memory parser handles '4g' correctly."""
    assert SandboxManager._parse_memory("4g") == 4 * 1024**3


def test_memory_parse_mebibytes() -> None:
    """Memory parser handles '512m' correctly."""
    assert SandboxManager._parse_memory("512m") == 512 * 1024**2


def test_memory_parse_kibibytes() -> None:
    """Memory parser handles '1024k' correctly."""
    assert SandboxManager._parse_memory("1024k") == 1024 * 1024


def test_memory_parse_bytes() -> None:
    """Memory parser handles raw bytes."""
    assert SandboxManager._parse_memory("1073741824") == 1073741824


def test_default_allowlist_contains_required_registries() -> None:
    """Default allowlist must contain all required registries."""
    required = [
        "pypi.org", "files.pythonhosted.org",
        "registry.npmjs.org", "crates.io",
    ]
    for domain in required:
        assert domain in DEFAULT_ALLOWLIST, f"{domain} missing from default allowlist"


def test_config_defaults() -> None:
    """Config defaults match validation contract expectations."""
    from src.sandbox.config import (
        SANDBOX_CPU_LIMIT,
        SANDBOX_MEMORY_LIMIT,
        SANDBOX_TIMEOUT_SECONDS,
        SANDBOX_USER_UID,
    )
    assert SANDBOX_MEMORY_LIMIT == "4g"
    assert SANDBOX_CPU_LIMIT == 2.0
    assert SANDBOX_TIMEOUT_SECONDS == 600
    assert SANDBOX_USER_UID == 1000


def test_sandbox_manager_naming_conventions() -> None:
    """Sandbox manager follows naming conventions per AGENTS.md."""
    task_id = "abc123"
    sb = SandboxManager(task_id=task_id)
    assert sb.container_name.startswith("sdlc-swarm-sandbox-")
    assert sb.network_name.startswith("sdlc-swarm-net-")
    assert sb.proxy_name.startswith("sdlc-swarm-proxy-")
    assert task_id in sb.container_name
    assert task_id in sb.network_name
    assert task_id in sb.proxy_name
