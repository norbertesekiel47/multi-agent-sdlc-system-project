"""Sandbox manager — provisions and tears down ephemeral Docker containers.

Each sandbox consists of:
- A per-task ``--internal`` Docker network (no direct internet egress)
- A sidecar Squid HTTP proxy on a dual-homed network (allowlist-enforced)
- The sandbox container itself, with HTTP_PROXY/HTTPS_PROXY pointing to the proxy
- The cloned repo mounted at /workspace

Teardown removes container + proxy + network + workspace directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from typing import Any

import docker
import docker.errors
import docker.models.containers
import docker.models.networks

from src.sandbox.config import (
    DEFAULT_ALLOWLIST,
    DEFAULT_SANDBOX_IMAGE,
    JANITOR_MAX_AGE_SECONDS,
    PROXY_IMAGE,
    PROXY_PORT,
    SANDBOX_CONTAINER_PREFIX,
    SANDBOX_CPU_LIMIT,
    SANDBOX_MEMORY_LIMIT,
    SANDBOX_NETWORK_PREFIX,
    SANDBOX_PROXY_PREFIX,
    SANDBOX_REPO_MOUNT_POINT,
    SANDBOX_TIMEOUT_SECONDS,
    SANDBOX_USER_UID,
    SANDBOX_WORKSPACE_ROOT,
)
from src.sandbox.errors import (
    PathOutsideSandboxError,
    ProxyError,
    SandboxNotRunningError,
    SandboxTimeoutError,
    SandboxToolError,
)

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages the lifecycle of ephemeral Docker sandbox containers.

    Usage::

        async with SandboxManager(task_id="abc123") as sb:
            await sb.run_command("pip install requests")
            content = await sb.read_file("src/main.py")
            await sb.write_file("src/patch.txt", "fixed")
    """

    def __init__(
        self,
        task_id: str,
        *,
        image: str = DEFAULT_SANDBOX_IMAGE,
        memory_limit: str = SANDBOX_MEMORY_LIMIT,
        cpu_limit: float = SANDBOX_CPU_LIMIT,
        timeout_seconds: int = SANDBOX_TIMEOUT_SECONDS,
        user_uid: int = SANDBOX_USER_UID,
        allowlist: list[str] | None = None,
        workspace_root: str = SANDBOX_WORKSPACE_ROOT,
    ) -> None:
        self.task_id = task_id
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout_seconds = timeout_seconds
        self.user_uid = user_uid
        self.allowlist = allowlist or DEFAULT_ALLOWLIST
        self.workspace_root = workspace_root

        # Derived names
        self.container_name = f"{SANDBOX_CONTAINER_PREFIX}{task_id}"
        self.network_name = f"{SANDBOX_NETWORK_PREFIX}{task_id}"
        self.proxy_name = f"{SANDBOX_PROXY_PREFIX}{task_id}"

        # Workspace directory on host
        self.workspace_dir = Path(workspace_root) / task_id

        # Populated during setup
        self._container: docker.models.containers.Container | None = None
        self._proxy_container: docker.models.containers.Container | None = None
        self._network: docker.models.networks.Network | None = None
        self._client: docker.DockerClient | None = None

    # ── Context manager ──────────────────────────────────

    async def __aenter__(self) -> SandboxManager:
        await self.setup()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.teardown()

    # ── Setup / Teardown ─────────────────────────────────

    async def setup(self) -> None:
        """Provision the sandbox: network, proxy, container."""
        loop = asyncio.get_running_loop()
        self._client = docker.from_env()

        # 1. Create workspace dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # 2. Create internal network
        self._network = await loop.run_in_executor(
            None, self._create_internal_network
        )

        # 3. Start sidecar proxy
        self._proxy_container = await loop.run_in_executor(
            None, self._start_proxy
        )

        # 4. Start sandbox container
        self._container = await loop.run_in_executor(
            None, self._start_container
        )

        logger.info("Sandbox %s provisioned", self.task_id)

    async def teardown(self) -> None:
        """Remove sandbox container, proxy, network, and workspace dir."""
        loop = asyncio.get_running_loop()
        errors: list[str] = []

        # Stop and remove sandbox container
        if self._container is not None:
            try:
                await loop.run_in_executor(None, self._remove_container, self._container)
            except Exception as e:
                errors.append(f"container: {e}")

        # Stop and remove proxy container
        if self._proxy_container is not None:
            try:
                await loop.run_in_executor(
                    None, self._remove_container, self._proxy_container
                )
            except Exception as e:
                errors.append(f"proxy: {e}")

        # Remove network
        if self._network is not None:
            try:
                await loop.run_in_executor(None, self._remove_network)
            except Exception as e:
                errors.append(f"network: {e}")

        # Remove workspace directory
        try:
            if self.workspace_dir.exists():
                shutil.rmtree(self.workspace_dir, ignore_errors=True)
        except Exception as e:
            errors.append(f"workspace: {e}")

        # Reset state
        self._container = None
        self._proxy_container = None
        self._network = None

        if errors:
            logger.warning("Sandbox teardown partial failure for %s: %s", self.task_id, errors)
        else:
            logger.info("Sandbox %s torn down cleanly", self.task_id)

    # ── Docker operations (sync, called via executor) ─────

    def _create_internal_network(self) -> docker.models.networks.Network:
        """Create an internal Docker network with no internet egress."""
        assert self._client is not None
        network = self._client.networks.create(
            self.network_name,
            driver="bridge",
            internal=True,
            labels={"sdlc-swarm": "true", "task_id": self.task_id},
        )
        logger.debug("Created internal network %s", self.network_name)
        return network

    def _start_proxy(self) -> docker.models.containers.Container:
        """Start the sidecar Squid proxy on dual-homed networks.

        The proxy must be on TWO networks:
        1. The internal network — to accept connections from the sandbox container.
           The proxy is started ON this network so Docker's embedded DNS makes
           the proxy hostname resolvable from the sandbox.
        2. A network with internet access (the default bridge) — to forward
           allowlisted traffic to the internet.

        This dual-homing is essential: the internal network has no internet
        egress by design, so the proxy needs its own path to the internet
        for allowlisted requests.
        """
        assert self._client is not None
        assert self._network is not None

        # Build allowlist env var for the proxy
        allowlist_str = ",".join(self.allowlist)

        try:
            # Start proxy on the INTERNAL network first — this ensures Docker's
            # embedded DNS registers the proxy hostname on the internal network,
            # making it resolvable from the sandbox container.
            proxy = self._client.containers.run(
                PROXY_IMAGE,
                name=self.proxy_name,
                detach=True,
                network=self._network.id,
                environment={
                    "ALLOWLIST_DOMAINS": allowlist_str,
                },
                labels={"sdlc-swarm": "true", "task_id": self.task_id},
            )
        except docker.errors.ImageNotFound:
            raise ProxyError(
                f"Proxy image {PROXY_IMAGE!r} not found. "
                "Build it: docker build -t sdlc-swarm/sandbox-proxy:latest "
                "infra/sandbox-proxy/"
            ) from None

        # Also connect proxy to the default bridge network for internet access.
        # The internal network is --internal (no egress), so the proxy needs
        # this second connection to actually reach the allowlisted hosts.
        try:
            bridge_network = self._client.networks.get("bridge")
            bridge_network.connect(proxy)
        except Exception as e:
            logger.warning("Failed to connect proxy to bridge network: %s", e)

        logger.debug(
            "Started proxy %s (internal network %s + bridge for internet)",
            self.proxy_name,
            self.network_name,
        )
        return proxy

    def _start_container(self) -> docker.models.containers.Container:
        """Start the sandbox container attached to the internal network only.

        The container is created connected directly to the internal network
        (not the default bridge), ensuring it has no internet egress.
        """
        assert self._client is not None
        assert self._network is not None

        # Parse memory limit to bytes for Docker SDK
        mem_bytes = self._parse_memory(self.memory_limit)
        nano_cpus = int(self.cpu_limit * 1_000_000_000)

        proxy_url = f"http://{self.proxy_name}:{PROXY_PORT}"

        # Create the container connected directly to the internal network
        # by passing network ID, avoiding the default bridge entirely.
        container = self._client.containers.run(
            self.image,
            name=self.container_name,
            detach=True,
            network=self._network.id,
            user=str(self.user_uid),
            mem_limit=mem_bytes,
            nano_cpus=nano_cpus,
            environment={
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "NO_PROXY": "localhost,127.0.0.1",
                "no_proxy": "localhost,127.0.0.1",
                "HOME": f"{SANDBOX_REPO_MOUNT_POINT}/.home",
                "PIP_NO_CACHE_DIR": "1",
            },
            volumes={
                str(self.workspace_dir.resolve()): {
                    "bind": SANDBOX_REPO_MOUNT_POINT,
                    "mode": "rw",
                },
            },
            working_dir=SANDBOX_REPO_MOUNT_POINT,
            labels={"sdlc-swarm": "true", "task_id": self.task_id},
            # Keep container running (tail /dev/null)
            command="tail -f /dev/null",
            tty=False,
        )

        logger.debug(
            "Started sandbox container %s on network %s (mem=%s, cpu=%s, uid=%s)",
            self.container_name,
            self.network_name,
            self.memory_limit,
            self.cpu_limit,
            self.user_uid,
        )
        return container

    @staticmethod
    def _parse_memory(limit: str) -> int:
        """Parse Docker memory limit string to bytes.

        Accepts: '4g', '512m', '1024k', or raw integer bytes.
        """
        limit = limit.strip().lower()
        multipliers = {"g": 1024**3, "m": 1024**2, "k": 1024}
        if limit[-1] in multipliers:
            return int(float(limit[:-1]) * multipliers[limit[-1]])
        return int(limit)

    def _remove_container(self, container: docker.models.containers.Container) -> None:
        """Force-remove a Docker container."""
        try:
            container.stop(timeout=5)
        except docker.errors.NotFound:
            return
        except Exception:
            pass
        with contextlib.suppress(docker.errors.NotFound):
            container.remove(force=True)

    def _remove_network(self) -> None:
        """Remove the Docker network."""
        assert self._network is not None
        try:
            self._network.remove()
        except Exception:
            # Try by name as fallback
            assert self._client is not None
            try:
                net = self._client.networks.get(self.network_name)
                net.remove()
            except docker.errors.NotFound:
                pass

    # ── Tool surface ─────────────────────────────────────

    def _validate_path(self, path: str) -> str:
        """Ensure *path* is inside the sandbox cwd. Returns the resolved path.

        Raises PathOutsideSandboxError if the path escapes /workspace.
        Handles directory-traversal attacks (e.g. ``../../../etc/passwd``).
        """
        cwd = SANDBOX_REPO_MOUNT_POINT.rstrip("/")

        # Reject absolute paths that don't start with /workspace
        if path.startswith("/"):
            abs_resolved = str(Path(path).resolve()).rstrip("/")
            if not (abs_resolved == cwd or abs_resolved.startswith(cwd + "/")):
                raise PathOutsideSandboxError(path=path, cwd=SANDBOX_REPO_MOUNT_POINT)
            return abs_resolved

        # For relative paths, resolve against /workspace and check the result
        resolved = str((Path(cwd) / path).resolve()).rstrip("/")

        # Check the resolved path starts with the cwd
        if not (resolved == cwd or resolved.startswith(cwd + "/")):
            raise PathOutsideSandboxError(path=path, cwd=SANDBOX_REPO_MOUNT_POINT)
        return resolved

    async def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """Execute a command inside the sandbox container.

        The cwd is always /workspace; it cannot be overridden to a host path.
        Returns stdout+stderr as a string.
        Raises SandboxTimeoutError if the command exceeds the timeout.
        Raises SandboxNotRunningError if the sandbox is not provisioned.
        """
        if self._container is None:
            raise SandboxNotRunningError(
                f"Sandbox {self.task_id} is not running. Call setup() first."
            )

        effective_timeout = timeout or self.timeout_seconds

        # Force cwd to sandbox mount point (VAL-SANDBOX-ISO-016)
        workdir = SANDBOX_REPO_MOUNT_POINT
        if cwd is not None:
            # Validate that cwd is inside sandbox (reject host paths)
            self._validate_path(cwd.lstrip("/"))
            workdir = str(Path(SANDBOX_REPO_MOUNT_POINT) / cwd.lstrip("/"))

        loop = asyncio.get_running_loop()

        try:
            exec_result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._exec_in_container,
                    command,
                    workdir,
                ),
                timeout=effective_timeout,
            )
        except TimeoutError:
            raise SandboxTimeoutError(effective_timeout, command) from None

        exit_code, output = exec_result
        if exit_code != 0:
            raise SandboxToolError(
                f"Command exited with code {exit_code}: {command[:200]}\n{output[:500]}"
            )
        return output

    def _exec_in_container(self, command: str, workdir: str) -> tuple[int, str]:
        """Synchronously execute a command in the container via /bin/sh.

        All commands are wrapped in ``/bin/sh -c`` to enable shell features
        (pipes, ``&&``, globs, subshells, etc.) since ``docker exec`` without
        a shell treats ``|``, ``&&``, etc. as literal arguments.
        """
        assert self._container is not None
        assert self._client is not None
        try:
            exec_instance = self._client.api.exec_create(
                self._container.id,
                ["/bin/sh", "-c", command],
                workdir=workdir,
                stdout=True,
                stderr=True,
            )
            raw_output = self._client.api.exec_start(exec_instance["Id"])
            exit_code_result = self._client.api.exec_inspect(exec_instance["Id"])
            exit_code = exit_code_result.get("ExitCode", -1)
            if isinstance(raw_output, bytes):
                output: str = raw_output.decode("utf-8", errors="replace")
            else:
                output = raw_output
            return exit_code, output
        except docker.errors.NotFound:
            raise SandboxNotRunningError(
                f"Sandbox container {self.container_name} not found."
            ) from None

    async def read_file(self, path: str) -> str:
        """Read a file inside the sandbox cwd.

        Raises PathOutsideSandboxError if the path is outside /workspace.
        """
        resolved = self._validate_path(path)
        # Use cat to read the file
        rel_path = resolved[len(SANDBOX_REPO_MOUNT_POINT):].lstrip("/")
        return await self.run_command(f"cat {rel_path}")

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file inside the sandbox cwd.

        Raises PathOutsideSandboxError if the path is outside /workspace.
        """
        self._validate_path(path)
        rel_path = path.lstrip("/")
        # Use base64 to safely transfer content (avoids shell quoting issues)
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        await self.run_command(
            f"mkdir -p /workspace/$(dirname '{rel_path}') && "
            f"echo '{encoded}' | base64 -d > '/workspace/{rel_path}'"
        )

    async def apply_diff(self, diff: str) -> None:
        """Apply a unified diff inside the sandbox cwd."""
        # Write the diff to a temp file and apply with patch
        escaped_diff = diff.replace("'", "'\\''")
        await self.run_command(
            f"echo '{escaped_diff}' | patch -p1"
        )

    async def run_tests(self, test_command: str = "pytest") -> str:
        """Run tests inside the sandbox. Returns the test output."""
        return await self.run_command(test_command)

    # ── Properties ────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """Check if the sandbox container is currently running."""
        if self._container is None:
            return False
        try:
            self._container.reload()
            return self._container.status == "running"
        except docker.errors.NotFound:
            return False

    @property
    def container_id(self) -> str | None:
        """Return the sandbox container ID (or None if not running)."""
        if self._container is None:
            return None
        return self._container.id

    # ── Static helpers ───────────────────────────────────

    @staticmethod
    async def janitor_sweep(max_age_seconds: int = JANITOR_MAX_AGE_SECONDS) -> int:
        """Remove all stale sdlc-swarm-* containers and networks.

        Returns the number of resources removed.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, SandboxManager._sync_janitor_sweep, max_age_seconds)

    @staticmethod
    def _sync_janitor_sweep(max_age_seconds: int = JANITOR_MAX_AGE_SECONDS) -> int:
        """Synchronously remove stale sdlc-swarm resources."""
        import time

        client = docker.from_env()
        removed = 0
        now = time.time()

        # Remove stale containers
        for container in client.containers.list(all=True, filters={"label": "sdlc-swarm"}):
            created = container.attrs.get("Created", "")
            try:
                # Docker created timestamp is ISO format
                from datetime import datetime

                if created:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    age = now - dt.timestamp()
                    if age > max_age_seconds:
                        container.remove(force=True)
                        removed += 1
            except docker.errors.NotFound:
                pass  # Already removed by another process
            except Exception:
                # If we can't determine age, remove if older than safe threshold
                try:
                    container.remove(force=True)
                    removed += 1
                except docker.errors.NotFound:
                    pass

        # Remove stale networks
        for network in client.networks.list(filters={"label": "sdlc-swarm"}):
            try:
                network.remove()
                removed += 1
            except Exception:
                pass

        return removed
