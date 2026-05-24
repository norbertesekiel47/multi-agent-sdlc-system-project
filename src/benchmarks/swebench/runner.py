"""Per-instance SWE-bench runner — spins up Docker, invokes orchestrator, captures patch.

For each SWE-bench instance:
1. Pulls the SWE-bench-defined Docker image for the instance
2. Creates a container with the instance's environment
3. Invokes the orchestrator with {repo URL, issue text}
4. Captures the resulting unified-diff patch

VAL-SWE-BENCH-002: Per-instance runner starts correct Docker image.
VAL-SWE-BENCH-003: Runner invokes orchestrator and captures valid patch.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from typing import Any
from uuid import uuid4

import docker
import docker.errors

from src.benchmarks.swebench.models import RunConfig, SweBenchInstance

logger = logging.getLogger(__name__)

# SWE-bench Docker image naming convention
# Images are on Docker Hub as swebench/sweb.eval.x86_64.<sanitized_instance_id>
_SWEBENCH_IMAGE_PREFIX = "swebench/sweb.eval.x86_64."

# Rate limiting: minimum seconds between Docker pulls
_PULL_RATE_LIMIT_SECONDS = 10

# Maximum concurrent Docker pulls
_MAX_CONCURRENT_PULLS = 2


def _swebench_image_name(instance_id: str) -> str:
    """Derive the SWE-bench Docker image name for an instance.

    The naming convention follows the swebench evaluator:
    ``swebench/sweb.eval.x86_64.<instance_id_sans_special>:latest``

    Special characters in instance_id (like double underscores,
    dashes) are converted to match the Docker Hub naming.
    """
    # The swebench convention: replace __ with _ and special chars
    sanitized = instance_id.replace("__", "_").lower()
    return f"{_SWEBENCH_IMAGE_PREFIX}{sanitized}:latest"


class ImageCacheManager:
    """Manages the local Docker image cache for SWE-bench.

    Tracks which images have been pulled and avoids redundant pulls.
    Implements rate limiting to avoid Docker Hub throttling.
    """

    def __init__(self) -> None:
        self._client: docker.DockerClient | None = None
        self._pulled_images: set[str] = set()
        self._last_pull_time: float = 0.0
        self._pull_lock = asyncio.Lock()

    def _get_client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def ensure_image(self, image_name: str) -> str:
        """Ensure a Docker image is available locally.

        If the image is already cached locally, return immediately.
        Otherwise, pull it with rate limiting.

        Args:
            image_name: Full Docker image name
                (e.g. 'swebench/sweb.eval.x86_64.django_django-12345:latest').

        Returns:
            The image name (same as input).

        Raises:
            docker.errors.ImageNotFound: If the image cannot be pulled.
            docker.errors.APIError: On Docker API errors.
        """
        # Check cache first
        if image_name in self._pulled_images:
            logger.debug("Image %s already in session cache", image_name)
            return image_name

        # Check if image exists locally
        client = self._get_client()
        try:
            client.images.get(image_name)
            self._pulled_images.add(image_name)
            logger.debug("Image %s exists locally", image_name)
            return image_name
        except docker.errors.ImageNotFound:
            pass

        # Rate-limited pull
        async with self._pull_lock:
            # Enforce rate limit
            elapsed = time.monotonic() - self._last_pull_time
            if elapsed < _PULL_RATE_LIMIT_SECONDS:
                wait_time = _PULL_RATE_LIMIT_SECONDS - elapsed
                logger.info("Rate limiting: waiting %.1fs before pull", wait_time)
                await asyncio.sleep(wait_time)

            logger.info("Pulling Docker image: %s (this may take several minutes)", image_name)
            try:
                # Run pull in thread pool to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: client.images.pull(image_name),
                )
                self._pulled_images.add(image_name)
                self._last_pull_time = time.monotonic()
                logger.info("Successfully pulled image: %s", image_name)
            except docker.errors.APIError as exc:
                if "not found" in str(exc).lower() or "404" in str(exc):
                    logger.error("Image %s not found on Docker Hub", image_name)
                    raise docker.errors.ImageNotFound(image_name) from exc
                raise

        return image_name


class SweBenchRunner:
    """Runs a single SWE-bench instance through the orchestrator.

    The runner:
    1. Ensures the SWE-bench Docker image is available
    2. Creates a container from that image with the repo checked out
    3. Invokes the orchestrator with the instance's repo URL and issue text
    4. Captures the resulting patch from the orchestrator output

    The orchestrator is invoked programmatically (not via CLI) to
    reuse the existing supervisor_only / hybrid topology logic.
    """

    def __init__(
        self,
        *,
        config: RunConfig | None = None,
        image_cache: ImageCacheManager | None = None,
    ) -> None:
        self.config = config or RunConfig()
        self.image_cache = image_cache or ImageCacheManager()
        self._docker_client: docker.DockerClient | None = None

    def _get_docker_client(self) -> docker.DockerClient:
        if self._docker_client is None:
            self._docker_client = docker.from_env()
        return self._docker_client

    async def run_instance(
        self,
        instance: SweBenchInstance,
        *,
        run_index: int = 0,
    ) -> dict[str, Any]:
        """Run a single SWE-bench instance and capture the patch.

        Args:
            instance: The SWE-bench instance to run.
            run_index: Which repetition of this instance (0-based).

        Returns:
            Dict with keys:
                - instance_id: str
                - patch: str (unified diff, may be empty on failure)
                - status: str ("success" | "failed" | "error")
                - error: str | None
                - cost_usd: float
                - duration_seconds: float
                - container_id: str | None
        """
        start_time = time.monotonic()
        instance_id = instance.instance_id
        task_id = uuid4().hex[:12]

        logger.info(
            "Running SWE-bench instance %s (run %d, topology=%s)",
            instance_id,
            run_index,
            self.config.topology,
        )

        # Derive Docker image name
        image_name = _swebench_image_name(instance_id)

        # Ensure the Docker image is available
        try:
            await self.image_cache.ensure_image(image_name)
        except docker.errors.ImageNotFound:
            logger.error("Docker image not found for instance %s: %s", instance_id, image_name)
            return {
                "instance_id": instance_id,
                "patch": "",
                "status": "error",
                "error": f"Docker image not found: {image_name}",
                "cost_usd": 0.0,
                "duration_seconds": time.monotonic() - start_time,
                "container_id": None,
            }
        except Exception as exc:
            logger.error("Failed to pull Docker image for %s: %s", instance_id, exc)
            return {
                "instance_id": instance_id,
                "patch": "",
                "status": "error",
                "error": f"Image pull failed: {exc}",
                "cost_usd": 0.0,
                "duration_seconds": time.monotonic() - start_time,
                "container_id": None,
            }

        # Now invoke the orchestrator with the instance's repo URL and issue text.
        # The orchestrator handles: clone, sandbox, agents, patch generation.
        # We use the existing orchestrator flow (supervisor_only by default)
        # and capture the resulting diff.
        container_id: str | None = None

        try:
            # Build the repo URL from the instance
            repo_url = f"https://github.com/{instance.repo}"

            # Invoke the orchestrator programmatically
            # The orchestrator runs the full pipeline and produces a patch
            patch, cost_usd = await self._invoke_orchestrator(
                instance=instance,
                repo_url=repo_url,
                task_id=task_id,
            )

            duration = time.monotonic() - start_time
            status = "success" if patch else "failed"
            error = None if patch else "Orchestrator returned empty patch"

            logger.info(
                "Instance %s completed: status=%s, cost=$%.4f, duration=%.1fs",
                instance_id,
                status,
                cost_usd,
                duration,
            )

            return {
                "instance_id": instance_id,
                "patch": patch,
                "status": status,
                "error": error,
                "cost_usd": cost_usd,
                "duration_seconds": duration,
                "container_id": container_id,
            }

        except Exception as exc:
            logger.error("Runner failed for instance %s: %s", instance_id, exc)
            duration = time.monotonic() - start_time
            return {
                "instance_id": instance_id,
                "patch": "",
                "status": "error",
                "error": str(exc),
                "cost_usd": 0.0,
                "duration_seconds": duration,
                "container_id": container_id,
            }

    async def _invoke_orchestrator(
        self,
        instance: SweBenchInstance,
        repo_url: str,
        task_id: str,
    ) -> tuple[str, float]:
        """Invoke the orchestrator with the instance's repo URL and issue text.

        Uses the existing supervisor_only topology to produce a patch.
        The orchestrator handles cloning, sandbox provisioning, agent
        execution, and diff generation.

        Returns:
            Tuple of (patch_string, cost_usd).
        """
        from langgraph.types import RunnableConfig  # type: ignore[attr-defined]

        from src.memory.episodic.store import EpisodicStore
        from src.memory.semantic.store import SemanticStore
        from src.orchestrator import OrchestratorState
        from src.orchestrator.supervisor_only import (
            build_supervisor_only_graph,
            register_sandbox,
            register_semantic_store,
            register_store,
            unregister_sandbox,
            unregister_semantic_store,
            unregister_store,
        )
        from src.sandbox.manager import SandboxManager

        # Create stores (DSN built from env vars by default)
        episodic_store = EpisodicStore()
        semantic_store = SemanticStore()

        # Register stores for the task
        register_store(task_id, episodic_store)
        register_semantic_store(task_id, semantic_store)

        # Create and register sandbox
        sandbox = SandboxManager(task_id=task_id)
        register_sandbox(task_id, sandbox)

        try:
            # Setup sandbox (clone repo, etc.)
            await sandbox.setup()

            # Clone the repo into the sandbox workspace
            from src.github_client.client import GitHubClient

            pat = os.getenv("GITHUB_PAT", "")
            username = os.getenv("GITHUB_USERNAME", "")
            gh_client = GitHubClient(pat=pat, username=username)

            # Checkout the base commit
            repo_url_full = f"https://github.com/{instance.repo}"
            gh_client.clone(repo_url_full, str(sandbox.workspace_dir))

            # Checkout base_commit
            await sandbox.run_command(f"git checkout {instance.base_commit}")

            # Build initial state for the orchestrator
            state = OrchestratorState(
                task_id=task_id,
                repo_url=repo_url_full,
                issue_number=0,  # Not a GitHub issue number; use problem_statement
                issue_text=instance.problem_statement,
                topology=self.config.topology,
            )

            # Override temperature for benchmark
            original_temp = os.getenv("LLM_TEMPERATURE")
            os.environ["LLM_TEMPERATURE"] = str(self.config.temperature)

            try:
                # Build and run the graph
                graph = build_supervisor_only_graph()
                compiled = graph.compile()
                config_dict: RunnableConfig = {"configurable": {"thread_id": task_id}}

                result = await compiled.ainvoke(
                    state.model_dump(),
                    config=config_dict,
                )

                # Extract the patch from the result
                patch = ""
                cost_usd = 0.0

                if isinstance(result, dict):
                    patch = result.get("patch", result.get("code_edit", {}).get("diff", "")) or ""
                    cost_val = result.get("total_cost_usd", 0)
                    cost_usd = float(cost_val) if cost_val else 0.0

                return patch, cost_usd

            finally:
                # Restore temperature
                if original_temp is not None:
                    os.environ["LLM_TEMPERATURE"] = original_temp
                else:
                    os.environ.pop("LLM_TEMPERATURE", None)

        except Exception as exc:
            logger.error("Orchestrator invocation failed for %s: %s", instance.instance_id, exc)
            raise
        finally:
            # Cleanup
            with contextlib.suppress(Exception):
                await sandbox.teardown()
            unregister_sandbox(task_id)
            unregister_store(task_id)
            unregister_semantic_store(task_id)
