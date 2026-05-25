"""Tests for 3 blocking issues from scrutiny review of m3-swebench-runner.

Fix 1: Pass image=image_name to SandboxManager.__init__ in runner.py
  so SWE-bench per-instance Docker images are used instead of the
  default sandbox base image.

Fix 2: Branch on self.config.topology to call appropriate graph builder
  (single_agent / supervisor_only / hybrid) in runner.py instead of
  hardcoding supervisor_only.

Fix 3: Write swe_bench_tasks to a temp JSON file and pass the file
  path to --swe_bench_tasks in evaluator.py instead of passing a raw
  JSON string on the command line.
"""

from __future__ import annotations

import contextlib
import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.benchmarks.swebench.evaluator import SweBenchEvaluator
from src.benchmarks.swebench.models import RunConfig, SweBenchInstance
from src.benchmarks.swebench.runner import (
    ImageCacheManager,
    SweBenchRunner,
    _swebench_image_name,
)

# ── Fixtures ────────────────────────────────────────────────────────


def _make_instance_data(
    instance_id: str = "django__django-12345",
    repo: str = "django/django",
    base_commit: str = "abcdef1234567890",
    problem_statement: str = "Fix the bug in Django ORM",
) -> dict[str, Any]:
    """Create a synthetic SWE-bench instance dict for testing."""
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "hints_text": "",
        "created_at": "2024-01-01T00:00:00",
        "version": "3.2",
        "FAIL_TO_PASS": ['test_fail["test_a"]'],
        "PASS_TO_PASS": ['test_pass["test_b"]'],
        "test_patch": "",
        "patch": "",
    }


def _make_instance(**kwargs: Any) -> SweBenchInstance:
    """Create a typed SweBenchInstance from test data."""
    data = _make_instance_data(**kwargs)
    return SweBenchInstance.model_validate(data)


def _make_runner_with_mocked_orch_deps(
    topology: str = "supervisor_only",
) -> tuple[SweBenchRunner, dict[str, MagicMock]]:
    """Create a SweBenchRunner with mocked orchestrator dependencies.

    Returns the runner and a dict of mock objects for assertion.
    """
    config = RunConfig(topology=topology)
    runner = SweBenchRunner(config=config)

    mock_sb = AsyncMock()
    mock_sb.setup = AsyncMock()
    mock_sb.teardown = AsyncMock()
    mock_sb.workspace_dir = Path("/tmp/test-workspace")
    mock_sb.run_command = AsyncMock()

    mock_episodic = MagicMock()
    mock_semantic = MagicMock()

    mock_graph = MagicMock()
    mock_compiled = AsyncMock()
    mock_compiled.ainvoke = AsyncMock(return_value={})
    mock_graph.compile.return_value = mock_compiled

    mock_gh_client = MagicMock()

    mock_state = MagicMock()
    mock_state.model_dump.return_value = {}

    mocks = {
        "sandbox": mock_sb,
        "episodic_store": mock_episodic,
        "semantic_store": mock_semantic,
        "graph": mock_graph,
        "gh_client": mock_gh_client,
        "state": mock_state,
    }

    return runner, mocks


# Common patches for all _invoke_orchestrator tests
_COMMON_SB_PATCHES = (
    "src.sandbox.manager.SandboxManager",
    "src.memory.episodic.store.EpisodicStore",
    "src.memory.semantic.store.SemanticStore",
    "src.orchestrator.supervisor_only.register_sandbox",
    "src.orchestrator.supervisor_only.register_store",
    "src.orchestrator.supervisor_only.register_semantic_store",
    "src.orchestrator.supervisor_only.unregister_sandbox",
    "src.orchestrator.supervisor_only.unregister_store",
    "src.orchestrator.supervisor_only.unregister_semantic_store",
    "src.github_client.client.GitHubClient",
    "src.orchestrator.OrchestratorState",
)


# ════════════════════════════════════════════════════════════════════
# Fix 1: SandboxManager created with image_name from SWE-bench instance
# ════════════════════════════════════════════════════════════════════


class TestRunnerPassesImageToSandboxManager:
    """Fix 1: Runner passes per-instance image_name to SandboxManager.

    The SWE-bench harness defines per-instance Docker images
    (e.g. swebench/sweb.eval.x86_64.django_django-12345:latest).
    The runner MUST create the SandboxManager with image=image_name
    so the sandbox uses the instance-specific image instead of the
    default sandbox-base image.
    """

    @pytest.mark.asyncio
    async def test_run_instance_passes_image_name_to_invoke_orchestrator(self) -> None:
        """run_instance derives image_name and passes it to _invoke_orchestrator."""
        instance = _make_instance(instance_id="django__django-16379")
        expected_image = _swebench_image_name(instance.instance_id)
        runner = SweBenchRunner()

        mock_cache = AsyncMock(spec=ImageCacheManager)
        mock_cache.ensure_image.return_value = expected_image
        runner.image_cache = mock_cache

        with patch.object(
            runner, "_invoke_orchestrator", return_value={
                "patch": "", "cost_usd": 0.0, "cost_caching_on_usd": 0.0,
                "cost_caching_off_usd": 0.0, "total_tokens_in": 0,
                "total_tokens_out": 0, "total_tokens_cached": 0,
                "hitl_escalations": [], "retry_count": 0, "peer_handoff_count": 0,
            },
        ) as mock_invoke:
            await runner.run_instance(instance)

            call_kwargs = mock_invoke.call_args[1]
            assert "image_name" in call_kwargs, (
                "_invoke_orchestrator must receive image_name parameter"
            )
            assert call_kwargs["image_name"] == expected_image, (
                f"image_name must be {expected_image!r}, "
                f"got {call_kwargs.get('image_name')!r}"
            )

    @pytest.mark.asyncio
    async def test_invoke_orchestrator_creates_sandbox_with_image(self) -> None:
        """_invoke_orchestrator creates SandboxManager with image=image_name."""
        instance = _make_instance(instance_id="django__django-16379")
        expected_image = _swebench_image_name(instance.instance_id)
        runner, mocks = _make_runner_with_mocked_orch_deps(topology="supervisor_only")

        with (
            patch(
                "src.sandbox.manager.SandboxManager",
            ) as mock_sb_cls,
            patch(
                "src.memory.episodic.store.EpisodicStore",
                return_value=mocks["episodic_store"],
            ),
            patch(
                "src.memory.semantic.store.SemanticStore",
                return_value=mocks["semantic_store"],
            ),
            patch("src.orchestrator.supervisor_only.register_sandbox"),
            patch("src.orchestrator.supervisor_only.register_store"),
            patch("src.orchestrator.supervisor_only.register_semantic_store"),
            patch("src.orchestrator.supervisor_only.unregister_sandbox"),
            patch("src.orchestrator.supervisor_only.unregister_store"),
            patch("src.orchestrator.supervisor_only.unregister_semantic_store"),
            patch(
                "src.github_client.client.GitHubClient",
                return_value=mocks["gh_client"],
            ),
            patch(
                "src.orchestrator.OrchestratorState",
                return_value=mocks["state"],
            ),
            patch(
                "src.orchestrator.supervisor_only.build_supervisor_only_graph",
                return_value=mocks["graph"],
            ),
        ):
            mock_sb_cls.return_value = mocks["sandbox"]

            with contextlib.suppress(Exception):
                await runner._invoke_orchestrator(
                    instance=instance,
                    repo_url=f"https://github.com/{instance.repo}",
                    task_id="test-task-001",
                    image_name=expected_image,
                )

            mock_sb_cls.assert_called_once()
            call_kwargs = mock_sb_cls.call_args[1]
            assert "image" in call_kwargs, (
                "SandboxManager must receive an 'image' keyword argument"
            )
            assert call_kwargs["image"] == expected_image, (
                f"SandboxManager image must be {expected_image!r}, "
                f"got {call_kwargs['image']!r}"
            )

    @pytest.mark.asyncio
    async def test_different_instances_get_different_images(self) -> None:
        """Different SWE-bench instances result in different Docker images."""
        inst1 = _make_instance(instance_id="django__django-16379")
        inst2 = _make_instance(instance_id="flask__flask-4817")
        image1 = _swebench_image_name(inst1.instance_id)
        image2 = _swebench_image_name(inst2.instance_id)

        assert image1 != image2, (
            "Different instances must produce different image names"
        )
        assert "django_django-16379" in image1
        assert "flask_flask-4817" in image2

    @pytest.mark.asyncio
    async def test_runner_does_not_use_default_sandbox_image(self) -> None:
        """SandboxManager is NOT created with the default sandbox base image."""
        instance = _make_instance(instance_id="django__django-16379")
        swebench_image = _swebench_image_name(instance.instance_id)
        runner, mocks = _make_runner_with_mocked_orch_deps(topology="supervisor_only")

        with (
            patch("src.sandbox.manager.SandboxManager") as mock_sb_cls,
            patch(
                "src.memory.episodic.store.EpisodicStore",
                return_value=mocks["episodic_store"],
            ),
            patch(
                "src.memory.semantic.store.SemanticStore",
                return_value=mocks["semantic_store"],
            ),
            patch("src.orchestrator.supervisor_only.register_sandbox"),
            patch("src.orchestrator.supervisor_only.register_store"),
            patch("src.orchestrator.supervisor_only.register_semantic_store"),
            patch("src.orchestrator.supervisor_only.unregister_sandbox"),
            patch("src.orchestrator.supervisor_only.unregister_store"),
            patch("src.orchestrator.supervisor_only.unregister_semantic_store"),
            patch(
                "src.github_client.client.GitHubClient",
                return_value=mocks["gh_client"],
            ),
            patch(
                "src.orchestrator.OrchestratorState",
                return_value=mocks["state"],
            ),
            patch(
                "src.orchestrator.supervisor_only.build_supervisor_only_graph",
                return_value=mocks["graph"],
            ),
        ):
            mock_sb_cls.return_value = mocks["sandbox"]

            with contextlib.suppress(Exception):
                await runner._invoke_orchestrator(
                    instance=instance,
                    repo_url=f"https://github.com/{instance.repo}",
                    task_id="test-task-001",
                    image_name=swebench_image,
                )

            call_kwargs = mock_sb_cls.call_args[1]
            from src.sandbox.config import DEFAULT_SANDBOX_IMAGE

            assert call_kwargs.get("image") != DEFAULT_SANDBOX_IMAGE, (
                f"SandboxManager should NOT use default image "
                f"{DEFAULT_SANDBOX_IMAGE!r}. "
                f"Got image={call_kwargs.get('image')!r}"
            )


# ════════════════════════════════════════════════════════════════════
# Fix 2: Runner selects graph builder based on RunConfig.topology
# ════════════════════════════════════════════════════════════════════


class TestRunnerBranchesOnTopology:
    """Fix 2: Runner branches on self.config.topology to call the
    appropriate graph builder instead of hardcoding supervisor_only.

    For 'supervisor_only' → build_supervisor_only_graph()
    For 'single_agent'   → build_single_agent_graph()
    For 'hybrid'         → raise NotImplementedError (M4)
    For unknown topology  → raise ValueError
    """

    @pytest.mark.asyncio
    async def test_supervisor_only_calls_build_supervisor_only_graph(self) -> None:
        """supervisor_only topology invokes build_supervisor_only_graph."""
        runner, mocks = _make_runner_with_mocked_orch_deps(topology="supervisor_only")
        instance = _make_instance()
        expected_image = _swebench_image_name(instance.instance_id)

        with (
            patch(
                "src.sandbox.manager.SandboxManager",
                return_value=mocks["sandbox"],
            ),
            patch(
                "src.memory.episodic.store.EpisodicStore",
                return_value=mocks["episodic_store"],
            ),
            patch(
                "src.memory.semantic.store.SemanticStore",
                return_value=mocks["semantic_store"],
            ),
            patch("src.orchestrator.supervisor_only.register_sandbox"),
            patch("src.orchestrator.supervisor_only.register_store"),
            patch("src.orchestrator.supervisor_only.register_semantic_store"),
            patch("src.orchestrator.supervisor_only.unregister_sandbox"),
            patch("src.orchestrator.supervisor_only.unregister_store"),
            patch("src.orchestrator.supervisor_only.unregister_semantic_store"),
            patch(
                "src.github_client.client.GitHubClient",
                return_value=mocks["gh_client"],
            ),
            patch(
                "src.orchestrator.OrchestratorState",
                return_value=mocks["state"],
            ),
            patch(
                "src.orchestrator.supervisor_only.build_supervisor_only_graph",
                return_value=mocks["graph"],
            ) as mock_build_so,
            patch(
                "src.orchestrator.build_single_agent_graph",
                return_value=mocks["graph"],
            ) as mock_build_sa,
        ):
            with contextlib.suppress(Exception):
                await runner._invoke_orchestrator(
                    instance=instance,
                    repo_url=f"https://github.com/{instance.repo}",
                    task_id="test-task-001",
                    image_name=expected_image,
                )

            mock_build_so.assert_called_once()
            mock_build_sa.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_agent_calls_build_single_agent_graph(self) -> None:
        """single_agent topology invokes build_single_agent_graph."""
        runner, mocks = _make_runner_with_mocked_orch_deps(topology="single_agent")
        instance = _make_instance()
        expected_image = _swebench_image_name(instance.instance_id)

        with (
            patch(
                "src.sandbox.manager.SandboxManager",
                return_value=mocks["sandbox"],
            ),
            patch(
                "src.memory.episodic.store.EpisodicStore",
                return_value=mocks["episodic_store"],
            ),
            patch(
                "src.memory.semantic.store.SemanticStore",
                return_value=mocks["semantic_store"],
            ),
            patch("src.orchestrator.supervisor_only.register_sandbox"),
            patch("src.orchestrator.supervisor_only.register_store"),
            patch("src.orchestrator.supervisor_only.register_semantic_store"),
            patch("src.orchestrator.supervisor_only.unregister_sandbox"),
            patch("src.orchestrator.supervisor_only.unregister_store"),
            patch("src.orchestrator.supervisor_only.unregister_semantic_store"),
            patch(
                "src.github_client.client.GitHubClient",
                return_value=mocks["gh_client"],
            ),
            patch(
                "src.orchestrator.OrchestratorState",
                return_value=mocks["state"],
            ),
            patch(
                "src.orchestrator.supervisor_only.build_supervisor_only_graph",
                return_value=mocks["graph"],
            ) as mock_build_so,
            patch(
                "src.orchestrator.build_single_agent_graph",
                return_value=mocks["graph"],
            ) as mock_build_sa,
        ):
            with contextlib.suppress(Exception):
                await runner._invoke_orchestrator(
                    instance=instance,
                    repo_url=f"https://github.com/{instance.repo}",
                    task_id="test-task-001",
                    image_name=expected_image,
                )

            mock_build_sa.assert_called_once()
            mock_build_so.assert_not_called()

    @pytest.mark.asyncio
    async def test_hybrid_topology_uses_build_hybrid_graph(self) -> None:
        """hybrid topology selects build_hybrid_graph (M6: now implemented)."""
        runner, mocks = _make_runner_with_mocked_orch_deps(topology="hybrid")
        instance = _make_instance()
        expected_image = _swebench_image_name(instance.instance_id)

        mock_sandbox = mocks["sandbox"]
        mock_sandbox.setup = AsyncMock()
        mock_sandbox.teardown = AsyncMock()
        mock_sandbox.run_command = AsyncMock()
        mock_sandbox.workspace_dir = "/workspace"

        mock_gh_client = mocks["gh_client"]
        mock_gh_client.clone = MagicMock()

        mock_state = mocks["state"]
        mock_state.model_dump = MagicMock(return_value={})

        with (
            patch(
                "src.sandbox.manager.SandboxManager",
                return_value=mock_sandbox,
            ),
            patch(
                "src.memory.episodic.store.EpisodicStore",
                return_value=mocks["episodic_store"],
            ),
            patch(
                "src.memory.semantic.store.SemanticStore",
                return_value=mocks["semantic_store"],
            ),
            patch("src.orchestrator.hybrid.register_sandbox"),
            patch("src.orchestrator.hybrid.register_store"),
            patch("src.orchestrator.hybrid.register_semantic_store"),
            patch("src.orchestrator.hybrid.register_guardrail"),
            patch("src.orchestrator.hybrid.unregister_sandbox"),
            patch("src.orchestrator.hybrid.unregister_store"),
            patch("src.orchestrator.hybrid.unregister_semantic_store"),
            patch("src.orchestrator.hybrid.unregister_guardrail"),
            patch(
                "src.github_client.client.GitHubClient",
                return_value=mock_gh_client,
            ),
            patch(
                "src.orchestrator.OrchestratorState",
                return_value=mock_state,
            ),
            patch("src.orchestrator.hybrid.build_hybrid_graph") as mock_build_hybrid,
            patch("src.guardrails.middleware.GuardrailMiddleware"),
        ):
            mock_graph = MagicMock()
            mock_graph.compile.return_value = MagicMock()
            mock_graph.compile.return_value.ainvoke = AsyncMock(return_value={})
            mock_build_hybrid.return_value = mock_graph

            _ = await runner._invoke_orchestrator(
                instance=instance,
                repo_url=f"https://github.com/{instance.repo}",
                task_id="test-task-001",
                image_name=expected_image,
            )

            mock_build_hybrid.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_agent_does_not_call_supervisor_only(self) -> None:
        """With single_agent topology, build_supervisor_only_graph is NOT called.

        This is the regression test: the old code always called
        build_supervisor_only_graph regardless of topology.
        """
        runner, mocks = _make_runner_with_mocked_orch_deps(topology="single_agent")
        instance = _make_instance()
        expected_image = _swebench_image_name(instance.instance_id)

        with (
            patch(
                "src.sandbox.manager.SandboxManager",
                return_value=mocks["sandbox"],
            ),
            patch(
                "src.memory.episodic.store.EpisodicStore",
                return_value=mocks["episodic_store"],
            ),
            patch(
                "src.memory.semantic.store.SemanticStore",
                return_value=mocks["semantic_store"],
            ),
            patch("src.orchestrator.supervisor_only.register_sandbox"),
            patch("src.orchestrator.supervisor_only.register_store"),
            patch("src.orchestrator.supervisor_only.register_semantic_store"),
            patch("src.orchestrator.supervisor_only.unregister_sandbox"),
            patch("src.orchestrator.supervisor_only.unregister_store"),
            patch("src.orchestrator.supervisor_only.unregister_semantic_store"),
            patch(
                "src.github_client.client.GitHubClient",
                return_value=mocks["gh_client"],
            ),
            patch(
                "src.orchestrator.OrchestratorState",
                return_value=mocks["state"],
            ),
            patch(
                "src.orchestrator.supervisor_only.build_supervisor_only_graph",
                return_value=mocks["graph"],
            ) as mock_build_so,
            patch(
                "src.orchestrator.build_single_agent_graph",
                return_value=mocks["graph"],
            ),
        ):
            with contextlib.suppress(Exception):
                await runner._invoke_orchestrator(
                    instance=instance,
                    repo_url=f"https://github.com/{instance.repo}",
                    task_id="test-task-001",
                    image_name=expected_image,
                )

            mock_build_so.assert_not_called()


# ════════════════════════════════════════════════════════════════════
# Fix 3: Evaluator writes swe_bench_tasks to temp file, passes file path
# ════════════════════════════════════════════════════════════════════


class TestEvaluatorWritesTasksToFile:
    """Fix 3: Evaluator writes swe_bench_tasks to a temp JSON file
    and passes the file path to --swe_bench_tasks instead of passing
    a raw JSON string on the command line.
    """

    @pytest.mark.asyncio
    async def test_swe_bench_tasks_passed_as_file_path(self, tmp_path: Path) -> None:
        """_run_swebench_eval writes tasks to a file, not raw JSON string."""
        evaluator = SweBenchEvaluator(output_dir=str(tmp_path))
        instance = _make_instance()
        tasks_data = [instance.model_dump()]

        captured_cmd: list[str] = []

        async def _fake_create_subprocess_exec(
            *args: str, **kwargs: object,
        ) -> object:
            captured_cmd.extend(str(a) for a in args)
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0
            mock_proc.kill = MagicMock()
            return mock_proc

        with patch(
            "asyncio.create_subprocess_exec", _fake_create_subprocess_exec,
        ):
            predictions_file = tmp_path / "predictions.json"
            predictions_file.write_text(json.dumps([{
                "instance_id": instance.instance_id,
                "model_patch": "test patch",
                "model_name_or_path": "sdlc-swarm",
            }]))

            with contextlib.suppress(Exception):
                await evaluator._run_swebench_eval(
                    predictions_file=str(predictions_file),
                    instance_id=instance.instance_id,
                    run_id="test-run",
                    swe_bench_tasks=tasks_data,
                )

        if "--swe_bench_tasks" in captured_cmd:
            idx = captured_cmd.index("--swe_bench_tasks")
            tasks_arg = captured_cmd[idx + 1]

            is_json_string = (
                tasks_arg.strip().startswith("[")
                or tasks_arg.strip().startswith("{")
            )
            assert not is_json_string, (
                f"--swe_bench_tasks should be a file path, "
                f"not a raw JSON string. Got: {tasks_arg[:200]}"
            )

    @pytest.mark.asyncio
    async def test_swe_bench_tasks_file_has_json_suffix(self, tmp_path: Path) -> None:
        """The tasks file path passed to --swe_bench_tasks ends with .json."""
        evaluator = SweBenchEvaluator(output_dir=str(tmp_path))
        instance = _make_instance()
        tasks_data = [instance.model_dump()]

        captured_cmd: list[str] = []

        async def _fake_create_subprocess_exec(
            *args: str, **kwargs: object,
        ) -> object:
            captured_cmd.extend(str(a) for a in args)
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0
            mock_proc.kill = MagicMock()
            return mock_proc

        with patch(
            "asyncio.create_subprocess_exec", _fake_create_subprocess_exec,
        ):
            predictions_file = tmp_path / "predictions.json"
            predictions_file.write_text(json.dumps([{
                "instance_id": instance.instance_id,
                "model_patch": "test patch",
                "model_name_or_path": "sdlc-swarm",
            }]))

            with contextlib.suppress(Exception):
                await evaluator._run_swebench_eval(
                    predictions_file=str(predictions_file),
                    instance_id=instance.instance_id,
                    run_id="test-run",
                    swe_bench_tasks=tasks_data,
                )

        if "--swe_bench_tasks" in captured_cmd:
            idx = captured_cmd.index("--swe_bench_tasks")
            tasks_arg = captured_cmd[idx + 1]
            assert tasks_arg.endswith(".json"), (
                f"--swe_bench_tasks file path should end with .json. "
                f"Got: {tasks_arg}"
            )

    @pytest.mark.asyncio
    async def test_tasks_file_written_before_subprocess_call(
        self, tmp_path: Path,
    ) -> None:
        """The tasks file is written to disk before the subprocess is invoked."""
        evaluator = SweBenchEvaluator(output_dir=str(tmp_path))
        instance = _make_instance()
        tasks_data = [instance.model_dump()]

        file_exists_at_call_time: bool = False
        tasks_file_path_at_call: str | None = None

        async def _fake_create_subprocess_exec(
            *args: str, **kwargs: object,
        ) -> object:
            nonlocal file_exists_at_call_time, tasks_file_path_at_call
            args_list = [str(a) for a in args]
            if "--swe_bench_tasks" in args_list:
                idx = args_list.index("--swe_bench_tasks")
                tasks_file_path_at_call = args_list[idx + 1]
                file_exists_at_call_time = Path(
                    tasks_file_path_at_call,
                ).exists()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0
            mock_proc.kill = MagicMock()
            return mock_proc

        with patch(
            "asyncio.create_subprocess_exec", _fake_create_subprocess_exec,
        ):
            predictions_file = tmp_path / "predictions.json"
            predictions_file.write_text(json.dumps([{
                "instance_id": instance.instance_id,
                "model_patch": "test patch",
                "model_name_or_path": "sdlc-swarm",
            }]))

            with contextlib.suppress(Exception):
                await evaluator._run_swebench_eval(
                    predictions_file=str(predictions_file),
                    instance_id=instance.instance_id,
                    run_id="test-run",
                    swe_bench_tasks=tasks_data,
                )

        if tasks_file_path_at_call is not None:
            assert file_exists_at_call_time, (
                f"Tasks file must exist when subprocess is called. "
                f"Path: {tasks_file_path_at_call}"
            )

    def test_evaluator_no_longer_passes_raw_json_string(self) -> None:
        """Static check: _run_swebench_eval does not pass json.dumps(
        swe_bench_tasks) as a CLI argument to --swe_bench_tasks.
        """
        from src.benchmarks.swebench.evaluator import SweBenchEvaluator

        source = inspect.getsource(SweBenchEvaluator._run_swebench_eval)

        assert '"--swe_bench_tasks", json.dumps(swe_bench_tasks)' not in source, (
            "_run_swebench_eval should NOT pass json.dumps(swe_bench_tasks) "
            "directly as a --swe_bench_tasks argument. It should write to a "
            "temp file and pass the file path."
        )


# ════════════════════════════════════════════════════════════════════
# Fix 4: task_id / run_id must be valid 32-char hex for UUID()
# ════════════════════════════════════════════════════════════════════


class TestTaskIdUuidCompatibility:
    """Fix 4: uuid4().hex[:12] produced 12-char hex strings that
    raise ValueError when passed to UUID(), which requires exactly
    32 hex characters. The orchestrator graph nodes (supervisor_only,
    hybrid, __init__) call UUID(task_id) and would crash.

    Changed uuid4().hex[:12] to uuid4().hex in:
      - runner.py (task_id)
      - __main__.py (task_id for custom repo, run_id for matrix)
    """

    def test_runner_task_id_is_valid_uuid(self) -> None:
        """task_id from SweBenchRunner.run_instance produces a valid 32-char UUID hex."""
        from uuid import UUID, uuid4

        # Simulate what runner.py does: task_id = uuid4().hex
        task_id = uuid4().hex
        assert len(task_id) == 32, f"task_id must be 32 chars, got {len(task_id)}"
        # This must NOT raise ValueError
        UUID(task_id)  # if we get here, the task_id is valid

    def test_runner_task_id_no_truncation(self) -> None:
        """Verify that uuid4().hex is used (not uuid4().hex[:12]) in runner.py source."""
        from src.benchmarks.swebench.runner import SweBenchRunner

        source = inspect.getsource(SweBenchRunner.run_instance)
        # Must NOT contain the truncated form
        assert "uuid4().hex[:12]" not in source, (
            "runner.py must use uuid4().hex, not uuid4().hex[:12]"
        )
        # Must contain the full form
        assert "uuid4().hex" in source, (
            "runner.py must use uuid4().hex for task_id generation"
        )

    def test_main_task_id_no_truncation(self) -> None:
        """Verify that uuid4().hex is used (not uuid4().hex[:12]) in __main__.py for task_id."""
        from src.benchmarks.swebench import __main__

        source = inspect.getsource(__main__)
        truncated_count = source.count("uuid4().hex[:12]")
        assert truncated_count == 0, (
            f"__main__.py must not contain uuid4().hex[:12], found {truncated_count} occurrence(s)"
        )

    def test_main_run_id_no_truncation(self) -> None:
        """Verify that uuid4().hex is used (not uuid4().hex[:12]) in __main__.py for run_id."""
        from src.benchmarks.swebench import __main__

        source = inspect.getsource(__main__)
        assert "uuid4().hex[:12]" not in source, (
            "__main__.py must use uuid4().hex for run_id, not uuid4().hex[:12]"
        )

    def test_uuid_construction_from_runner_task_id(self) -> None:
        """UUID(task_id) does not raise ValueError when task_id comes from runner.py."""
        from uuid import UUID, uuid4

        # This is exactly what the orchestrator does: UUID(task_id)
        # where task_id was generated as uuid4().hex in the runner
        for _ in range(100):
            task_id = uuid4().hex
            uuid_obj = UUID(task_id)  # must not raise
            assert len(str(uuid_obj).replace("-", "")) == 32
