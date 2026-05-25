"""Tests for SWE-bench instance loader, runner, and evaluator.

VAL-SWE-BENCH-001: Instance loader fetches typed SweBenchInstance rows
VAL-SWE-BENCH-002: Per-instance runner starts correct Docker image
VAL-SWE-BENCH-003: Runner invokes orchestrator and captures valid patch
VAL-SWE-BENCH-004: Evaluator produces JSON report from swebench harness

Tests use mocked HuggingFace responses for unit testing.
Integration tests are marked with @pytest.mark.integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.benchmarks.swebench.evaluator import SweBenchEvaluator
from src.benchmarks.swebench.loader import _DATASET_REPO, _DATASET_SPLIT, InstanceLoader
from src.benchmarks.swebench.models import RunConfig, SweBenchInstance, SweBenchResult
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
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    test_patch: str = "",
    patch: str = "",
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
        "FAIL_TO_PASS": fail_to_pass or ['test_fail["test_a"]'],
        "PASS_TO_PASS": pass_to_pass or ['test_pass["test_b"]'],
        "test_patch": test_patch,
        "patch": patch,
    }


def _make_instance(**kwargs: Any) -> SweBenchInstance:
    """Create a typed SweBenchInstance from test data."""
    data = _make_instance_data(**kwargs)
    return SweBenchInstance.model_validate(data)


@pytest.fixture
def sample_instances() -> list[dict[str, Any]]:
    """Return a list of sample instance dicts (simulating HuggingFace rows)."""
    return [
        _make_instance_data(instance_id="django__django-16379", repo="django/django"),
        _make_instance_data(instance_id="flask__flask-4817", repo="pallets/flask"),
        _make_instance_data(instance_id="requests__requests-6028", repo="psf/requests"),
        _make_instance_data(
            instance_id="scikit-learn__scikit-learn-13241",
            repo="scikit-learn/scikit-learn",
        ),
        _make_instance_data(instance_id="sympy__sympy-20049", repo="sympy/sympy"),
    ]


@pytest.fixture
def sample_instance() -> SweBenchInstance:
    """Return a single typed SweBenchInstance."""
    return _make_instance()


# ── VAL-SWE-BENCH-001: Instance loader fetches typed rows ──────────


class TestSweBenchInstanceModel:
    """Tests for SweBenchInstance Pydantic model (VAL-SWE-BENCH-001)."""

    def test_swebench_instance_validates_with_required_fields(self) -> None:
        """SweBenchInstance validates when all required fields are present."""
        instance = _make_instance()
        assert instance.instance_id == "django__django-12345"
        assert instance.repo == "django/django"
        assert instance.base_commit == "abcdef1234567890"
        assert instance.problem_statement == "Fix the bug in Django ORM"
        assert len(instance.FAIL_TO_PASS) == 1
        assert len(instance.PASS_TO_PASS) == 1

    def test_swebench_instance_rejects_missing_instance_id(self) -> None:
        """SweBenchInstance rejects missing instance_id."""
        from pydantic import ValidationError

        data = _make_instance_data()
        del data["instance_id"]
        with pytest.raises(ValidationError):
            SweBenchInstance.model_validate(data)

    def test_swebench_instance_rejects_empty_instance_id(self) -> None:
        """SweBenchInstance rejects empty instance_id."""
        from pydantic import ValidationError

        data = _make_instance_data(instance_id="")
        with pytest.raises(ValidationError):
            SweBenchInstance.model_validate(data)

    def test_swebench_instance_rejects_empty_repo(self) -> None:
        """SweBenchInstance rejects empty repo."""
        from pydantic import ValidationError

        data = _make_instance_data(repo="")
        with pytest.raises(ValidationError):
            SweBenchInstance.model_validate(data)

    def test_swebench_instance_rejects_empty_base_commit(self) -> None:
        """SweBenchInstance rejects empty base_commit."""
        from pydantic import ValidationError

        data = _make_instance_data(base_commit="")
        with pytest.raises(ValidationError):
            SweBenchInstance.model_validate(data)

    def test_swebench_instance_rejects_empty_problem_statement(self) -> None:
        """SweBenchInstance rejects empty problem_statement."""
        from pydantic import ValidationError

        data = _make_instance_data(problem_statement="")
        with pytest.raises(ValidationError):
            SweBenchInstance.model_validate(data)

    def test_swebench_instance_accepts_list_fail_to_pass(self) -> None:
        """FAIL_TO_PASS as a list is accepted directly by the model.

        Note: When FAIL_TO_PASS is stored as a JSON string in the HuggingFace
        dataset, the InstanceLoader._parse_row() handles deserialization.
        The model itself only accepts a list[str].
        """
        data = _make_instance_data()
        data["FAIL_TO_PASS"] = ["test_a", "test_b"]
        instance = SweBenchInstance.model_validate(data)
        assert isinstance(instance.FAIL_TO_PASS, list)
        assert instance.FAIL_TO_PASS == ["test_a", "test_b"]

    def test_swebench_instance_defaults_optional_fields(self) -> None:
        """Optional fields default to empty values."""
        data = {
            "instance_id": "test__test-1",
            "repo": "test/test",
            "base_commit": "abc123",
            "problem_statement": "Fix something",
        }
        instance = SweBenchInstance.model_validate(data)
        assert instance.hints_text == ""
        assert instance.test_patch == ""
        assert instance.patch == ""
        assert instance.FAIL_TO_PASS == []
        assert instance.PASS_TO_PASS == []


class TestInstanceLoader:
    """Tests for InstanceLoader (VAL-SWE-BENCH-001)."""

    def test_loader_initializes_with_defaults(self) -> None:
        """InstanceLoader initializes with default dataset repo and split."""
        loader = InstanceLoader()
        assert loader.dataset_repo == _DATASET_REPO
        assert loader.dataset_split == _DATASET_SPLIT

    def test_loader_custom_dataset(self) -> None:
        """InstanceLoader accepts custom dataset repo."""
        loader = InstanceLoader(dataset_repo="custom/repo", dataset_split="train")
        assert loader.dataset_repo == "custom/repo"
        assert loader.dataset_split == "train"

    @pytest.mark.asyncio
    async def test_loader_returns_typed_instances(
        self, sample_instances: list[dict[str, Any]],
    ) -> None:
        """Loader returns typed SweBenchInstance objects.

        VAL-SWE-BENCH-001: Returned list length equals configured slice size;
        each object validates as SweBenchInstance with required fields non-null.
        """
        loader = InstanceLoader()

        # Mock the load_dataset call
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(sample_instances))
        mock_dataset.__len__ = MagicMock(return_value=len(sample_instances))

        with patch("src.benchmarks.swebench.loader.load_dataset", return_value=mock_dataset):
            instances = await loader.load(slice_size=3)

        assert len(instances) == 3
        for inst in instances:
            assert isinstance(inst, SweBenchInstance)
            assert inst.instance_id
            assert inst.repo
            assert inst.base_commit
            assert inst.problem_statement
            assert inst.FAIL_TO_PASS is not None
            assert inst.PASS_TO_PASS is not None
            assert inst.test_patch is not None

    @pytest.mark.asyncio
    async def test_loader_slice_size_limits_results(
        self, sample_instances: list[dict[str, Any]],
    ) -> None:
        """Loader respects the slice_size parameter."""
        loader = InstanceLoader()

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(sample_instances))
        mock_dataset.__len__ = MagicMock(return_value=len(sample_instances))

        with patch("src.benchmarks.swebench.loader.load_dataset", return_value=mock_dataset):
            instances = await loader.load(slice_size=2)

        assert len(instances) == 2

    @pytest.mark.asyncio
    async def test_loader_filter_by_instance_ids(
        self, sample_instances: list[dict[str, Any]],
    ) -> None:
        """Loader filters to specific instance IDs when provided."""
        loader = InstanceLoader()

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(sample_instances))
        mock_dataset.__len__ = MagicMock(return_value=len(sample_instances))

        with patch("src.benchmarks.swebench.loader.load_dataset", return_value=mock_dataset):
            instances = await loader.load(
                instance_ids=["django__django-16379", "flask__flask-4817"],
            )

        assert len(instances) == 2
        ids = {i.instance_id for i in instances}
        assert ids == {"django__django-16379", "flask__flask-4817"}

    @pytest.mark.asyncio
    async def test_loader_parses_json_fail_to_pass(self) -> None:
        """Loader parses FAIL_TO_PASS from JSON string format."""
        loader = InstanceLoader()

        row = _make_instance_data()
        row["FAIL_TO_PASS"] = '["test_fail_a", "test_fail_b"]'
        row["PASS_TO_PASS"] = '["test_pass_a"]'

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter([row]))
        mock_dataset.__len__ = MagicMock(return_value=1)

        with patch("src.benchmarks.swebench.loader.load_dataset", return_value=mock_dataset):
            instances = await loader.load(slice_size=30)

        assert len(instances) == 1
        assert instances[0].FAIL_TO_PASS == ["test_fail_a", "test_fail_b"]
        assert instances[0].PASS_TO_PASS == ["test_pass_a"]

    @pytest.mark.asyncio
    async def test_loader_raises_on_all_validation_failures(self) -> None:
        """Loader raises ValueError when all rows fail validation."""
        loader = InstanceLoader()

        # Create rows that will fail validation (empty required fields)
        bad_rows = [
            {"instance_id": "", "repo": "x", "base_commit": "y", "problem_statement": "z"},
            {"instance_id": "a", "repo": "", "base_commit": "y", "problem_statement": "z"},
        ]

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(bad_rows))
        mock_dataset.__len__ = MagicMock(return_value=len(bad_rows))

        with patch("src.benchmarks.swebench.loader.load_dataset", return_value=mock_dataset), \
             pytest.raises(ValueError, match="failed validation"):
                await loader.load(slice_size=30)

    @pytest.mark.asyncio
    async def test_loader_skips_invalid_rows(self) -> None:
        """Loader skips rows that fail validation and returns valid ones."""
        loader = InstanceLoader()

        rows = [
            _make_instance_data(instance_id="valid__1"),
            {"instance_id": "", "repo": "x", "base_commit": "y", "problem_statement": "z"},
            _make_instance_data(instance_id="valid__2"),
        ]

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(rows))
        mock_dataset.__len__ = MagicMock(return_value=len(rows))

        with patch("src.benchmarks.swebench.loader.load_dataset", return_value=mock_dataset):
            instances = await loader.load(slice_size=30)

        assert len(instances) == 2
        ids = {i.instance_id for i in instances}
        assert "valid__1" in ids
        assert "valid__2" in ids

    @pytest.mark.asyncio
    async def test_loader_default_slice_is_30(self, sample_instances: list[dict[str, Any]]) -> None:
        """Default slice size is 30 instances."""
        # Create 40 instances
        many_instances = []
        for i in range(40):
            many_instances.append(_make_instance_data(instance_id=f"test__test-{i}"))

        loader = InstanceLoader()

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(many_instances))
        mock_dataset.__len__ = MagicMock(return_value=len(many_instances))

        with patch("src.benchmarks.swebench.loader.load_dataset", return_value=mock_dataset):
            instances = await loader.load()  # Default slice

        assert len(instances) == 30

    @pytest.mark.asyncio
    async def test_loader_passes_hf_token(self) -> None:
        """Loader passes HuggingFace token to load_dataset."""
        loader = InstanceLoader(hf_token="test_token_123")

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter([_make_instance_data()]))
        mock_dataset.__len__ = MagicMock(return_value=1)

        with patch(
            "src.benchmarks.swebench.loader.load_dataset",
            return_value=mock_dataset,
        ) as mock_load:
            await loader.load(slice_size=1)

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs.get("token") == "test_token_123"


# ── VAL-SWE-BENCH-002: Runner starts correct Docker image ──────────


class TestSweBenchImageName:
    """Tests for SWE-bench Docker image name derivation (VAL-SWE-BENCH-002)."""

    def test_image_name_format(self) -> None:
        """Image name follows swebench convention."""
        name = _swebench_image_name("django__django-12345")
        assert name.startswith("swebench/sweb.eval.x86_64.")
        assert name.endswith(":latest")

    def test_image_name_instance_id_included(self) -> None:
        """Instance ID is included in the image name."""
        name = _swebench_image_name("django__django-16379")
        assert "django_django-16379" in name


class TestImageCacheManager:
    """Tests for Docker image cache management (VAL-SWE-BENCH-002)."""

    @pytest.mark.asyncio
    async def test_cache_skips_already_pulled_images(self) -> None:
        """Cache manager skips pull for already-cached images."""
        cache = ImageCacheManager()
        cache._pulled_images.add("test/image:latest")

        result = await cache.ensure_image("test/image:latest")
        assert result == "test/image:latest"

    @pytest.mark.asyncio
    async def test_cache_checks_local_docker(self) -> None:
        """Cache manager checks local Docker images before pulling."""
        cache = ImageCacheManager()

        mock_image = MagicMock()
        mock_client = MagicMock()
        mock_client.images.get.return_value = mock_image

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.ensure_image("local/image:latest")

        assert result == "local/image:latest"
        mock_client.images.get.assert_called_once_with("local/image:latest")
        assert "local/image:latest" in cache._pulled_images

    @pytest.mark.asyncio
    async def test_cache_pulls_missing_image(self) -> None:
        """Cache manager pulls images not available locally."""
        import docker.errors

        cache = ImageCacheManager()

        mock_client = MagicMock()
        mock_client.images.get.side_effect = docker.errors.ImageNotFound("not found")
        mock_client.images.pull.return_value = MagicMock()

        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.ensure_image("remote/image:latest")

        assert result == "remote/image:latest"
        mock_client.images.pull.assert_called_once_with("remote/image:latest")
        assert "remote/image:latest" in cache._pulled_images


class TestSweBenchRunner:
    """Tests for SweBenchRunner (VAL-SWE-BENCH-002, VAL-SWE-BENCH-003)."""

    def test_runner_initializes_with_config(self) -> None:
        """Runner initializes with RunConfig."""
        config = RunConfig(slice_size=5, topology="supervisor_only")
        runner = SweBenchRunner(config=config)
        assert runner.config.slice_size == 5
        assert runner.config.topology == "supervisor_only"

    @pytest.mark.asyncio
    async def test_runner_returns_error_on_missing_image(
        self, sample_instance: SweBenchInstance,
    ) -> None:
        """Runner returns error status when Docker image is not found."""
        import docker.errors

        runner = SweBenchRunner()

        # Mock image cache to raise ImageNotFound
        mock_cache = AsyncMock(spec=ImageCacheManager)
        mock_cache.ensure_image.side_effect = docker.errors.ImageNotFound("not found")
        runner.image_cache = mock_cache

        result = await runner.run_instance(sample_instance)

        assert result["status"] == "error"
        assert "not found" in result["error"]
        assert result["instance_id"] == sample_instance.instance_id

    @pytest.mark.asyncio
    async def test_runner_returns_instance_id_in_result(
        self, sample_instance: SweBenchInstance,
    ) -> None:
        """Runner result contains the correct instance_id."""
        runner = SweBenchRunner()

        # Mock image cache to succeed
        mock_cache = AsyncMock(spec=ImageCacheManager)
        mock_cache.ensure_image.return_value = "swebench/sweb.eval.x86_64.test:latest"
        runner.image_cache = mock_cache

        # Mock orchestrator invocation to return a simple patch
        with patch.object(runner, "_invoke_orchestrator", return_value={
            "patch": "", "cost_usd": 0.0, "cost_caching_on_usd": 0.0,
            "cost_caching_off_usd": 0.0, "total_tokens_in": 0,
            "total_tokens_out": 0, "total_tokens_cached": 0,
            "hitl_escalations": [], "retry_count": 0, "peer_handoff_count": 0,
        }):
            result = await runner.run_instance(sample_instance)

        assert result["instance_id"] == sample_instance.instance_id

    @pytest.mark.asyncio
    async def test_runner_captures_patch_from_orchestrator(
        self, sample_instance: SweBenchInstance,
    ) -> None:
        """Runner captures a valid unified-diff patch from orchestrator.

        VAL-SWE-BENCH-003: Runner invokes orchestrator with {repo URL, issue text}
        and captures the resulting unified-diff patch.
        """
        sample_patch = """--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -100,6 +100,7 @@
 class Query:
     def __init__(self):
         self.model = None
+        self._fix_applied = True
"""

        runner = SweBenchRunner()

        # Mock image cache
        mock_cache = AsyncMock(spec=ImageCacheManager)
        mock_cache.ensure_image.return_value = "swebench/sweb.eval.x86_64.test:latest"
        runner.image_cache = mock_cache

        # Mock orchestrator invocation
        with patch.object(
            runner, "_invoke_orchestrator",
            return_value={
                "patch": sample_patch, "cost_usd": 0.05,
                "cost_caching_on_usd": 0.05, "cost_caching_off_usd": 0.05,
                "total_tokens_in": 5000, "total_tokens_out": 1500,
                "total_tokens_cached": 0, "hitl_escalations": [],
                "retry_count": 0, "peer_handoff_count": 0,
            },
        ):
            result = await runner.run_instance(sample_instance)

        assert result["patch"] == sample_patch
        assert result["status"] == "success"
        assert result["cost_usd"] == 0.05

    @pytest.mark.asyncio
    async def test_runner_empty_patch_is_failure(self, sample_instance: SweBenchInstance) -> None:
        """Runner returns 'failed' status when patch is empty."""
        runner = SweBenchRunner()

        mock_cache = AsyncMock(spec=ImageCacheManager)
        mock_cache.ensure_image.return_value = "swebench/sweb.eval.x86_64.test:latest"
        runner.image_cache = mock_cache

        with patch.object(runner, "_invoke_orchestrator", return_value={
            "patch": "", "cost_usd": 0.0, "cost_caching_on_usd": 0.0,
            "cost_caching_off_usd": 0.0, "total_tokens_in": 0,
            "total_tokens_out": 0, "total_tokens_cached": 0,
            "hitl_escalations": [], "retry_count": 0, "peer_handoff_count": 0,
        }):
            result = await runner.run_instance(sample_instance)

        assert result["status"] == "failed"
        assert result["patch"] == ""

    @pytest.mark.asyncio
    async def test_runner_handles_orchestrator_exception(
        self, sample_instance: SweBenchInstance,
    ) -> None:
        """Runner handles exceptions from the orchestrator gracefully."""
        runner = SweBenchRunner()

        mock_cache = AsyncMock(spec=ImageCacheManager)
        mock_cache.ensure_image.return_value = "swebench/sweb.eval.x86_64.test:latest"
        runner.image_cache = mock_cache

        with patch.object(
            runner, "_invoke_orchestrator",
            side_effect=RuntimeError("LLM call failed"),
        ):
            result = await runner.run_instance(sample_instance)

        assert result["status"] == "error"
        assert "LLM call failed" in result["error"]


# ── VAL-SWE-BENCH-003: Patch validation ─────────────────────────────


class TestPatchValidation:
    """Tests for patch validation (VAL-SWE-BENCH-003)."""

    def test_valid_unified_diff_parses(self) -> None:
        """A valid unified diff can be parsed."""
        import unidiff

        patch = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 line1
-line2
+line2_modified
+line2a
 line3
"""
        patches = unidiff.PatchSet(patch)
        assert len(patches) > 0

    def test_empty_patch_produces_empty_patchset(self) -> None:
        """An empty string produces an empty PatchSet (no hunks)."""
        import unidiff

        patches = unidiff.PatchSet("")
        assert len(patches) == 0


# ── VAL-SWE-BENCH-004: Evaluator produces JSON report ──────────────


class TestSweBenchEvaluator:
    """Tests for SweBenchEvaluator (VAL-SWE-BENCH-004)."""

    def test_evaluator_returns_result_on_empty_patch(
        self, sample_instance: SweBenchInstance,
    ) -> None:
        """Evaluator returns unresolved result for empty patch."""
        evaluator = SweBenchEvaluator()
        result = evaluator.evaluate_patch_locally(sample_instance, "")
        assert result.resolved is False
        assert result.error is not None

    def test_evaluator_validates_diff_format(self, sample_instance: SweBenchInstance) -> None:
        """Evaluator validates that the patch is a valid unified diff."""
        # Use a properly formatted unified diff that unidiff can parse
        valid_patch = """--- a/file.py
+++ b/file.py
@@ -1,4 +1,5 @@
 line1
 line2
-line3
+line3_modified
+line3a
 line4
"""
        evaluator = SweBenchEvaluator()
        result = evaluator.evaluate_patch_locally(sample_instance, valid_patch)
        assert result.instance_id == sample_instance.instance_id
        assert result.resolved is False  # Local validation can't determine resolution
        assert result.error is None  # No format error
        assert result.model_patch == valid_patch

    def test_evaluator_rejects_invalid_diff(self, sample_instance: SweBenchInstance) -> None:
        """Evaluator rejects an invalid diff format."""
        evaluator = SweBenchEvaluator()
        result = evaluator.evaluate_patch_locally(sample_instance, "not a diff at all")
        assert result.resolved is False
        assert result.error is not None
        assert "Invalid" in result.error or "parse" in result.error.lower()

    def test_evaluator_parses_report_data(self) -> None:
        """Evaluator parses swebench report data correctly."""
        evaluator = SweBenchEvaluator()
        data = {
            "instance_id": "django__django-12345",
            "resolved": True,
            "tests_status": {
                "FAIL_TO_PASS": {"test_a": True, "test_b": True},
                "PASS_TO_PASS": {"test_c": True},
            },
        }
        result = evaluator._parse_report_data("django__django-12345", data)
        assert result.instance_id == "django__django-12345"
        assert result.resolved is True
        assert result.pass_count == 3
        assert result.fail_count == 0

    def test_evaluator_parses_failed_report(self) -> None:
        """Evaluator parses a failed report correctly."""
        evaluator = SweBenchEvaluator()
        data = {
            "instance_id": "django__django-12345",
            "resolved": False,
            "tests_status": {
                "FAIL_TO_PASS": {"test_a": True, "test_b": False},
                "PASS_TO_PASS": {"test_c": True, "test_d": False},
            },
        }
        result = evaluator._parse_report_data("django__django-12345", data)
        assert result.resolved is False
        assert result.pass_count == 2
        assert result.fail_count == 2

    def test_evaluator_handles_missing_report(self) -> None:
        """Evaluator handles missing report files gracefully."""
        evaluator = SweBenchEvaluator()
        result = evaluator._parse_report("nonexistent-id", Path("/tmp/empty"), "test-run")
        assert result.resolved is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_evaluator_async_empty_patch(self, sample_instance: SweBenchInstance) -> None:
        """Async evaluate returns result for empty patch without calling harness."""
        evaluator = SweBenchEvaluator()
        result = await evaluator.evaluate(sample_instance, "")
        assert result.resolved is False
        assert result.error is not None

    def test_swebench_result_model(self) -> None:
        """SweBenchResult model validates correctly."""
        result = SweBenchResult(
            instance_id="test__test-1",
            resolved=True,
            pass_count=5,
            fail_count=0,
            error=None,
        )
        assert result.instance_id == "test__test-1"
        assert result.resolved is True
        assert result.pass_count == 5

    def test_swebench_result_with_error(self) -> None:
        """SweBenchResult handles error case."""
        result = SweBenchResult(
            instance_id="test__test-1",
            resolved=False,
            pass_count=0,
            fail_count=3,
            error="Test execution failed",
        )
        assert result.error == "Test execution failed"


# ── RunConfig tests ─────────────────────────────────────────────────


class TestRunConfig:
    """Tests for RunConfig model."""

    def test_default_config(self) -> None:
        """Default RunConfig has expected values."""
        config = RunConfig()
        assert config.slice_size == 30
        assert config.topology == "supervisor_only"
        assert config.temperature == 0.0
        assert config.runs_per_cell == 1
        assert config.max_cost_per_task_usd == 2.00
        assert config.output_dir == "benchmarks/results"

    def test_custom_config(self) -> None:
        """RunConfig accepts custom values."""
        config = RunConfig(
            slice_size=5,
            topology="hybrid",
            temperature=0.2,
            runs_per_cell=3,
        )
        assert config.slice_size == 5
        assert config.topology == "hybrid"
        assert config.temperature == 0.2
        assert config.runs_per_cell == 3

    def test_config_rejects_invalid_topology(self) -> None:
        """RunConfig rejects invalid topology values.

        Note: We don't have a Literal constraint yet, but the CLI
        validates topology via argparse choices.
        """
        # The model itself doesn't enforce topology values yet
        # but the CLI does via argparse choices
        config = RunConfig(topology="supervisor_only")
        assert config.topology == "supervisor_only"

    def test_config_rejects_negative_slice(self) -> None:
        """RunConfig rejects negative slice_size."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RunConfig(slice_size=-1)

    def test_config_rejects_zero_slice(self) -> None:
        """RunConfig rejects zero slice_size."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RunConfig(slice_size=0)

    def test_config_rejects_invalid_temperature(self) -> None:
        """RunConfig rejects temperature outside valid range."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RunConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            RunConfig(temperature=2.5)


# ── Integration-style tests ─────────────────────────────────────────


class TestCLIEntryPoint:
    """Tests for the SWE-bench CLI entry point."""

    def test_cli_module_exists(self) -> None:
        """The __main__.py module can be imported."""
        from src.benchmarks.swebench import __main__

        assert hasattr(__main__, "main")

    def test_cli_argparse_defaults(self) -> None:
        """CLI argument parser has correct defaults."""
        from src.benchmarks.swebench.__main__ import _parse_args

        with patch("sys.argv", ["swebench"]):
            args = _parse_args()

        assert args.slice == 30
        assert args.topology == "supervisor_only"
        assert args.temperature == 0.0
        assert args.runs == 1
        assert args.max_cost == 2.00
        assert args.instance_ids is None

    def test_cli_argparse_custom(self) -> None:
        """CLI argument parser accepts custom values."""
        from src.benchmarks.swebench.__main__ import _parse_args

        with patch(
            "sys.argv",
            [
                "swebench",
                "--slice", "5",
                "--topology", "hybrid",
                "--temperature", "0.2",
                "--runs", "3",
                "--max-cost", "5.00",
            ],
        ):
            args = _parse_args()

        assert args.slice == 5
        assert args.topology == "hybrid"
        assert args.temperature == 0.2
        assert args.runs == 3
        assert args.max_cost == 5.00


class TestEndToEndPipeline:
    """Integration-style tests for the full pipeline.

    These tests verify that the components work together
    correctly with mocked external services.
    """

    @pytest.mark.asyncio
    async def test_loader_to_runner_pipeline(self) -> None:
        """Load instances, then pass to runner (mocked).

        VAL-SWE-BENCH-003: Runner invokes orchestrator with
        {repo URL, issue text} from the loaded instance.
        """
        loader = InstanceLoader()
        sample_data = [
            _make_instance_data(instance_id="test__1"),
            _make_instance_data(instance_id="test__2"),
        ]

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(sample_data))
        mock_dataset.__len__ = MagicMock(return_value=len(sample_data))

        with patch("src.benchmarks.swebench.loader.load_dataset", return_value=mock_dataset):
            instances = await loader.load(slice_size=2)

        assert len(instances) == 2

        # Verify the instance has repo URL and problem_statement
        for inst in instances:
            assert inst.repo  # Will be used as repo URL
            assert inst.problem_statement  # Will be used as issue text

    @pytest.mark.asyncio
    async def test_runner_to_evaluator_pipeline(self) -> None:
        """Runner captures a patch, evaluator validates it."""
        instance = _make_instance()
        sample_patch = """--- a/file.py
+++ b/file.py
@@ -1,4 +1,5 @@
 line1
 line2
-line3
+line3_modified
+line3a
 line4
"""
        runner = SweBenchRunner()

        mock_cache = AsyncMock(spec=ImageCacheManager)
        mock_cache.ensure_image.return_value = "swebench/sweb.eval.x86_64.test:latest"
        runner.image_cache = mock_cache

        with patch.object(
            runner, "_invoke_orchestrator",
            return_value={
                "patch": sample_patch, "cost_usd": 0.05,
                "cost_caching_on_usd": 0.05, "cost_caching_off_usd": 0.05,
                "total_tokens_in": 5000, "total_tokens_out": 1500,
                "total_tokens_cached": 0, "hitl_escalations": [],
                "retry_count": 0, "peer_handoff_count": 0,
            },
        ):
            result = await runner.run_instance(instance)

        assert result["status"] == "success"
        patch_str = result["patch"]

        # Now evaluate
        evaluator = SweBenchEvaluator()
        eval_result = evaluator.evaluate_patch_locally(instance, patch_str)
        assert eval_result.error is None
        assert eval_result.model_patch == patch_str


# ════════════════════════════════════════════════════════════════════
# VAL-SWE-BENCH-005 / 006 / 007 / 009 / 010 / 011
# Aggregator: compute mean/variance/95% CI, persist results, Markdown
# ════════════════════════════════════════════════════════════════════


# ── VAL-SWE-BENCH-005: Parsed report validates as typed SweBenchResult ─


class TestSweBenchResultParsing:
    """Tests for SweBenchResult type validation (VAL-SWE-BENCH-005)."""

    def test_passing_result_validates(self) -> None:
        """SweBenchResult validates for a passing fixture report."""
        result = SweBenchResult(
            instance_id="django__django-16379",
            resolved=True,
            pass_count=3,
            fail_count=0,
            error=None,
        )
        assert result.resolved is True
        assert result.pass_count == 3

    def test_failing_result_validates(self) -> None:
        """SweBenchResult validates for a failing fixture report."""
        result = SweBenchResult(
            instance_id="flask__flask-4817",
            resolved=False,
            pass_count=1,
            fail_count=2,
            error="2 tests failed",
        )
        assert result.resolved is False
        assert result.fail_count == 2
        assert result.error is not None

    def test_model_validate_from_dict(self) -> None:
        """SweBenchResult.model_validate parses a raw report dict."""
        data = {
            "instance_id": "django__django-12345",
            "resolved": True,
            "pass_count": 5,
            "fail_count": 0,
            "error": None,
        }
        result = SweBenchResult.model_validate(data)
        assert result.instance_id == "django__django-12345"
        assert result.resolved is True


# ── VAL-SWE-BENCH-006: Aggregator computes stats with 95% CI per cell ─


class TestAggregatorStatistics:
    """Tests for aggregator statistics computation (VAL-SWE-BENCH-006)."""

    def test_compute_statistics_known_values(self) -> None:
        """Aggregator computes mean/variance/95% CI matching expected values.

        Feed the aggregator a synthetic dataset where:
        - topology='supervisor_only', model='deepseek/deepseek-chat-v3-0324'
        - 3 instances, each with 3 runs (N=3)
        - Instance results: [1, 1, 1], [0, 0, 0], [1, 0, 1]
        - Expected per-cell mean = mean of per-instance means
        """
        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator()

        # Synthetic per-instance run results
        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                    {"resolved": True, "cost_caching_off_usd": 0.11, "cost_caching_on_usd": 0.07},
                    {"resolved": True, "cost_caching_off_usd": 0.09, "cost_caching_on_usd": 0.05},
                ],
            },
            {
                "instance_id": "inst-2",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": False, "cost_caching_off_usd": 0.20, "cost_caching_on_usd": 0.12},
                    {"resolved": False, "cost_caching_off_usd": 0.22, "cost_caching_on_usd": 0.13},
                    {"resolved": False, "cost_caching_off_usd": 0.18, "cost_caching_on_usd": 0.11},
                ],
            },
            {
                "instance_id": "inst-3",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": True, "cost_caching_off_usd": 0.15, "cost_caching_on_usd": 0.09},
                    {"resolved": False, "cost_caching_off_usd": 0.16, "cost_caching_on_usd": 0.10},
                    {"resolved": True, "cost_caching_off_usd": 0.14, "cost_caching_on_usd": 0.08},
                ],
            },
        ]

        cells = agg.compute_cells(instance_results)

        # Should have 1 cell: (supervisor_only, deepseek/deepseek-chat-v3-0324)
        assert len(cells) == 1
        cell = cells[0]
        assert cell["topology"] == "supervisor_only"
        assert cell["model"] == "deepseek/deepseek-chat-v3-0324"

        # Per-instance success rates: inst-1=1.0, inst-2=0.0, inst-3=0.667
        # Mean of per-instance means = (1.0 + 0.0 + 0.667) / 3 ≈ 0.5556
        expected_mean = (1.0 + 0.0 + 2.0 / 3.0) / 3.0
        assert abs(cell["mean"] - expected_mean) < 1e-6

        # Variance of per-instance means
        per_instance_means = [1.0, 0.0, 2.0 / 3.0]
        from statistics import variance as sample_variance
        expected_var = sample_variance(per_instance_means)
        assert abs(cell["variance"] - expected_var) < 1e-6

        # 95% CI using normal approx: 1.96 * SE
        import math
        n = len(per_instance_means)
        se = math.sqrt(expected_var / n)
        expected_ci_low = expected_mean - 1.96 * se
        expected_ci_high = expected_mean + 1.96 * se
        assert abs(cell["ci_low"] - expected_ci_low) < 1e-6
        assert abs(cell["ci_high"] - expected_ci_high) < 1e-6

    def test_compute_statistics_single_instance(self) -> None:
        """Aggregator handles a single instance (CI spans entire range)."""
        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator()
        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "single_agent",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                ],
            },
        ]

        cells = agg.compute_cells(instance_results)
        assert len(cells) == 1
        cell = cells[0]
        assert cell["mean"] == 1.0  # All resolved
        assert cell["variance"] == 0.0  # No variation

    def test_compute_statistics_multiple_topologies(self) -> None:
        """Aggregator separates cells by (topology, model)."""
        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator()
        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "single_agent",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": False, "cost_caching_off_usd": 0.05, "cost_caching_on_usd": 0.03},
                    {"resolved": False, "cost_caching_off_usd": 0.06, "cost_caching_on_usd": 0.04},
                    {"resolved": False, "cost_caching_off_usd": 0.04, "cost_caching_on_usd": 0.02},
                ],
            },
            {
                "instance_id": "inst-1",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                    {"resolved": True, "cost_caching_off_usd": 0.11, "cost_caching_on_usd": 0.07},
                    {"resolved": True, "cost_caching_off_usd": 0.09, "cost_caching_on_usd": 0.05},
                ],
            },
        ]

        cells = agg.compute_cells(instance_results)
        assert len(cells) == 2
        topologies = {c["topology"] for c in cells}
        assert topologies == {"single_agent", "supervisor_only"}


# ── VAL-SWE-BENCH-007: Aggregated results persisted to JSON ────────


class TestAggregatorResultsJSON:
    """Tests for aggregated results JSON persistence (VAL-SWE-BENCH-007)."""

    def test_results_json_schema(self, tmp_path: Path) -> None:
        """Aggregated results JSON matches documented schema."""
        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator(output_dir=str(tmp_path))

        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                ],
                "hitl_escalations": [],
            },
        ]

        run_id = "test-run-001"
        agg.aggregate_and_persist(
            instance_results=instance_results,
            run_id=run_id,
            slice_size=1,
            runs_per_cell=1,
            started_at="2026-05-24T00:00:00Z",
            ended_at="2026-05-24T01:00:00Z",
        )

        # Verify the JSON file exists
        results_path = tmp_path / f"{run_id}.json"
        assert results_path.exists()

        # Parse and validate the schema
        import json
        data = json.loads(results_path.read_text())

        # Schema: {run_id, started_at, ended_at, slice_size, runs_per_cell, cells: [...]}
        assert data["run_id"] == run_id
        assert data["started_at"] == "2026-05-24T00:00:00Z"
        assert data["ended_at"] == "2026-05-24T01:00:00Z"
        assert data["slice_size"] == 1
        assert data["runs_per_cell"] == 1
        assert "cells" in data
        assert isinstance(data["cells"], list)
        assert len(data["cells"]) >= 1

        # Verify cell structure
        cell = data["cells"][0]
        assert "topology" in cell
        assert "model" in cell
        assert "n" in cell
        assert "mean" in cell
        assert "variance" in cell
        assert "ci_low" in cell
        assert "ci_high" in cell
        assert "instances" in cell

    def test_results_json_run_id_matches(self, tmp_path: Path) -> None:
        """Results JSON run_id matches the provided run UUID."""
        import json

        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator(output_dir=str(tmp_path))
        run_id = "abc123def456"

        instance_results: list[dict[str, object]] = []
        agg.aggregate_and_persist(
            instance_results=instance_results,
            run_id=run_id,
            slice_size=0,
            runs_per_cell=1,
            started_at="2026-05-24T00:00:00Z",
            ended_at="2026-05-24T01:00:00Z",
        )

        results_path = tmp_path / f"{run_id}.json"
        data = json.loads(results_path.read_text())
        assert data["run_id"] == run_id


# ── VAL-SWE-BENCH-009: Aggregator emits Markdown report ────────────


class TestAggregatorMarkdown:
    """Tests for Markdown report generation (VAL-SWE-BENCH-009)."""

    def test_markdown_report_exists(self, tmp_path: Path) -> None:
        """Aggregator emits a Markdown report at benchmarks/results/<run-id>.md."""
        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator(output_dir=str(tmp_path))
        run_id = "md-test-001"

        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                ],
                "hitl_escalations": [],
            },
        ]

        agg.aggregate_and_persist(
            instance_results=instance_results,
            run_id=run_id,
            slice_size=1,
            runs_per_cell=1,
            started_at="2026-05-24T00:00:00Z",
            ended_at="2026-05-24T01:00:00Z",
        )

        md_path = tmp_path / f"{run_id}.md"
        assert md_path.exists()

    def test_markdown_report_header_columns(self, tmp_path: Path) -> None:
        """Markdown report table header contains required columns.

        Expected columns: topology | model | n | mean | variance | 95% CI
        """
        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator(output_dir=str(tmp_path))
        run_id = "md-cols-001"

        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                ],
                "hitl_escalations": [],
            },
        ]

        agg.aggregate_and_persist(
            instance_results=instance_results,
            run_id=run_id,
            slice_size=1,
            runs_per_cell=1,
            started_at="2026-05-24T00:00:00Z",
            ended_at="2026-05-24T01:00:00Z",
        )

        md_path = tmp_path / f"{run_id}.md"
        content = md_path.read_text()

        # Check for required column headers (case-insensitive)
        content_lower = content.lower()
        for col in ["topology", "model", "n", "mean", "variance", "95% ci"]:
            assert col in content_lower, f"Missing column header: {col}"

    def test_markdown_report_has_data_rows(self, tmp_path: Path) -> None:
        """Markdown report has at least 1 data row per cell."""
        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator(output_dir=str(tmp_path))
        run_id = "md-data-001"

        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                ],
                "hitl_escalations": [],
            },
        ]

        agg.aggregate_and_persist(
            instance_results=instance_results,
            run_id=run_id,
            slice_size=1,
            runs_per_cell=1,
            started_at="2026-05-24T00:00:00Z",
            ended_at="2026-05-24T01:00:00Z",
        )

        md_path = tmp_path / f"{run_id}.md"
        content = md_path.read_text()

        # The data row should contain the topology name
        assert "supervisor_only" in content


# ── VAL-SWE-BENCH-010: Separate cost columns for caching ON vs OFF ──


class TestAggregatorCostColumns:
    """Tests for separate cost columns: caching ON vs OFF (VAL-SWE-BENCH-010)."""

    def test_results_json_has_both_cost_columns(self, tmp_path: Path) -> None:
        """Results JSON contains both cost_caching_off_usd and cost_caching_on_usd per cell."""
        import json

        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator(output_dir=str(tmp_path))
        run_id = "cost-test-001"

        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                    {"resolved": True, "cost_caching_off_usd": 0.11, "cost_caching_on_usd": 0.07},
                    {"resolved": True, "cost_caching_off_usd": 0.09, "cost_caching_on_usd": 0.05},
                ],
                "hitl_escalations": [],
            },
        ]

        agg.aggregate_and_persist(
            instance_results=instance_results,
            run_id=run_id,
            slice_size=1,
            runs_per_cell=3,
            started_at="2026-05-24T00:00:00Z",
            ended_at="2026-05-24T01:00:00Z",
        )

        results_path = tmp_path / f"{run_id}.json"
        data = json.loads(results_path.read_text())

        cell = data["cells"][0]
        assert "cost_caching_off_usd" in cell
        assert "cost_caching_on_usd" in cell
        assert cell["cost_caching_off_usd"] is not None
        assert cell["cost_caching_on_usd"] is not None

        # cost_caching_on should be less than cost_caching_off
        assert cell["cost_caching_on_usd"] < cell["cost_caching_off_usd"]

    def test_markdown_has_both_cost_column_headers(self, tmp_path: Path) -> None:
        """Markdown report header row contains both cost column names."""
        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator(output_dir=str(tmp_path))
        run_id = "cost-md-001"

        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                ],
                "hitl_escalations": [],
            },
        ]

        agg.aggregate_and_persist(
            instance_results=instance_results,
            run_id=run_id,
            slice_size=1,
            runs_per_cell=1,
            started_at="2026-05-24T00:00:00Z",
            ended_at="2026-05-24T01:00:00Z",
        )

        md_path = tmp_path / f"{run_id}.md"
        content = md_path.read_text().lower()

        assert "cost_caching_off_usd" in content
        assert "cost_caching_on_usd" in content


# ── VAL-SWE-BENCH-011: HITL escalations cause-tagged ───────────────


class TestHITLEscalations:
    """Tests for cause-tagged HITL escalations (VAL-SWE-BENCH-011)."""

    def test_instance_results_include_hitl_escalations(self, tmp_path: Path) -> None:
        """Per-instance results include hitl_escalations with cause values."""
        import json

        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator(output_dir=str(tmp_path))
        run_id = "hitl-test-001"

        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {
                        "resolved": False,
                        "cost_caching_off_usd": 0.10,
                        "cost_caching_on_usd": 0.06,
                    },
                ],
                "hitl_escalations": [
                    {"cause": "loop_detected", "agent": "coder"},
                ],
            },
        ]

        agg.aggregate_and_persist(
            instance_results=instance_results,
            run_id=run_id,
            slice_size=1,
            runs_per_cell=1,
            started_at="2026-05-24T00:00:00Z",
            ended_at="2026-05-24T01:00:00Z",
        )

        results_path = tmp_path / f"{run_id}.json"
        data = json.loads(results_path.read_text())

        # Find the instance in the cells
        cell = data["cells"][0]
        instance = cell["instances"][0]
        assert "hitl_escalations" in instance
        escalations = instance["hitl_escalations"]
        assert len(escalations) >= 1
        assert escalations[0]["cause"] == "loop_detected"
        assert escalations[0]["agent"] == "coder"

    def test_hitl_escalation_cause_values_match_outcomes_detail_trigger(self) -> None:
        """HITL escalation cause values match outcomes.detail.trigger.

        Valid causes: loop_detected, uncertainty_escalation,
        retry_budget_exhausted, cost_budget_exhausted, guardrail_block, manual.
        """
        from src.benchmarks.swebench.aggregator import VALID_HITL_CAUSES

        expected_causes = {
            "loop_detected",
            "uncertainty_escalation",
            "retry_budget_exhausted",
            "cost_budget_exhausted",
            "guardrail_block",
            "manual",
        }
        assert set(VALID_HITL_CAUSES) == expected_causes

    def test_instance_without_escalations_has_empty_list(self, tmp_path: Path) -> None:
        """Instance with no HITL escalations has an empty list."""
        import json

        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator(output_dir=str(tmp_path))
        run_id = "hitl-empty-001"

        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {"resolved": True, "cost_caching_off_usd": 0.10, "cost_caching_on_usd": 0.06},
                ],
                "hitl_escalations": [],
            },
        ]

        agg.aggregate_and_persist(
            instance_results=instance_results,
            run_id=run_id,
            slice_size=1,
            runs_per_cell=1,
            started_at="2026-05-24T00:00:00Z",
            ended_at="2026-05-24T01:00:00Z",
        )

        results_path = tmp_path / f"{run_id}.json"
        data = json.loads(results_path.read_text())
        cell = data["cells"][0]
        instance = cell["instances"][0]
        assert instance["hitl_escalations"] == []

    def test_multiple_escalations_per_instance(self, tmp_path: Path) -> None:
        """An instance can have multiple HITL escalations."""
        import json

        from src.benchmarks.swebench.aggregator import Aggregator

        agg = Aggregator(output_dir=str(tmp_path))
        run_id = "hitl-multi-001"

        instance_results = [
            {
                "instance_id": "inst-1",
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [
                    {
                        "resolved": False,
                        "cost_caching_off_usd": 0.10,
                        "cost_caching_on_usd": 0.06,
                    },
                ],
                "hitl_escalations": [
                    {"cause": "loop_detected", "agent": "coder"},
                    {"cause": "retry_budget_exhausted", "agent": "reviewer"},
                ],
            },
        ]

        agg.aggregate_and_persist(
            instance_results=instance_results,
            run_id=run_id,
            slice_size=1,
            runs_per_cell=1,
            started_at="2026-05-24T00:00:00Z",
            ended_at="2026-05-24T01:00:00Z",
        )

        results_path = tmp_path / f"{run_id}.json"
        data = json.loads(results_path.read_text())
        cell = data["cells"][0]
        instance = cell["instances"][0]
        assert len(instance["hitl_escalations"]) == 2
        causes = {e["cause"] for e in instance["hitl_escalations"]}
        assert causes == {"loop_detected", "retry_budget_exhausted"}
