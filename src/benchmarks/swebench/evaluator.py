"""SWE-bench evaluator wrapper — runs official evaluator on captured patches.

Invokes ``python -m swebench.harness.run_evaluation`` with the captured
patch and the instance metadata, producing a JSON report file.

VAL-SWE-BENCH-004: Evaluator produces JSON report from swebench harness.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.benchmarks.swebench.models import SweBenchInstance, SweBenchResult

logger = logging.getLogger(__name__)


class SweBenchEvaluator:
    """Wraps the official swebench evaluator.

    Takes a captured patch and instance metadata, runs the official
    ``swebench.harness.run_evaluation`` evaluation, and parses the
    resulting JSON report into a typed ``SweBenchResult``.

    Usage::

        evaluator = SweBenchEvaluator()
        result = await evaluator.evaluate(instance, patch)
    """

    def __init__(
        self,
        *,
        output_dir: str = "benchmarks/results",
        timeout_seconds: int = 1800,
    ) -> None:
        """Initialize the evaluator.

        Args:
            output_dir: Directory for evaluation output files.
            timeout_seconds: Maximum time to wait for evaluation
                (default 30 minutes per instance — SWE-bench evals
                can be slow due to Docker image operations).
        """
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds

    async def evaluate(
        self,
        instance: SweBenchInstance,
        patch: str,
        *,
        run_id: str = "",
    ) -> SweBenchResult:
        """Evaluate a patch against a SWE-bench instance.

        Writes the model patch to a temporary file, invokes the
        official swebench evaluator, and parses the result.

        Args:
            instance: The SWE-bench instance to evaluate against.
            patch: The unified diff patch produced by the orchestrator.
            run_id: Optional run identifier for organizing output files.

        Returns:
            Typed SweBenchResult with resolved status and test counts.
        """
        if not patch or not patch.strip():
            logger.warning("Empty patch for instance %s; skipping evaluation", instance.instance_id)
            return SweBenchResult(
                instance_id=instance.instance_id,
                resolved=False,
                pass_count=0,
                fail_count=len(instance.FAIL_TO_PASS) if instance.FAIL_TO_PASS else 0,
                error="Empty patch — no evaluation performed",
                model_patch=patch,
            )

        instance_id = instance.instance_id
        logger.info("Evaluating patch for instance %s", instance_id)

        # Create output directory
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Write the model patch to a file expected by swebench
        # The swebench evaluator expects predictions in a specific format
        predictions = [
            {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": "sdlc-swarm",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_file = Path(tmpdir) / "predictions.json"
            predictions_file.write_text(json.dumps(predictions))

            # The swebench harness expects a specific directory structure
            # We write the predictions file and invoke the evaluator
            try:
                result = await self._run_swebench_eval(
                    predictions_file=str(predictions_file),
                    instance_id=instance_id,
                    run_id=run_id,
                    swe_bench_tasks=[instance.model_dump()],
                )
                return result
            except Exception as exc:
                logger.error("SWE-bench evaluation failed for %s: %s", instance_id, exc)
                return SweBenchResult(
                    instance_id=instance_id,
                    resolved=False,
                    pass_count=0,
                    fail_count=0,
                    error=str(exc),
                    model_patch=patch,
                )

    async def _run_swebench_eval(
        self,
        predictions_file: str,
        instance_id: str,
        run_id: str,
        swe_bench_tasks: list[dict[str, Any]],
    ) -> SweBenchResult:
        """Run the official swebench evaluator.

        Uses the swebench Python package's evaluation harness.
        The evaluator creates Docker containers for each instance,
        applies the patch, runs tests, and reports results.

        Args:
            predictions_file: Path to the predictions JSON file.
            instance_id: The instance ID to evaluate.
            run_id: Run identifier for output organization.
            swe_bench_tasks: List of task dicts in swebench format.

        Returns:
            Typed evaluation result.
        """
        import asyncio

        output_dir = Path(self.output_dir) / (run_id or "default")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try to use the swebench package's run_evaluation
        try:
            import importlib

            spec = importlib.util.find_spec("swebench")  # type: ignore[attr-defined]
            if spec is None:
                raise ImportError("swebench not installed")

            # The swebench evaluator writes results to a specific location
            # We need to set up the call properly
            logger.info("Invoking swebench evaluator for instance %s", instance_id)

            # Build command-line arguments for swebench
            args = [
                "--instance_ids", instance_id,
                "--max_workers", "1",
                "--run_id", run_id or "sdlc-swarm",
                "--predictions_path", predictions_file,
                "--swe_bench_tasks", json.dumps(swe_bench_tasks),
                "--output_dir", str(output_dir),
            ]

            # Run in a subprocess to avoid mutating sys.argv
            cmd = [
                sys.executable, "-m", "swebench.harness.run_evaluation",
                *args,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError:
                proc.kill()
                msg = f"Evaluation timed out after {self.timeout_seconds}s"
                raise TimeoutError(msg) from None

            if proc.returncode != 0:
                error_msg = stderr.decode(errors="replace")[:500] if stderr else "Unknown error"
                logger.warning(
                    "SWE-bench evaluator returned code %d for %s: %s",
                    proc.returncode,
                    instance_id,
                    error_msg,
                )
                # Even on non-zero exit, a report may have been produced
                # Fall through to check for the report file

        except ImportError:
            logger.warning("swebench package not available; using fallback evaluation")
            return await self._fallback_evaluate(instance_id, predictions_file, output_dir)

        # Parse the evaluation report
        report = self._parse_report(instance_id, output_dir, run_id or "sdlc-swarm")
        return report

    async def _fallback_evaluate(
        self,
        instance_id: str,
        predictions_file: str,
        output_dir: Path,
    ) -> SweBenchResult:
        """Fallback evaluation when swebench package is not available.

        Reads the predictions file and produces a basic result.
        The actual test execution is not performed — this is a
        placeholder that marks instances as unresolved.
        """
        logger.warning(
            "Using fallback evaluator for %s — results will not reflect actual test execution",
            instance_id,
        )
        return SweBenchResult(
            instance_id=instance_id,
            resolved=False,
            pass_count=0,
            fail_count=0,
            error="Fallback evaluator used — swebench package not available",
        )

    def _parse_report(
        self,
        instance_id: str,
        output_dir: Path,
        run_id: str,
    ) -> SweBenchResult:
        """Parse the swebench evaluation report JSON.

        The swebench evaluator produces a report file at a
        conventional path within the output directory.

        Args:
            instance_id: The instance ID.
            output_dir: Directory containing evaluation output.
            run_id: Run identifier used in the evaluator invocation.

        Returns:
            Typed evaluation result.
        """
        # The swebench evaluator produces a log directory
        # and a report file at a conventional path
        report_paths = [
            output_dir / f"{run_id}" / f"{instance_id}.json",
            output_dir / "swebench-evaluation-report.json",
            output_dir / f"{instance_id}.json",
        ]

        for report_path in report_paths:
            if report_path.exists():
                try:
                    data = json.loads(report_path.read_text())
                    return self._parse_report_data(instance_id, data)
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Failed to parse report at %s: %s", report_path, exc)
                    continue

        # Check for the standard swebench report location
        # swebench.harness typically writes to:
        # <output_dir>/<run_id>.<instance_id>.json
        for report_path in output_dir.glob(f"**/{instance_id}*.json"):
            try:
                data = json.loads(report_path.read_text())
                return self._parse_report_data(instance_id, data)
            except (json.JSONDecodeError, KeyError):
                continue

        logger.warning("No evaluation report found for instance %s in %s", instance_id, output_dir)
        return SweBenchResult(
            instance_id=instance_id,
            resolved=False,
            pass_count=0,
            fail_count=0,
            error="No evaluation report found",
        )

    def _parse_report_data(
        self,
        instance_id: str,
        data: dict[str, Any],
    ) -> SweBenchResult:
        """Parse evaluation report data into a SweBenchResult.

        The swebench report format varies slightly between versions.
        This method handles the common fields.
        """
        # Extract resolved status
        resolved = data.get("resolved", False)

        # Extract test counts
        tests_status = data.get("tests_status", {})

        # Count passes and failures from FAIL_TO_PASS results
        fail_to_pass = tests_status.get("FAIL_TO_PASS", {})
        pass_to_pass = tests_status.get("PASS_TO_PASS", {})

        # In FAIL_TO_PASS: true means the test now passes (was failing before)
        pass_count = sum(1 for v in fail_to_pass.values() if v is True)
        pass_count += sum(1 for v in pass_to_pass.values() if v is True)
        fail_count = sum(1 for v in fail_to_pass.values() if v is False)
        fail_count += sum(1 for v in pass_to_pass.values() if v is False)

        error = data.get("error") or None

        return SweBenchResult(
            instance_id=instance_id,
            resolved=resolved,
            pass_count=pass_count,
            fail_count=fail_count,
            error=error,
            model_patch=data.get("model_patch", ""),
            tests_status=tests_status,
        )

    def evaluate_patch_locally(
        self,
        instance: SweBenchInstance,
        patch: str,
    ) -> SweBenchResult:
        """Synchronous evaluation for cases where the full harness isn't needed.

        Validates the patch format without running Docker-based tests.
        Useful for quick validation and unit testing.

        Args:
            instance: The SWE-bench instance.
            patch: The unified diff patch to validate.

        Returns:
            SweBenchResult with resolved=False (local validation
            cannot determine resolution).
        """
        import unidiff  # type: ignore[import-untyped]

        if not patch or not patch.strip():
            return SweBenchResult(
                instance_id=instance.instance_id,
                resolved=False,
                pass_count=0,
                fail_count=0,
                error="Empty patch",
            )

        # Validate that the patch is a valid unified diff
        try:
            patches = unidiff.PatchSet(patch)
            if not patches:
                return SweBenchResult(
                    instance_id=instance.instance_id,
                    resolved=False,
                    pass_count=0,
                    fail_count=0,
                    error="Patch parsed but contains no file changes",
                    model_patch=patch,
                )
        except Exception as exc:
            return SweBenchResult(
                instance_id=instance.instance_id,
                resolved=False,
                pass_count=0,
                fail_count=0,
                error=f"Invalid unified diff: {exc}",
                model_patch=patch,
            )

        # Patch format is valid
        return SweBenchResult(
            instance_id=instance.instance_id,
            resolved=False,  # Can't determine without running tests
            pass_count=0,
            fail_count=0,
            error=None,
            model_patch=patch,
        )
