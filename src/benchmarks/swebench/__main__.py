"""CLI entry point for the SWE-bench harness.

Usage::

    python -m sdlc_swarm.benchmarks.swebench --slice 30 --topology supervisor_only
    python -m sdlc_swarm.benchmarks.swebench --instance-ids django__django-12345
    python -m sdlc_swarm.benchmarks.swebench --slice 3 --runs 1 --temperature 0

Architecture reference: §2.11 SWE-bench Harness.
The harness is a separate entry point, not a default user flow.
It does not open PRs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the SWE-bench harness."""
    parser = argparse.ArgumentParser(
        description="SWE-bench harness for SDLC-Swarm",
    )
    parser.add_argument(
        "--slice", type=int, default=30,
        help="Number of instances to run (default 30)",
    )
    parser.add_argument(
        "--topology", type=str, default="supervisor_only",
        choices=["single_agent", "supervisor_only", "hybrid"],
        help="Orchestrator topology (default supervisor_only)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="LLM temperature (default 0 for benchmark)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of repeated runs per instance (default 1)",
    )
    parser.add_argument(
        "--max-cost", type=float, default=2.00,
        help="Maximum cost per task in USD (default 2.00)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--instance-ids", type=str, nargs="*", default=None,
        help="Specific instance IDs to run (overrides --slice)",
    )
    return parser.parse_args()


async def _run_harness(args: argparse.Namespace) -> int:
    """Execute the SWE-bench harness with the given configuration."""
    from src.benchmarks.swebench.evaluator import SweBenchEvaluator
    from src.benchmarks.swebench.loader import InstanceLoader
    from src.benchmarks.swebench.models import RunConfig
    from src.benchmarks.swebench.runner import SweBenchRunner

    run_id = uuid4().hex[:12]
    config = RunConfig(
        slice_size=args.slice,
        topology=args.topology,
        temperature=args.temperature,
        runs_per_cell=args.runs,
        max_cost_per_task_usd=args.max_cost,
        output_dir=args.output_dir,
        instance_ids=args.instance_ids,
    )

    logger.info(
        "Starting SWE-bench harness: run_id=%s, slice=%d, topology=%s, temperature=%.1f",
        run_id,
        config.slice_size,
        config.topology,
        config.temperature,
    )

    # Step 1: Load instances
    loader = InstanceLoader()
    try:
        instances = await loader.load(
            slice_size=config.slice_size,
            instance_ids=config.instance_ids,
        )
    except Exception as exc:
        logger.error("Failed to load instances: %s", exc)
        return 1

    if not instances:
        logger.error("No instances loaded")
        return 1

    logger.info("Loaded %d instances", len(instances))

    # Step 2: Run each instance
    runner = SweBenchRunner(config=config)
    evaluator = SweBenchEvaluator(
        output_dir=config.output_dir,
        timeout_seconds=1800,
    )

    all_results: list[dict[str, object]] = []
    run_start = datetime.now(UTC)

    for i, instance in enumerate(instances):
        logger.info(
            "Processing instance %d/%d: %s",
            i + 1,
            len(instances),
            instance.instance_id,
        )

        for run_idx in range(config.runs_per_cell):
            # Run the instance through the orchestrator
            run_result = await runner.run_instance(
                instance=instance,
                run_index=run_idx,
            )

            # Evaluate the captured patch
            patch = run_result.get("patch", "")
            eval_result = await evaluator.evaluate(
                instance=instance,
                patch=patch or "",
                run_id=run_id,
            )

            # Combine results
            combined = {
                "instance_id": instance.instance_id,
                "run_index": run_idx,
                "run_status": run_result.get("status", "unknown"),
                "run_error": run_result.get("error"),
                "cost_usd": run_result.get("cost_usd", 0.0),
                "duration_seconds": run_result.get("duration_seconds", 0.0),
                "eval_resolved": eval_result.resolved,
                "eval_pass_count": eval_result.pass_count,
                "eval_fail_count": eval_result.fail_count,
                "eval_error": eval_result.error,
                "patch": patch,
            }
            all_results.append(combined)

    run_end = datetime.now(UTC)

    # Step 3: Write results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_json = {
        "run_id": run_id,
        "started_at": run_start.isoformat(),
        "ended_at": run_end.isoformat(),
        "slice_size": config.slice_size,
        "runs_per_cell": config.runs_per_cell,
        "topology": config.topology,
        "temperature": config.temperature,
        "instances": all_results,
    }

    results_path = output_dir / f"{run_id}.json"
    results_path.write_text(json.dumps(results_json, indent=2, default=str))
    logger.info("Results written to %s", results_path)

    # Summary
    total = len(all_results)
    resolved = sum(1 for r in all_results if r.get("eval_resolved"))
    logger.info(
        "Harness complete: %d/%d resolved (%.1f%%)",
        resolved,
        total,
        100.0 * resolved / total if total > 0 else 0.0,
    )

    return 0


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    exit_code = asyncio.run(_run_harness(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
