"""CLI entry point for the SWE-bench harness.

Usage::

    # Single topology run
    python -m src.benchmarks.swebench --slice 30 --topology supervisor_only
    python -m src.benchmarks.swebench --instance-ids django__django-12345
    python -m src.benchmarks.swebench --slice 3 --runs 1 --temperature 0

    # Full benchmark matrix (M6): all 3 topologies × instances × N=3
    python -m src.benchmarks.swebench --matrix --slice 30 --runs 3
    python -m src.benchmarks.swebench --matrix --slice 30 --runs 3 --include-custom-repo

Architecture reference: §2.11 SWE-bench Harness.
The harness is a separate entry point, not a default user flow.
It does not open PRs.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from src.benchmarks.swebench.models import RunConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# All supported topologies for the benchmark matrix
ALL_TOPOLOGIES = ["single_agent", "supervisor_only", "hybrid"]

# Custom curated repo for ablation / demo
_CUSTOM_CURATED_REPO_URL = "https://github.com/norbertesekiel47/sdlc-swarm-curated"


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
        help="Orchestrator topology (default supervisor_only; ignored with --matrix)",
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
    # M6: Full matrix mode
    parser.add_argument(
        "--matrix", action="store_true",
        help="Run full benchmark matrix: all 3 topologies × instances × N runs",
    )
    parser.add_argument(
        "--topologies", type=str, nargs="*", default=None,
        help="Topologies to include in matrix (default: all three)",
    )
    parser.add_argument(
        "--include-custom-repo", action="store_true",
        help="Include custom curated repo issues (~10) in addition to SWE-bench",
    )
    parser.add_argument(
        "--custom-repo-issues", type=int, nargs="*", default=None,
        help="Specific issue numbers from the custom curated repo",
    )
    return parser.parse_args()


async def _load_custom_repo_issues(
    issue_numbers: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Load issues from the custom curated repo.

    Uses the GitHub client to read issues from the test repo.
    Each issue is returned as a dict that can be run through the harness.

    Args:
        issue_numbers: Specific issue numbers to load.  If None,
            loads all open issues (up to ~10).

    Returns:
        List of dicts with keys: issue_number, issue_text, repo.
    """
    from src.github_client.client import GitHubClient

    pat = os.getenv("GITHUB_PAT", "")
    username = os.getenv("GITHUB_USERNAME", "")
    gh_client = GitHubClient(pat=pat, username=username)

    repo_slug = "norbertesekiel47/sdlc-swarm-curated"
    issues: list[dict[str, Any]] = []

    try:
        if issue_numbers:
            for num in issue_numbers:
                try:
                    issue_data = gh_client.read_issue(repo=repo_slug, issue_number=num)
                    issues.append({
                        "issue_number": num,
                        "issue_text": issue_data.get("body", ""),
                        "repo": repo_slug,
                        "title": issue_data.get("title", ""),
                    })
                except Exception as exc:
                    logger.warning("Failed to read issue #%d: %s", num, exc)
        else:
            # Load all open issues from the repo
            from github import Github

            g = Github(pat)
            repo = g.get_repo(repo_slug)
            for issue in repo.get_issues(state="open"):
                if issue.pull_request is not None:
                    continue  # Skip PRs
                issues.append({
                    "issue_number": issue.number,
                    "issue_text": issue.body or "",
                    "repo": repo_slug,
                    "title": issue.title,
                })
    except Exception as exc:
        logger.warning("Failed to load custom repo issues: %s", exc)

    logger.info("Loaded %d custom repo issues", len(issues))
    return issues


async def _run_custom_repo_instance(
    issue: dict[str, Any],
    topology: str,
    run_index: int,
    temperature: float,
    max_cost: float,
    run_id: str,
) -> dict[str, Any]:
    """Run a single custom-repo issue through the orchestrator.

    Unlike SWE-bench instances, custom repo issues don't have
    a gold-standard patch or test suite.  We evaluate success
    by checking if the task completed without errors and produced
    a non-empty patch.

    Args:
        issue: Dict with issue_number, issue_text, repo.
        topology: Orchestrator topology.
        run_index: Which repetition (0-based).
        temperature: LLM temperature.
        max_cost: Max cost per task in USD.
        run_id: Run identifier for output organization.

    Returns:
        Dict with run results.
    """
    import time
    from uuid import uuid4

    from src.benchmarks.swebench.models import RunConfig
    from src.benchmarks.swebench.runner import SweBenchRunner

    start_time = time.monotonic()
    task_id = uuid4().hex
    repo_url = f"https://github.com/{issue['repo']}"

    logger.info(
        "Running custom repo issue #%d (run %d, topology=%s)",
        issue["issue_number"],
        run_index,
        topology,
    )

    # We use the runner's _invoke_orchestrator directly by creating
    # a synthetic SweBenchInstance
    from src.benchmarks.swebench.models import SweBenchInstance

    synthetic_instance = SweBenchInstance(
        instance_id=f"custom-{issue['repo']}-{issue['issue_number']}",
        repo=issue["repo"],
        base_commit="main",  # Use latest main
        problem_statement=issue["issue_text"],
    )

    config = RunConfig(
        topology=topology,
        temperature=temperature,
        max_cost_per_task_usd=max_cost,
        runs_per_cell=1,
    )

    runner = SweBenchRunner(config=config)
    image_name = "sdlc-swarm/sandbox-base:latest"

    try:
        orch_result = await runner._invoke_orchestrator(
            instance=synthetic_instance,
            repo_url=repo_url,
            task_id=task_id,
            image_name=image_name,
        )

        patch = orch_result.get("patch", "")
        cost_usd = orch_result.get("cost_usd", 0.0)
        cost_caching_on_usd = orch_result.get("cost_caching_on_usd", cost_usd)
        cost_caching_off_usd = orch_result.get("cost_caching_off_usd", cost_usd)
        total_tokens_in = orch_result.get("total_tokens_in", 0)
        total_tokens_out = orch_result.get("total_tokens_out", 0)
        total_tokens_cached = orch_result.get("total_tokens_cached", 0)
        hitl_escalations = orch_result.get("hitl_escalations", [])
        retry_count = orch_result.get("retry_count", 0)
        peer_handoff_count = orch_result.get("peer_handoff_count", 0)

        # For custom repo: "resolved" means non-empty patch was produced
        resolved = bool(patch and patch.strip())

        duration = time.monotonic() - start_time

        return {
            "instance_id": synthetic_instance.instance_id,
            "resolved": resolved,
            "patch": patch,
            "cost_usd": cost_usd,
            "cost_caching_on_usd": cost_caching_on_usd,
            "cost_caching_off_usd": cost_caching_off_usd,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_tokens_cached": total_tokens_cached,
            "duration_seconds": duration,
            "hitl_escalations": hitl_escalations,
            "retry_count": retry_count,
            "peer_handoff_count": peer_handoff_count,
            "error": None,
        }
    except Exception as exc:
        duration = time.monotonic() - start_time
        logger.error(
            "Custom repo issue #%d failed: %s",
            issue["issue_number"], exc,
        )
        return {
            "instance_id": f"custom-{issue['repo']}-{issue['issue_number']}",
            "resolved": False,
            "patch": "",
            "cost_usd": 0.0,
            "cost_caching_on_usd": 0.0,
            "cost_caching_off_usd": 0.0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_tokens_cached": 0,
            "duration_seconds": duration,
            "hitl_escalations": [],
            "retry_count": 0,
            "peer_handoff_count": 0,
            "error": str(exc),
        }


async def _run_single_topology(
    topology: str,
    instances: list[Any],
    config: RunConfig,
    run_id: str,
    evaluator: Any,
    custom_issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run all instances for a single topology.

    Args:
        topology: The topology to run.
        instances: List of SweBenchInstance objects.
        config: RunConfig with topology overridden.
        run_id: Run identifier.
        evaluator: SweBenchEvaluator instance.
        custom_issues: List of custom repo issue dicts.

    Returns:
        List of instance result dicts for the aggregator.
    """
    from src.benchmarks.swebench.runner import SweBenchRunner

    # Override topology in config
    topo_config = config.model_copy(update={"topology": topology})
    runner = SweBenchRunner(config=topo_config)

    instance_run_map: dict[str, dict[str, Any]] = {}

    # ── SWE-bench instances ──────────────────────────────────
    for i, instance in enumerate(instances):
        instance_id = instance.instance_id
        logger.info(
            "[%s] Processing SWE-bench instance %d/%d: %s",
            topology, i + 1, len(instances), instance_id,
        )

        instance_key = f"{instance_id}:{topology}"
        if instance_key not in instance_run_map:
            instance_run_map[instance_key] = {
                "instance_id": instance_id,
                "topology": topology,
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [],
                "hitl_escalations": [],
            }

        for run_idx in range(config.runs_per_cell):
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

            # Collect all metrics from the runner result
            cost_caching_off_usd = float(run_result.get("cost_caching_off_usd", 0.0))
            cost_caching_on_usd = float(run_result.get("cost_caching_on_usd", 0.0))
            hitl_escalations = run_result.get("hitl_escalations", [])
            if isinstance(hitl_escalations, list):
                instance_run_map[instance_key]["hitl_escalations"].extend(hitl_escalations)

            instance_run_map[instance_key]["run_results"].append({
                "resolved": eval_result.resolved,
                "cost_caching_off_usd": cost_caching_off_usd,
                "cost_caching_on_usd": cost_caching_on_usd,
                "total_tokens_in": int(run_result.get("total_tokens_in", 0)),
                "total_tokens_out": int(run_result.get("total_tokens_out", 0)),
                "total_tokens_cached": int(run_result.get("total_tokens_cached", 0)),
                "duration_seconds": float(run_result.get("duration_seconds", 0.0)),
                "retry_count": int(run_result.get("retry_count", 0)),
                "peer_handoff_count": int(run_result.get("peer_handoff_count", 0)),
                "pass_count": eval_result.pass_count,
                "fail_count": eval_result.fail_count,
                "error": eval_result.error,
            })

    # ── Custom repo issues ────────────────────────────────────
    for issue in custom_issues:
        issue_id = f"custom-{issue['repo']}-{issue['issue_number']}"
        logger.info(
            "[%s] Processing custom repo issue #%d: %s",
            topology, issue["issue_number"], issue_id,
        )

        instance_key = f"{issue_id}:{topology}"
        if instance_key not in instance_run_map:
            instance_run_map[instance_key] = {
                "instance_id": issue_id,
                "topology": topology,
                "model": "deepseek/deepseek-chat-v3-0324",
                "run_results": [],
                "hitl_escalations": [],
            }

        for run_idx in range(config.runs_per_cell):
            run_result = await _run_custom_repo_instance(
                issue=issue,
                topology=topology,
                run_index=run_idx,
                temperature=config.temperature,
                max_cost=config.max_cost_per_task_usd,
                run_id=run_id,
            )

            hitl_escalations = run_result.get("hitl_escalations", [])
            if isinstance(hitl_escalations, list):
                instance_run_map[instance_key]["hitl_escalations"].extend(hitl_escalations)

            instance_run_map[instance_key]["run_results"].append({
                "resolved": run_result.get("resolved", False),
                "cost_caching_off_usd": float(run_result.get("cost_caching_off_usd", 0.0)),
                "cost_caching_on_usd": float(run_result.get("cost_caching_on_usd", 0.0)),
                "total_tokens_in": int(run_result.get("total_tokens_in", 0)),
                "total_tokens_out": int(run_result.get("total_tokens_out", 0)),
                "total_tokens_cached": int(run_result.get("total_tokens_cached", 0)),
                "duration_seconds": float(run_result.get("duration_seconds", 0.0)),
                "retry_count": int(run_result.get("retry_count", 0)),
                "peer_handoff_count": int(run_result.get("peer_handoff_count", 0)),
                "pass_count": 0,
                "fail_count": 0,
                "error": run_result.get("error"),
            })

    return list(instance_run_map.values())


async def _run_harness(args: argparse.Namespace) -> int:
    """Execute the SWE-bench harness with the given configuration."""
    from src.benchmarks.swebench.aggregator import Aggregator
    from src.benchmarks.swebench.evaluator import SweBenchEvaluator
    from src.benchmarks.swebench.loader import InstanceLoader

    run_id = uuid4().hex

    # Determine topologies to run
    topologies = (
        (args.topologies or ALL_TOPOLOGIES) if args.matrix else [args.topology]
    )

    config = RunConfig(
        slice_size=args.slice,
        topology=topologies[0] if not args.matrix else "supervisor_only",
        temperature=args.temperature,
        runs_per_cell=args.runs,
        max_cost_per_task_usd=args.max_cost,
        output_dir=args.output_dir,
        instance_ids=args.instance_ids,
    )

    logger.info(
        "Starting SWE-bench harness: run_id=%s, topologies=%s, slice=%d, "
        "temperature=%.1f, runs_per_cell=%d, matrix=%s",
        run_id,
        topologies,
        config.slice_size,
        config.temperature,
        config.runs_per_cell,
        args.matrix,
    )

    # Step 1: Load SWE-bench instances
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

    logger.info("Loaded %d SWE-bench instances", len(instances))

    # Step 2: Load custom repo issues if requested
    custom_issues: list[dict[str, Any]] = []
    if args.include_custom_repo:
        custom_issues = await _load_custom_repo_issues(
            issue_numbers=args.custom_repo_issues,
        )

    # Step 3: Run the harness
    evaluator = SweBenchEvaluator(
        output_dir=config.output_dir,
        timeout_seconds=1800,
    )

    all_instance_results: list[dict[str, Any]] = []
    run_start = datetime.now(UTC)

    for topo_idx, topology in enumerate(topologies):
        logger.info(
            "=== Running topology %d/%d: %s ===",
            topo_idx + 1,
            len(topologies),
            topology,
        )

        try:
            topo_results = await _run_single_topology(
                topology=topology,
                instances=instances,
                config=config,
                run_id=run_id,
                evaluator=evaluator,
                custom_issues=custom_issues,
            )
            all_instance_results.extend(topo_results)
        except Exception as exc:
            logger.error("Topology %s failed: %s", topology, exc)
            # Continue with other topologies rather than aborting

    run_end = datetime.now(UTC)

    # Step 4: Aggregate results and persist
    total_instances = len(instances) + len(custom_issues)
    aggregator = Aggregator(output_dir=config.output_dir)

    aggregator.aggregate_and_persist(
        instance_results=all_instance_results,
        run_id=run_id,
        slice_size=total_instances,
        runs_per_cell=config.runs_per_cell,
        started_at=run_start.isoformat(),
        ended_at=run_end.isoformat(),
    )

    # Summary
    total_runs = sum(len(ir["run_results"]) for ir in all_instance_results)
    total_resolved = sum(
        1 for ir in all_instance_results
        for r in ir["run_results"] if r.get("resolved")
    )
    total_cost = sum(
        r.get("cost_caching_on_usd", 0.0)
        for ir in all_instance_results
        for r in ir["run_results"]
    )
    total_hitl = sum(
        len(ir.get("hitl_escalations", []))
        for ir in all_instance_results
    )
    logger.info(
        "Harness complete: %d/%d resolved (%.1f%%), total cost=$%.2f, "
        "HITL escalations=%d",
        total_resolved,
        total_runs,
        100.0 * total_resolved / total_runs if total_runs > 0 else 0.0,
        total_cost,
        total_hitl,
    )

    return 0


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    exit_code = asyncio.run(_run_harness(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
