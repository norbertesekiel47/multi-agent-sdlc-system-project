"""SWE-bench aggregator — computes statistics, persists results, emits Markdown.

Reads per-instance evaluation results across N runs, computes
mean/variance/95% CI per (topology, model) cell, and persists:

- ``benchmarks/results/<run-id>.json`` — machine-readable results
- ``benchmarks/results/<run-id>.md`` — human-readable Markdown report

VAL-SWE-BENCH-006: Aggregator computes stats with 95% CI per cell.
VAL-SWE-BENCH-007: Aggregated results persisted to JSON with documented schema.
VAL-SWE-BENCH-009: Aggregator emits Markdown report with result table.
VAL-SWE-BENCH-010: Separate cost columns for caching ON vs OFF.
VAL-SWE-BENCH-011: HITL escalations cause-tagged.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Valid HITL escalation causes (matching outcomes.detail.trigger) ──

VALID_HITL_CAUSES: list[str] = [
    "loop_detected",
    "uncertainty_escalation",
    "retry_budget_exhausted",
    "cost_budget_exhausted",
    "guardrail_block",
    "manual",
]

# 95% CI z-value for normal approximation
_Z_95 = 1.96


def _mean(values: list[float]) -> float:
    """Compute the arithmetic mean of a list of floats."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    """Compute the sample variance (Bessel-corrected) of a list of floats.

    Returns 0.0 for a single-element list (no variance).
    """
    n = len(values)
    if n <= 1:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)


def _ci_95(values: list[float]) -> tuple[float, float]:
    """Compute the 95% confidence interval using normal approximation.

    CI = mean ± z_{0.975} × SE, where SE = sqrt(var / n).

    Returns (ci_low, ci_high).  Returns (mean, mean) for n <= 1.
    """
    n = len(values)
    if n <= 1:
        m = _mean(values)
        return m, m
    m = _mean(values)
    se = math.sqrt(_variance(values) / n)
    return m - _Z_95 * se, m + _Z_95 * se


class Aggregator:
    """Aggregates SWE-bench evaluation results across runs.

    Computes per-cell statistics (mean, variance, 95% CI) from
    per-instance per-run resolved indicators and costs, then
    persists the results as JSON and Markdown.

    Usage::

        agg = Aggregator(output_dir="benchmarks/results")
        agg.aggregate_and_persist(
            instance_results=...,
            run_id="abc123",
            slice_size=30,
            runs_per_cell=3,
            started_at="...",
            ended_at="...",
        )
    """

    def __init__(self, *, output_dir: str = "benchmarks/results") -> None:
        self.output_dir = output_dir

    # ── Public API ──────────────────────────────────────────────────

    def aggregate_and_persist(
        self,
        instance_results: list[dict[str, Any]],
        run_id: str,
        slice_size: int,
        runs_per_cell: int,
        started_at: str,
        ended_at: str,
    ) -> dict[str, Any]:
        """Aggregate results, persist JSON + Markdown, return the data.

        Args:
            instance_results: List of per-instance dicts, each with:
                - instance_id: str
                - topology: str
                - model: str
                - run_results: list of dicts with resolved, cost_caching_off_usd,
                  cost_caching_on_usd
                - hitl_escalations: list of {cause, agent} dicts
            run_id: Unique run identifier.
            slice_size: Number of instances in the slice.
            runs_per_cell: Number of runs per (topology, instance) cell.
            started_at: ISO timestamp when the run started.
            ended_at: ISO timestamp when the run ended.

        Returns:
            The aggregated results dict that was persisted.
        """
        cells = self.compute_cells(instance_results)

        # Build the full results document
        results_doc: dict[str, Any] = {
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "slice_size": slice_size,
            "runs_per_cell": runs_per_cell,
            "cells": cells,
        }

        # Persist JSON
        self._persist_json(results_doc, run_id)

        # Persist Markdown
        self._persist_markdown(results_doc, run_id)

        return results_doc

    def compute_cells(
        self,
        instance_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compute per-cell statistics from instance results.

        Groups instance results by (topology, model), then computes
        mean/variance/95% CI on per-instance success rates and
        aggregate costs.

        Args:
            instance_results: List of per-instance result dicts.

        Returns:
            List of cell dicts, each with:
                topology, model, n, mean, variance, ci_low, ci_high,
                cost_caching_off_usd, cost_caching_on_usd, instances.
        """
        # Group by (topology, model)
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for ir in instance_results:
            key = (ir.get("topology", "unknown"), ir.get("model", "unknown"))
            groups.setdefault(key, []).append(ir)

        cells: list[dict[str, Any]] = []
        for (topology, model), group in sorted(groups.items()):
            # Compute per-instance success rates
            per_instance_means: list[float] = []
            cost_off_values: list[float] = []
            cost_on_values: list[float] = []
            instances_out: list[dict[str, Any]] = []

            for ir in group:
                run_results = ir.get("run_results", [])
                hitl_escalations = ir.get("hitl_escalations", [])

                # Per-instance success rate = fraction of runs resolved
                resolved_flags = [1.0 if r.get("resolved", False) else 0.0 for r in run_results]
                inst_mean = _mean(resolved_flags) if resolved_flags else 0.0
                per_instance_means.append(inst_mean)

                # Per-instance cost averages
                off_costs = [r.get("cost_caching_off_usd", 0.0) for r in run_results]
                on_costs = [r.get("cost_caching_on_usd", 0.0) for r in run_results]
                avg_off = _mean(off_costs) if off_costs else 0.0
                avg_on = _mean(on_costs) if on_costs else 0.0
                cost_off_values.append(avg_off)
                cost_on_values.append(avg_on)

                # Build instance output dict
                instance_out: dict[str, Any] = {
                    "instance_id": ir.get("instance_id", "unknown"),
                    "success_rate": inst_mean,
                    "avg_cost_caching_off_usd": avg_off,
                    "avg_cost_caching_on_usd": avg_on,
                    "hitl_escalations": hitl_escalations,
                }
                instances_out.append(instance_out)

            # Cell-level statistics
            n = len(per_instance_means)
            cell_mean = _mean(per_instance_means)
            cell_variance = _variance(per_instance_means)
            ci_low, ci_high = _ci_95(per_instance_means)

            # Cell-level cost averages (average of per-instance averages)
            cell_cost_off = _mean(cost_off_values) if cost_off_values else 0.0
            cell_cost_on = _mean(cost_on_values) if cost_on_values else 0.0

            cell: dict[str, Any] = {
                "topology": topology,
                "model": model,
                "n": n,
                "mean": cell_mean,
                "variance": cell_variance,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "cost_caching_off_usd": cell_cost_off,
                "cost_caching_on_usd": cell_cost_on,
                "instances": instances_out,
            }
            cells.append(cell)

        return cells

    # ── JSON persistence ────────────────────────────────────────────

    def _persist_json(self, results_doc: dict[str, Any], run_id: str) -> Path:
        """Persist the aggregated results to a JSON file.

        Args:
            results_doc: The full results document.
            run_id: Run identifier used in the filename.

        Returns:
            Path to the written JSON file.
        """
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_path = output_path / f"{run_id}.json"
        results_path.write_text(json.dumps(results_doc, indent=2, default=str))
        logger.info("Results JSON written to %s", results_path)
        return results_path

    # ── Markdown persistence ────────────────────────────────────────

    def _persist_markdown(self, results_doc: dict[str, Any], run_id: str) -> Path:
        """Persist the aggregated results to a Markdown file.

        The Markdown report contains a results table with columns:
        topology | model | n | mean | variance | 95% CI |
        cost_caching_off_usd | cost_caching_on_usd

        Args:
            results_doc: The full results document.
            run_id: Run identifier used in the filename.

        Returns:
            Path to the written Markdown file.
        """
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        md_path = output_path / f"{run_id}.md"
        md_content = self._render_markdown(results_doc)
        md_path.write_text(md_content)
        logger.info("Results Markdown written to %s", md_path)
        return md_path

    def _render_markdown(self, results_doc: dict[str, Any]) -> str:
        """Render the aggregated results as a Markdown report.

        Args:
            results_doc: The full results document.

        Returns:
            Markdown string.
        """
        lines: list[str] = []

        # Header
        run_id = results_doc.get("run_id", "unknown")
        lines.append(f"# SWE-bench Results — {run_id}")
        lines.append("")
        lines.append(f"- **Started:** {results_doc.get('started_at', 'N/A')}")
        lines.append(f"- **Ended:** {results_doc.get('ended_at', 'N/A')}")
        lines.append(f"- **Slice size:** {results_doc.get('slice_size', 'N/A')}")
        lines.append(f"- **Runs per cell:** {results_doc.get('runs_per_cell', 'N/A')}")
        lines.append("")

        # Results table
        lines.append("## Results by Cell")
        lines.append("")
        header = (
            "| topology | model | n | mean | variance | 95% CI | "
            "cost_caching_off_usd | cost_caching_on_usd |"
        )
        separator = (
            "| --- | --- | --- | --- | --- | --- | --- | --- |"
        )
        lines.append(header)
        lines.append(separator)

        for cell in results_doc.get("cells", []):
            ci_str = f"[{cell['ci_low']:.4f}, {cell['ci_high']:.4f}]"
            row = (
                f"| {cell['topology']} "
                f"| {cell['model']} "
                f"| {cell['n']} "
                f"| {cell['mean']:.4f} "
                f"| {cell['variance']:.6f} "
                f"| {ci_str} "
                f"| {cell['cost_caching_off_usd']:.4f} "
                f"| {cell['cost_caching_on_usd']:.4f} |"
            )
            lines.append(row)

        lines.append("")

        # Per-instance details
        lines.append("## Per-Instance Details")
        lines.append("")

        for cell in results_doc.get("cells", []):
            lines.append(f"### {cell['topology']} / {cell['model']}")
            lines.append("")
            inst_header = (
                "| instance_id | success_rate | "
                "avg_cost_caching_off_usd | avg_cost_caching_on_usd | hitl_escalations |"
            )
            inst_sep = "| --- | --- | --- | --- | --- |"
            lines.append(inst_header)
            lines.append(inst_sep)

            for inst in cell.get("instances", []):
                escalations = inst.get("hitl_escalations", [])
                esc_str = ", ".join(
                    f"{e['cause']}({e.get('agent', '?')})" for e in escalations
                ) or "—"
                row = (
                    f"| {inst['instance_id']} "
                    f"| {inst['success_rate']:.4f} "
                    f"| {inst['avg_cost_caching_off_usd']:.4f} "
                    f"| {inst['avg_cost_caching_on_usd']:.4f} "
                    f"| {esc_str} |"
                )
                lines.append(row)

            lines.append("")

        return "\n".join(lines)
