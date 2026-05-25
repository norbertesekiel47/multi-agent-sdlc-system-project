"""Results analysis: parse benchmark results JSON, generate Markdown tables + README.

M6 feature: m6-results-analysis

Provides:
- ResultsParser: reads and validates benchmark results JSON
- MarkdownTableGenerator: generates Markdown tables from results
- ReadmeGenerator: produces full README.md with tables and chart references

Expected behavior:
- README.md contains results table with success rate, cost, latency, retries, HITL escalations
- Charts generated and embedded in README
- Cost comparison with/without caching included
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.benchmarks.swebench.charts import ChartGenerator

logger = logging.getLogger(__name__)


class ResultsParser:
    """Parses benchmark results JSON into structured data.

    Usage::

        parser = ResultsParser()
        results = parser.parse_file(Path("benchmarks/results/run-id.json"))
        topologies = parser.get_topologies(results)
    """

    def parse_file(self, path: Path) -> dict[str, Any]:
        """Read and parse a benchmark results JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Parsed results dict.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON is invalid or missing required keys.
        """
        if not path.exists():
            msg = f"Results file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {path}: {exc}"
            raise ValueError(msg) from exc

        return self._validate(data)

    def parse_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse a results dict directly (validates structure).

        Args:
            data: Benchmark results dict.

        Returns:
            Validated results dict.

        Raises:
            ValueError: If required keys are missing.
        """
        return self._validate(data)

    def get_topologies(self, results: dict[str, Any]) -> list[str]:
        """Extract topology names from results.

        Args:
            results: Parsed benchmark results dict.

        Returns:
            List of topology name strings.
        """
        return [c.get("topology", "unknown") for c in results.get("cells", [])]

    def get_instance_ids(self, results: dict[str, Any]) -> list[str]:
        """Extract unique instance IDs across all cells.

        Args:
            results: Parsed benchmark results dict.

        Returns:
            Sorted list of unique instance ID strings.
        """
        ids: set[str] = set()
        for cell in results.get("cells", []):
            for inst in cell.get("instances", []):
                iid = inst.get("instance_id", "")
                if iid:
                    ids.add(iid)
        return sorted(ids)

    # ── Private ──────────────────────────────────────────────────────

    def _validate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate the structure of the results dict.

        Args:
            data: Raw parsed data.

        Returns:
            Validated data.

        Raises:
            ValueError: If required keys are missing.
        """
        if "cells" not in data:
            msg = "Results JSON missing required 'cells' key"
            raise ValueError(msg)
        return data


class MarkdownTableGenerator:
    """Generates Markdown tables from benchmark results.

    Usage::

        gen = MarkdownTableGenerator()
        table = gen.generate_results_table(results)
        cost_table = gen.generate_cost_comparison_table(results)
    """

    def generate_results_table(self, results: dict[str, Any]) -> str:
        """Generate the main results summary table.

        Columns: topology, success rate, 95% CI, avg cost, avg latency,
        avg retries, HITL escalations.

        Args:
            results: Parsed benchmark results dict.

        Returns:
            Markdown table string.
        """
        cells = results.get("cells", [])
        if not cells:
            return (
                "| topology | success rate | 95% CI | avg cost (USD) "
                "| avg latency (s) | avg retries | HITL escalations |\n"
                "| --- | --- | --- | --- | --- | --- | --- |\n"
                "| — | — | — | — | — | — | — |\n"
            )

        lines: list[str] = []
        header = (
            "| topology | success rate | 95% CI | avg cost (USD) "
            "| avg latency (s) | avg retries | HITL escalations |"
        )
        separator = "| --- | --- | --- | --- | --- | --- | --- |"
        lines.append(header)
        lines.append(separator)

        for cell in cells:
            topology = cell.get("topology", "unknown")
            mean = cell.get("mean", 0.0)
            ci_low = cell.get("ci_low", 0.0)
            ci_high = cell.get("ci_high", 0.0)
            cost_on = cell.get("cost_caching_on_usd", 0.0)
            duration = cell.get("avg_duration_seconds", 0.0)
            retries = cell.get("avg_retry_count", 0.0)

            # HITL escalation summary
            esc_summary = cell.get("hitl_escalation_summary", {})
            esc_str = (
                ", ".join(f"{k}: {v}" for k, v in esc_summary.items())
                if esc_summary
                else "—"
            )

            row = (
                f"| {topology} "
                f"| {mean:.1%} "
                f"| [{ci_low:.1%}, {ci_high:.1%}] "
                f"| ${cost_on:.4f} "
                f"| {duration:.1f}s "
                f"| {retries:.2f} "
                f"| {esc_str} |"
            )
            lines.append(row)

        return "\n".join(lines)

    def generate_cost_comparison_table(self, results: dict[str, Any]) -> str:
        """Generate a cost comparison table: caching ON vs OFF.

        Columns: topology, cost w/o caching, cost w/ caching, savings, savings %.

        Args:
            results: Parsed benchmark results dict.

        Returns:
            Markdown table string.
        """
        cells = results.get("cells", [])
        if not cells:
            return (
                "| topology | cost w/o caching | cost w/ caching "
                "| savings | savings % |\n"
                "| --- | --- | --- | --- | --- |\n"
                "| — | — | — | — | — |\n"
            )

        lines: list[str] = []
        header = (
            "| topology | cost w/o caching (USD) "
            "| cost w/ caching (USD) | savings (USD) | savings % |"
        )
        separator = "| --- | --- | --- | --- | --- |"
        lines.append(header)
        lines.append(separator)

        for cell in cells:
            topology = cell.get("topology", "unknown")
            cost_off = cell.get("cost_caching_off_usd", 0.0)
            cost_on = cell.get("cost_caching_on_usd", 0.0)
            savings = cost_off - cost_on
            savings_pct = (savings / cost_off * 100) if cost_off > 0 else 0.0

            row = (
                f"| {topology} "
                f"| ${cost_off:.4f} "
                f"| ${cost_on:.4f} "
                f"| ${savings:.4f} "
                f"| {savings_pct:.1f}% |"
            )
            lines.append(row)

        return "\n".join(lines)

    def generate_per_instance_table(self, results: dict[str, Any]) -> str:
        """Generate a per-instance outcomes table.

        Columns: instance_id, topology, success rate, cost w/ caching,
        cost w/o caching, latency, retries, HITL escalations.

        Args:
            results: Parsed benchmark results dict.

        Returns:
            Markdown table string.
        """
        cells = results.get("cells", [])
        if not cells:
            return ""

        lines: list[str] = []
        header = (
            "| instance_id | topology | success rate "
            "| cost w/ caching | cost w/o caching "
            "| latency (s) | retries | HITL |"
        )
        separator = "| --- | --- | --- | --- | --- | --- | --- | --- |"
        lines.append(header)
        lines.append(separator)

        for cell in cells:
            topology = cell.get("topology", "unknown")
            for inst in cell.get("instances", []):
                iid = inst.get("instance_id", "unknown")
                sr = inst.get("success_rate", 0.0)
                cost_on = inst.get("avg_cost_caching_on_usd", 0.0)
                cost_off = inst.get("avg_cost_caching_off_usd", 0.0)
                duration = inst.get("avg_duration_seconds", 0.0)
                retries = inst.get("avg_retry_count", 0.0)

                # HITL escalations
                escalations = inst.get("hitl_escalations", [])
                if escalations:
                    esc_str = ", ".join(
                        e.get("cause", "?") for e in escalations
                    )
                else:
                    esc_str = "—"

                row = (
                    f"| {iid} "
                    f"| {topology} "
                    f"| {sr:.1%} "
                    f"| ${cost_on:.4f} "
                    f"| ${cost_off:.4f} "
                    f"| {duration:.1f}s "
                    f"| {retries:.2f} "
                    f"| {esc_str} |"
                )
                lines.append(row)

        return "\n".join(lines)

    def generate_hitl_escalation_table(self, results: dict[str, Any]) -> str:
        """Generate an HITL escalation summary table.

        Columns: topology, cause, count.

        Args:
            results: Parsed benchmark results dict.

        Returns:
            Markdown table string.
        """
        cells = results.get("cells", [])
        if not cells:
            return ""

        lines: list[str] = []
        header = "| topology | escalation cause | count |"
        separator = "| --- | --- | --- |"
        lines.append(header)
        lines.append(separator)

        has_data = False
        for cell in cells:
            topology = cell.get("topology", "unknown")
            esc_summary = cell.get("hitl_escalation_summary", {})
            for cause, count in esc_summary.items():
                lines.append(f"| {topology} | {cause} | {count} |")
                has_data = True

        if not has_data:
            lines.append("| — | No HITL escalations recorded | — |")

        return "\n".join(lines)


class ReadmeGenerator:
    """Generates README.md with results tables and chart references.

    Usage::

        gen = ReadmeGenerator(output_dir="benchmarks")
        readme_path = gen.generate_readme(results)
    """

    def __init__(self, *, output_dir: str = "benchmarks") -> None:
        self.output_dir = output_dir
        self.chart_gen = ChartGenerator(output_dir=f"{output_dir}/charts")
        self.table_gen = MarkdownTableGenerator()

    def generate_readme(self, results: dict[str, Any]) -> Path:
        """Generate README.md with tables, chart images, and cost comparison.

        Args:
            results: Parsed benchmark results dict.

        Returns:
            Path to the generated README.md file.
        """
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate charts first
        chart_paths = self.chart_gen.generate_all_charts(results)

        # Build README content
        lines: list[str] = []

        # ── Title ───────────────────────────────────────────────
        lines.append("# SDLC-Swarm Benchmark Results")
        lines.append("")

        # ── Run metadata ──────────────────────────────────────
        run_id = results.get("run_id", "unknown")
        started_at = results.get("started_at", "N/A")
        ended_at = results.get("ended_at", "N/A")
        slice_size = results.get("slice_size", 0)
        runs_per_cell = results.get("runs_per_cell", 0)

        lines.append("## Run Metadata")
        lines.append("")
        lines.append(f"- **Run ID:** `{run_id}`")
        lines.append(f"- **Started:** {started_at}")
        lines.append(f"- **Ended:** {ended_at}")
        lines.append(f"- **Slice size:** {slice_size} instances")
        lines.append(f"- **Runs per cell:** {runs_per_cell}")
        lines.append("")

        # ── Results summary table ─────────────────────────────
        cells = results.get("cells", [])
        if not cells:
            lines.append("## Results Summary")
            lines.append("")
            lines.append("No benchmark results available.")
            lines.append("")
        else:
            lines.append("## Results Summary")
            lines.append("")
            lines.append(
                "Success rate, cost, latency, retries, and HITL escalations "
                "per topology."
            )
            lines.append("")
            lines.append(self.table_gen.generate_results_table(results))
            lines.append("")

            # ── Charts ──────────────────────────────────────────
            lines.append("## Charts")
            lines.append("")

            # Bar chart
            lines.append("### Success Rate by Topology")
            lines.append("")
            bar_path = chart_paths[0] if len(chart_paths) > 0 else None
            if bar_path:
                rel_path = self._relative_path(bar_path, output_path)
                lines.append(f"![Success Rate by Topology]({rel_path})")
            lines.append("")

            # Line chart
            lines.append("### Cost vs. Quality")
            lines.append("")
            line_path = chart_paths[1] if len(chart_paths) > 1 else None
            if line_path:
                rel_path = self._relative_path(line_path, output_path)
                lines.append(f"![Cost vs Quality]({rel_path})")
            lines.append("")

            # Heatmap
            lines.append("### Per-Instance Outcomes Heatmap")
            lines.append("")
            heatmap_path = chart_paths[2] if len(chart_paths) > 2 else None
            if heatmap_path:
                rel_path = self._relative_path(heatmap_path, output_path)
                lines.append(f"![Per-Instance Heatmap]({rel_path})")
            lines.append("")

            # ── Cost comparison ────────────────────────────────
            lines.append("## Cost Comparison: Caching ON vs OFF")
            lines.append("")
            lines.append(
                "Prompt caching reduces cost by reusing cached token blocks "
                "for Coder/Reviewer repo-context. The table below shows the "
                "average per-instance cost with and without caching, and the "
                "savings."
            )
            lines.append("")
            lines.append(self.table_gen.generate_cost_comparison_table(results))
            lines.append("")

            # ── HITL escalation summary ────────────────────────
            lines.append("## HITL Escalation Summary")
            lines.append("")
            lines.append(
                "Cause-tagged HITL escalation counts per topology. "
                "Escalations are triggered deterministically per §2.9."
            )
            lines.append("")
            lines.append(self.table_gen.generate_hitl_escalation_table(results))
            lines.append("")

            # ── Per-instance details ───────────────────────────
            lines.append("## Per-Instance Details")
            lines.append("")
            lines.append(self.table_gen.generate_per_instance_table(results))
            lines.append("")

        # Write README
        readme_path = output_path / "README.md"
        readme_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("README generated at %s", readme_path)
        return readme_path

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _relative_path(path: Path, base: Path) -> Path:
        """Compute a relative path, falling back to absolute if not relative.

        Args:
            path: The target path.
            base: The base directory.

        Returns:
            Relative path if possible, else the original path.
        """
        if path.is_relative_to(base):
            return path.relative_to(base)
        return path
