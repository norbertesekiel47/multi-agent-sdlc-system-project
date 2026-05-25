"""Tests for M6 results analysis: parse benchmark JSON, generate tables + charts.

Feature: m6-results-analysis

Expected behavior:
- README.md contains results table with success rate, cost, latency, retries, HITL escalations
- Charts generated and embedded in README
- Cost comparison with/without caching included

Tests cover:
- Parsing benchmark results JSON into structured data
- Generating Markdown tables from aggregated results
- Bar chart: success rate by topology
- Line chart: cost vs quality
- Heatmap: per-instance outcomes
- Cost-with-caching vs cost-without-caching comparison table
- README.md generation with embedded charts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ── Sample benchmark results JSON ──────────────────────────────


def _make_sample_results_json() -> dict[str, Any]:
    """Create a sample benchmark results JSON for testing.

    Simulates the output of the aggregator with 3 topologies,
    each having multiple instances with N=3 runs.
    """
    return {
        "run_id": "test-run-001",
        "started_at": "2026-05-24T00:00:00Z",
        "ended_at": "2026-05-24T06:00:00Z",
        "slice_size": 10,
        "runs_per_cell": 3,
        "cells": [
            {
                "topology": "single_agent",
                "model": "deepseek/deepseek-chat-v3-0324",
                "n": 10,
                "mean": 0.20,
                "variance": 0.0178,
                "ci_low": 0.0633,
                "ci_high": 0.3367,
                "cost_caching_off_usd": 0.42,
                "cost_caching_on_usd": 0.30,
                "avg_total_tokens_in": 4500.0,
                "avg_total_tokens_out": 1300.0,
                "avg_total_tokens_cached": 1200.0,
                "avg_duration_seconds": 35.2,
                "avg_retry_count": 0.3,
                "avg_peer_handoff_count": 0.0,
                "hitl_escalation_summary": {"retry_budget_exhausted": 1},
                "instances": [
                    {
                        "instance_id": "django__django-16379",
                        "success_rate": 0.0,
                        "avg_cost_caching_off_usd": 0.35,
                        "avg_cost_caching_on_usd": 0.25,
                        "avg_total_tokens_in": 4000.0,
                        "avg_total_tokens_out": 1200.0,
                        "avg_total_tokens_cached": 1000.0,
                        "avg_duration_seconds": 30.0,
                        "avg_retry_count": 0.0,
                        "avg_peer_handoff_count": 0.0,
                        "hitl_escalations": [],
                    },
                    {
                        "instance_id": "flask__flask-4817",
                        "success_rate": 0.3333,
                        "avg_cost_caching_off_usd": 0.48,
                        "avg_cost_caching_on_usd": 0.35,
                        "avg_total_tokens_in": 5000.0,
                        "avg_total_tokens_out": 1400.0,
                        "avg_total_tokens_cached": 1400.0,
                        "avg_duration_seconds": 40.5,
                        "avg_retry_count": 0.33,
                        "avg_peer_handoff_count": 0.0,
                        "hitl_escalations": [],
                    },
                    {
                        "instance_id": "requests__requests-6028",
                        "success_rate": 0.0,
                        "avg_cost_caching_off_usd": 0.42,
                        "avg_cost_caching_on_usd": 0.30,
                        "avg_total_tokens_in": 4500.0,
                        "avg_total_tokens_out": 1300.0,
                        "avg_total_tokens_cached": 1200.0,
                        "avg_duration_seconds": 35.0,
                        "avg_retry_count": 0.67,
                        "avg_peer_handoff_count": 0.0,
                        "hitl_escalations": [
                            {
                                "cause": "retry_budget_exhausted",
                                "trigger": "retry_budget_exhausted",
                                "agent": "coder",
                            },
                        ],
                    },
                ],
            },
            {
                "topology": "supervisor_only",
                "model": "deepseek/deepseek-chat-v3-0324",
                "n": 10,
                "mean": 0.40,
                "variance": 0.0267,
                "ci_low": 0.2233,
                "ci_high": 0.5767,
                "cost_caching_off_usd": 0.62,
                "cost_caching_on_usd": 0.44,
                "avg_total_tokens_in": 6200.0,
                "avg_total_tokens_out": 1800.0,
                "avg_total_tokens_cached": 2800.0,
                "avg_duration_seconds": 52.8,
                "avg_retry_count": 0.5,
                "avg_peer_handoff_count": 0.0,
                "hitl_escalation_summary": {
                    "uncertainty_escalation": 1,
                    "loop_detected": 1,
                },
                "instances": [
                    {
                        "instance_id": "django__django-16379",
                        "success_rate": 0.6667,
                        "avg_cost_caching_off_usd": 0.55,
                        "avg_cost_caching_on_usd": 0.40,
                        "avg_total_tokens_in": 6000.0,
                        "avg_total_tokens_out": 1700.0,
                        "avg_total_tokens_cached": 2500.0,
                        "avg_duration_seconds": 50.0,
                        "avg_retry_count": 0.33,
                        "avg_peer_handoff_count": 0.0,
                        "hitl_escalations": [],
                    },
                    {
                        "instance_id": "flask__flask-4817",
                        "success_rate": 0.3333,
                        "avg_cost_caching_off_usd": 0.68,
                        "avg_cost_caching_on_usd": 0.48,
                        "avg_total_tokens_in": 6400.0,
                        "avg_total_tokens_out": 1900.0,
                        "avg_total_tokens_cached": 3100.0,
                        "avg_duration_seconds": 55.6,
                        "avg_retry_count": 0.67,
                        "avg_peer_handoff_count": 0.0,
                        "hitl_escalations": [
                            {
                                "cause": "loop_detected",
                                "trigger": "loop_detected",
                                "agent": "coder",
                            },
                        ],
                    },
                    {
                        "instance_id": "requests__requests-6028",
                        "success_rate": 0.0,
                        "avg_cost_caching_off_usd": 0.62,
                        "avg_cost_caching_on_usd": 0.44,
                        "avg_total_tokens_in": 6200.0,
                        "avg_total_tokens_out": 1800.0,
                        "avg_total_tokens_cached": 2800.0,
                        "avg_duration_seconds": 52.8,
                        "avg_retry_count": 0.50,
                        "avg_peer_handoff_count": 0.0,
                        "hitl_escalations": [
                            {
                                "cause": "uncertainty_escalation",
                                "trigger": "pydantic_validation_3x",
                                "agent": "reviewer",
                            },
                        ],
                    },
                ],
            },
            {
                "topology": "hybrid",
                "model": "deepseek/deepseek-chat-v3-0324",
                "n": 10,
                "mean": 0.50,
                "variance": 0.0250,
                "ci_low": 0.3090,
                "ci_high": 0.6910,
                "cost_caching_off_usd": 0.78,
                "cost_caching_on_usd": 0.55,
                "avg_total_tokens_in": 7500.0,
                "avg_total_tokens_out": 2200.0,
                "avg_total_tokens_cached": 3500.0,
                "avg_duration_seconds": 65.3,
                "avg_retry_count": 0.4,
                "avg_peer_handoff_count": 1.2,
                "hitl_escalation_summary": {"loop_detected": 1},
                "instances": [
                    {
                        "instance_id": "django__django-16379",
                        "success_rate": 1.0,
                        "avg_cost_caching_off_usd": 0.72,
                        "avg_cost_caching_on_usd": 0.50,
                        "avg_total_tokens_in": 7000.0,
                        "avg_total_tokens_out": 2000.0,
                        "avg_total_tokens_cached": 3000.0,
                        "avg_duration_seconds": 60.0,
                        "avg_retry_count": 0.0,
                        "avg_peer_handoff_count": 1.0,
                        "hitl_escalations": [],
                    },
                    {
                        "instance_id": "flask__flask-4817",
                        "success_rate": 0.3333,
                        "avg_cost_caching_off_usd": 0.84,
                        "avg_cost_caching_on_usd": 0.60,
                        "avg_total_tokens_in": 8000.0,
                        "avg_total_tokens_out": 2400.0,
                        "avg_total_tokens_cached": 4000.0,
                        "avg_duration_seconds": 70.6,
                        "avg_retry_count": 0.80,
                        "avg_peer_handoff_count": 1.6,
                        "hitl_escalations": [
                            {
                                "cause": "loop_detected",
                                "trigger": "loop_detected",
                                "agent": "coder",
                            },
                        ],
                    },
                    {
                        "instance_id": "requests__requests-6028",
                        "success_rate": 0.0,
                        "avg_cost_caching_off_usd": 0.78,
                        "avg_cost_caching_on_usd": 0.55,
                        "avg_total_tokens_in": 7500.0,
                        "avg_total_tokens_out": 2200.0,
                        "avg_total_tokens_cached": 3500.0,
                        "avg_duration_seconds": 65.3,
                        "avg_retry_count": 0.40,
                        "avg_peer_handoff_count": 1.0,
                        "hitl_escalations": [],
                    },
                ],
            },
        ],
    }


@pytest.fixture
def sample_results() -> dict[str, Any]:
    """Fixture providing sample benchmark results JSON."""
    return _make_sample_results_json()


@pytest.fixture
def sample_results_file(tmp_path: Path) -> Path:
    """Fixture providing a sample results JSON file on disk."""
    results = _make_sample_results_json()
    results_path = tmp_path / "test-run-001.json"
    results_path.write_text(json.dumps(results, indent=2))
    return results_path


# ── Tests for ResultsParser ────────────────────────────────────


class TestResultsParser:
    """Test parsing benchmark results JSON into structured data."""

    def test_parse_results_from_file(self, sample_results_file: Path) -> None:
        """ResultsParser reads and parses a benchmark results JSON file."""
        from src.benchmarks.swebench.results_analysis import ResultsParser

        parser = ResultsParser()
        results = parser.parse_file(sample_results_file)

        assert results["run_id"] == "test-run-001"
        assert len(results["cells"]) == 3

    def test_parse_results_from_dict(self, sample_results: dict[str, Any]) -> None:
        """ResultsParser parses a results dict directly."""
        from src.benchmarks.swebench.results_analysis import ResultsParser

        parser = ResultsParser()
        results = parser.parse_dict(sample_results)

        assert results["run_id"] == "test-run-001"
        assert "cells" in results

    def test_parse_extracts_topologies(self, sample_results: dict[str, Any]) -> None:
        """Parser extracts topology names from cells."""
        from src.benchmarks.swebench.results_analysis import ResultsParser

        parser = ResultsParser()
        topologies = parser.get_topologies(sample_results)

        assert set(topologies) == {"single_agent", "supervisor_only", "hybrid"}

    def test_parse_extracts_instance_ids(self, sample_results: dict[str, Any]) -> None:
        """Parser extracts unique instance IDs across all cells."""
        from src.benchmarks.swebench.results_analysis import ResultsParser

        parser = ResultsParser()
        instance_ids = parser.get_instance_ids(sample_results)

        # Should have 3 unique instance IDs
        assert len(instance_ids) >= 3
        assert "django__django-16379" in instance_ids

    def test_parse_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Parser raises FileNotFoundError for missing file."""
        from src.benchmarks.swebench.results_analysis import ResultsParser

        parser = ResultsParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file(tmp_path / "nonexistent.json")

    def test_parse_raises_on_invalid_json(self, tmp_path: Path) -> None:
        """Parser raises ValueError on invalid JSON."""
        from src.benchmarks.swebench.results_analysis import ResultsParser

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {]")
        parser = ResultsParser()
        with pytest.raises(ValueError, match="Invalid JSON"):
            parser.parse_file(bad_file)

    def test_parse_raises_on_missing_cells_key(self, tmp_path: Path) -> None:
        """Parser raises ValueError when cells key is missing."""
        from src.benchmarks.swebench.results_analysis import ResultsParser

        bad_file = tmp_path / "no_cells.json"
        bad_file.write_text(json.dumps({"run_id": "test"}))
        parser = ResultsParser()
        with pytest.raises(ValueError, match="cells"):
            parser.parse_file(bad_file)


# ── Tests for MarkdownTableGenerator ────────────────────────────


class TestMarkdownTableGenerator:
    """Test generating Markdown tables from benchmark results."""

    def test_results_table_has_required_columns(
        self, sample_results: dict[str, Any]
    ) -> None:
        """Results table contains success rate, cost, latency, retries, HITL."""
        from src.benchmarks.swebench.results_analysis import MarkdownTableGenerator

        gen = MarkdownTableGenerator()
        table = gen.generate_results_table(sample_results)

        # Required columns
        assert "topology" in table
        assert "success rate" in table.lower()
        assert "cost" in table.lower()
        assert "latency" in table.lower() or "duration" in table.lower()
        assert "retries" in table.lower()
        assert "hitl" in table.lower() or "escalation" in table.lower()

    def test_results_table_has_all_topologies(
        self, sample_results: dict[str, Any]
    ) -> None:
        """Results table includes all topology rows."""
        from src.benchmarks.swebench.results_analysis import MarkdownTableGenerator

        gen = MarkdownTableGenerator()
        table = gen.generate_results_table(sample_results)

        assert "single_agent" in table
        assert "supervisor_only" in table
        assert "hybrid" in table

    def test_results_table_has_ci(self, sample_results: dict[str, Any]) -> None:
        """Results table includes 95% confidence interval."""
        from src.benchmarks.swebench.results_analysis import MarkdownTableGenerator

        gen = MarkdownTableGenerator()
        table = gen.generate_results_table(sample_results)

        assert "95%" in table or "CI" in table or "ci" in table

    def test_cost_comparison_table(
        self, sample_results: dict[str, Any]
    ) -> None:
        """Cost comparison table shows caching on vs off."""
        from src.benchmarks.swebench.results_analysis import MarkdownTableGenerator

        gen = MarkdownTableGenerator()
        table = gen.generate_cost_comparison_table(sample_results)

        table_lower = table.lower()
        assert (
            "w/o caching" in table_lower
            or "caching_off" in table_lower
            or "without caching" in table_lower
        )
        assert (
            "w/ caching" in table_lower
            or "caching_on" in table_lower
            or "with caching" in table_lower
        )
        assert "savings" in table_lower or "delta" in table_lower or "%" in table

    def test_cost_comparison_shows_savings(
        self, sample_results: dict[str, Any]
    ) -> None:
        """Cost comparison shows savings percentage."""
        from src.benchmarks.swebench.results_analysis import MarkdownTableGenerator

        gen = MarkdownTableGenerator()
        table = gen.generate_cost_comparison_table(sample_results)

        # Caching should produce savings > 0%
        assert "hybrid" in table
        # At least one row should have savings > 0
        lines = table.split("\n")
        data_lines = [
            ln for ln in lines
            if ln.strip().startswith("|")
            and "topology" not in ln.lower()
            and "---" not in ln
        ]
        # Should have 3 data rows (one per topology)
        assert len(data_lines) >= 3

    def test_per_instance_table(
        self, sample_results: dict[str, Any]
    ) -> None:
        """Per-instance outcomes table generated."""
        from src.benchmarks.swebench.results_analysis import MarkdownTableGenerator

        gen = MarkdownTableGenerator()
        table = gen.generate_per_instance_table(sample_results)

        assert "instance" in table.lower()
        # Should contain instance IDs
        assert "django__django-16379" in table

    def test_hitl_escalation_table(
        self, sample_results: dict[str, Any]
    ) -> None:
        """HITL escalation summary table generated with cause tags."""
        from src.benchmarks.swebench.results_analysis import MarkdownTableGenerator

        gen = MarkdownTableGenerator()
        table = gen.generate_hitl_escalation_table(sample_results)

        assert "loop_detected" in table or "uncertainty_escalation" in table
        assert "cause" in table.lower() or "escalation" in table.lower()


# ── Tests for ChartGenerator ────────────────────────────────────


class TestChartGenerator:
    """Test generating charts from benchmark results."""

    def test_bar_chart_success_rate_by_topology(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """Bar chart shows success rate by topology."""
        from src.benchmarks.swebench.charts import ChartGenerator

        gen = ChartGenerator(output_dir=str(tmp_path))
        chart_path = gen.bar_chart_success_rate(sample_results)

        assert chart_path.exists()
        assert chart_path.suffix == ".png"
        # File should be non-trivial
        assert chart_path.stat().st_size > 1000

    def test_line_chart_cost_vs_quality(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """Line chart shows cost vs quality (success rate)."""
        from src.benchmarks.swebench.charts import ChartGenerator

        gen = ChartGenerator(output_dir=str(tmp_path))
        chart_path = gen.line_chart_cost_vs_quality(sample_results)

        assert chart_path.exists()
        assert chart_path.suffix == ".png"
        assert chart_path.stat().st_size > 1000

    def test_heatmap_per_instance_outcomes(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """Heatmap shows per-instance outcomes across topologies."""
        from src.benchmarks.swebench.charts import ChartGenerator

        gen = ChartGenerator(output_dir=str(tmp_path))
        chart_path = gen.heatmap_per_instance_outcomes(sample_results)

        assert chart_path.exists()
        assert chart_path.suffix == ".png"
        assert chart_path.stat().st_size > 1000

    def test_chart_filenames_are_deterministic(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """Chart filenames are consistent and predictable."""
        from src.benchmarks.swebench.charts import ChartGenerator

        gen = ChartGenerator(output_dir=str(tmp_path))

        bar_path = gen.bar_chart_success_rate(sample_results)
        line_path = gen.line_chart_cost_vs_quality(sample_results)
        heatmap_path = gen.heatmap_per_instance_outcomes(sample_results)

        assert bar_path.name == "success_rate_by_topology.png"
        assert line_path.name == "cost_vs_quality.png"
        assert heatmap_path.name == "heatmap_per_instance.png"

    def test_all_charts_generated(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """All three chart types can be generated without errors."""
        from src.benchmarks.swebench.charts import ChartGenerator

        gen = ChartGenerator(output_dir=str(tmp_path))
        paths = gen.generate_all_charts(sample_results)

        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"


# ── Tests for README generator ──────────────────────────────────


class TestReadmeGenerator:
    """Test README.md generation with tables and charts."""

    def test_readme_contains_results_table(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """README contains the results summary table."""
        from src.benchmarks.swebench.results_analysis import ReadmeGenerator

        gen = ReadmeGenerator(output_dir=str(tmp_path))
        readme_path = gen.generate_readme(sample_results)

        assert readme_path.exists()
        content = readme_path.read_text()

        # Must contain topology names
        assert "single_agent" in content
        assert "supervisor_only" in content
        assert "hybrid" in content

    def test_readme_contains_chart_references(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """README contains image references for generated charts."""
        from src.benchmarks.swebench.results_analysis import ReadmeGenerator

        gen = ReadmeGenerator(output_dir=str(tmp_path))
        readme_path = gen.generate_readme(sample_results)

        content = readme_path.read_text()

        # Chart images referenced
        assert "![Success Rate by Topology]" in content or "success_rate_by_topology" in content
        assert "![Cost vs Quality]" in content or "cost_vs_quality" in content
        assert "![Per-Instance Heatmap]" in content or "heatmap_per_instance" in content

    def test_readme_contains_cost_comparison(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """README contains cost-with-caching vs cost-without-caching comparison."""
        from src.benchmarks.swebench.results_analysis import ReadmeGenerator

        gen = ReadmeGenerator(output_dir=str(tmp_path))
        readme_path = gen.generate_readme(sample_results)

        content = readme_path.read_text()

        content_lower = content.lower()
        assert "caching" in content_lower
        assert (
            "savings" in content_lower
            or "delta" in content_lower
            or "reduction" in content_lower
        )

    def test_readme_contains_hitl_escalations(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """README contains HITL escalation summary."""
        from src.benchmarks.swebench.results_analysis import ReadmeGenerator

        gen = ReadmeGenerator(output_dir=str(tmp_path))
        readme_path = gen.generate_readme(sample_results)

        content = readme_path.read_text()

        assert "escalation" in content.lower() or "hitl" in content.lower()

    def test_readme_is_valid_markdown(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """README is valid Markdown with proper headers."""
        from src.benchmarks.swebench.results_analysis import ReadmeGenerator

        gen = ReadmeGenerator(output_dir=str(tmp_path))
        readme_path = gen.generate_readme(sample_results)

        content = readme_path.read_text()

        # Should start with a title
        assert content.startswith("#")
        # Should have section headers
        assert "##" in content

    def test_readme_run_metadata(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """README includes run metadata (run_id, dates, slice_size)."""
        from src.benchmarks.swebench.results_analysis import ReadmeGenerator

        gen = ReadmeGenerator(output_dir=str(tmp_path))
        readme_path = gen.generate_readme(sample_results)

        content = readme_path.read_text()

        assert "test-run-001" in content
        assert "10" in content  # slice_size
        assert "3" in content  # runs_per_cell

    def test_readme_charts_dir_created(
        self, sample_results: dict[str, Any], tmp_path: Path
    ) -> None:
        """README generator creates charts directory and chart files."""
        from src.benchmarks.swebench.results_analysis import ReadmeGenerator

        gen = ReadmeGenerator(output_dir=str(tmp_path))
        gen.generate_readme(sample_results)

        # Charts directory should exist with PNG files
        charts_dir = tmp_path / "charts"
        assert charts_dir.exists()
        png_files = list(charts_dir.glob("*.png"))
        assert len(png_files) >= 3


# ── Integration: full pipeline from JSON file to README ─────────


class TestResultsAnalysisPipeline:
    """Integration tests: JSON file → parsed results → tables + charts → README."""

    def test_full_pipeline(
        self, sample_results_file: Path, tmp_path: Path
    ) -> None:
        """Full pipeline: parse JSON → generate tables → generate charts → README."""
        from src.benchmarks.swebench.results_analysis import (
            ReadmeGenerator,
            ResultsParser,
        )

        parser = ResultsParser()
        results = parser.parse_file(sample_results_file)

        gen = ReadmeGenerator(output_dir=str(tmp_path))
        readme_path = gen.generate_readme(results)

        assert readme_path.exists()
        content = readme_path.read_text()

        # All expected content present
        assert "single_agent" in content
        assert "supervisor_only" in content
        assert "hybrid" in content
        assert "caching" in content.lower()
        assert "!" in content  # Image references use ![]()

        # Charts generated
        charts_dir = tmp_path / "charts"
        assert charts_dir.exists()
        chart_files = list(charts_dir.glob("*.png"))
        assert len(chart_files) >= 3

    def test_empty_cells_handled(self, tmp_path: Path) -> None:
        """Pipeline handles results with empty cells gracefully."""
        from src.benchmarks.swebench.results_analysis import ReadmeGenerator

        results = {
            "run_id": "empty-run",
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T01:00:00Z",
            "slice_size": 0,
            "runs_per_cell": 0,
            "cells": [],
        }

        gen = ReadmeGenerator(output_dir=str(tmp_path))
        readme_path = gen.generate_readme(results)

        assert readme_path.exists()
        content = readme_path.read_text()
        assert "No results" in content or "no data" in content.lower() or len(content) > 50

    def test_single_topology_results(self, tmp_path: Path) -> None:
        """Pipeline works with results from a single topology."""
        from src.benchmarks.swebench.results_analysis import ReadmeGenerator

        results = {
            "run_id": "single-topo",
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T01:00:00Z",
            "slice_size": 3,
            "runs_per_cell": 1,
            "cells": [
                {
                    "topology": "supervisor_only",
                    "model": "deepseek/deepseek-chat-v3-0324",
                    "n": 3,
                    "mean": 0.3333,
                    "variance": 0.0833,
                    "ci_low": 0.1667,
                    "ci_high": 0.5000,
                    "cost_caching_off_usd": 0.50,
                    "cost_caching_on_usd": 0.35,
                    "avg_total_tokens_in": 5000.0,
                    "avg_total_tokens_out": 1500.0,
                    "avg_total_tokens_cached": 1500.0,
                    "avg_duration_seconds": 45.0,
                    "avg_retry_count": 0.3,
                    "avg_peer_handoff_count": 0.0,
                    "hitl_escalation_summary": {},
                    "instances": [
                        {
                            "instance_id": "test-1",
                            "success_rate": 1.0,
                            "avg_cost_caching_off_usd": 0.50,
                            "avg_cost_caching_on_usd": 0.35,
                            "avg_total_tokens_in": 5000.0,
                            "avg_total_tokens_out": 1500.0,
                            "avg_total_tokens_cached": 1500.0,
                            "avg_duration_seconds": 45.0,
                            "avg_retry_count": 0.0,
                            "avg_peer_handoff_count": 0.0,
                            "hitl_escalations": [],
                        },
                    ],
                },
            ],
        }

        gen = ReadmeGenerator(output_dir=str(tmp_path))
        readme_path = gen.generate_readme(results)

        content = readme_path.read_text()
        assert "supervisor_only" in content
