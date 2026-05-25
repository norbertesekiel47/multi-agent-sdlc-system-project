"""Chart generation for benchmark results analysis (M6).

Generates three chart types from benchmark results JSON:
- Bar chart: success rate by topology
- Line chart: cost vs quality (success rate)
- Heatmap: per-instance outcomes across topologies

Charts are saved as PNG files for embedding in README.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless generation

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

# Chart styling constants
_FIGSIZE_WIDE = (10, 6)
_FIGSIZE_SQUARE = (8, 8)
_DPI = 150
_FONT_SIZE = 11
_TITLE_SIZE = 14
_TICK_SIZE = 10
_COLOR_PALETTE = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
_HEATMAP_CMAP = "RdYlGn"  # Red (fail) → Yellow (partial) → Green (pass)


class ChartGenerator:
    """Generates charts from benchmark results.

    Usage::

        gen = ChartGenerator(output_dir="benchmarks/charts")
        paths = gen.generate_all_charts(results_dict)
    """

    def __init__(self, *, output_dir: str = "benchmarks/charts") -> None:
        self.output_dir = output_dir

    # ── Public API ──────────────────────────────────────────────────

    def generate_all_charts(
        self, results: dict[str, Any]
    ) -> list[Path]:
        """Generate all three chart types and return their paths.

        Args:
            results: Parsed benchmark results dict with "cells" key.

        Returns:
            List of Paths to generated PNG files.
        """
        paths = [
            self.bar_chart_success_rate(results),
            self.line_chart_cost_vs_quality(results),
            self.heatmap_per_instance_outcomes(results),
        ]
        return paths

    def bar_chart_success_rate(
        self, results: dict[str, Any]
    ) -> Path:
        """Generate a bar chart showing success rate by topology.

        Each bar represents a topology, with height = mean success rate.
        Error bars show the 95% CI.

        Args:
            results: Parsed benchmark results dict.

        Returns:
            Path to the generated PNG file.
        """
        cells = results.get("cells", [])
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not cells:
            return self._generate_empty_chart(
                output_path / "success_rate_by_topology.png",
                "No data available",
            )

        topologies = [c.get("topology", "unknown") for c in cells]
        means = [c.get("mean", 0.0) for c in cells]
        ci_lows = [c.get("ci_low", 0.0) for c in cells]
        ci_highs = [c.get("ci_high", 0.0) for c in cells]
        errors_low = [m - cl for m, cl in zip(means, ci_lows, strict=True)]
        errors_high = [ch - m for m, ch in zip(means, ci_highs, strict=True)]

        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=_DPI)

        x_pos = np.arange(len(topologies))
        bars = ax.bar(
            x_pos,
            means,
            width=0.6,
            color=_COLOR_PALETTE[: len(topologies)],
            edgecolor="black",
            linewidth=0.5,
        )

        # Error bars for 95% CI
        ax.errorbar(
            x_pos,
            means,
            yerr=[errors_low, errors_high],
            fmt="none",
            ecolor="black",
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
        )

        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means, strict=True)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + errors_high[i] + 0.02,
                f"{mean:.1%}",
                ha="center",
                va="bottom",
                fontsize=_FONT_SIZE,
                fontweight="bold",
            )

        ax.set_xlabel("Topology", fontsize=_FONT_SIZE)
        ax.set_ylabel("Success Rate", fontsize=_FONT_SIZE)
        ax.set_title("Success Rate by Topology (with 95% CI)", fontsize=_TITLE_SIZE)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(topologies, fontsize=_TICK_SIZE)
        ax.set_ylim(0, max(1.0, max(means) + max(errors_high) + 0.15))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        chart_path = output_path / "success_rate_by_topology.png"
        fig.savefig(chart_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)

        logger.info("Bar chart saved to %s", chart_path)
        return chart_path

    def line_chart_cost_vs_quality(
        self, results: dict[str, Any]
    ) -> Path:
        """Generate a line chart showing cost vs quality (success rate).

        X-axis: cost (USD, caching ON), Y-axis: success rate.
        Each point is a topology, with error bars for both axes.

        Args:
            results: Parsed benchmark results dict.

        Returns:
            Path to the generated PNG file.
        """
        cells = results.get("cells", [])
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not cells:
            return self._generate_empty_chart(
                output_path / "cost_vs_quality.png",
                "No data available",
            )

        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=_DPI)

        for i, cell in enumerate(cells):
            topology = cell.get("topology", "unknown")
            mean = cell.get("mean", 0.0)
            ci_low = cell.get("ci_low", mean)
            ci_high = cell.get("ci_high", mean)
            cost_on = cell.get("cost_caching_on_usd", 0.0)
            cost_off = cell.get("cost_caching_off_usd", 0.0)

            color = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]

            # Plot with caching ON (solid marker)
            ax.errorbar(
                cost_on,
                mean,
                yerr=[[mean - ci_low], [ci_high - mean]],
                xerr=[[cost_off - cost_on]],  # Show cost range as x-error
                fmt="o",
                color=color,
                markersize=10,
                capsize=4,
                label=f"{topology} (with caching)",
            )

            # Plot without caching (open marker)
            ax.scatter(
                [cost_off],
                [mean],
                s=80,
                facecolors="none",
                edgecolors=color,
                linewidths=2,
                zorder=5,
            )

        ax.set_xlabel("Cost per Instance (USD)", fontsize=_FONT_SIZE)
        ax.set_ylabel("Success Rate", fontsize=_FONT_SIZE)
        ax.set_title("Cost vs. Quality (Success Rate)", fontsize=_TITLE_SIZE)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))
        ax.legend(fontsize=_TICK_SIZE, loc="best")
        ax.grid(alpha=0.3)

        # Annotate cost savings direction
        ax.annotate(
            "← More cost-effective",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=9,
            style="italic",
            color="gray",
        )

        plt.tight_layout()
        chart_path = output_path / "cost_vs_quality.png"
        fig.savefig(chart_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)

        logger.info("Line chart saved to %s", chart_path)
        return chart_path

    def heatmap_per_instance_outcomes(
        self, results: dict[str, Any]
    ) -> Path:
        """Generate a heatmap showing per-instance outcomes across topologies.

        Rows = instance IDs, Columns = topologies.
        Cell color = success rate (0.0=red, 1.0=green).

        Args:
            results: Parsed benchmark results dict.

        Returns:
            Path to the generated PNG file.
        """
        cells = results.get("cells", [])
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not cells:
            return self._generate_empty_chart(
                output_path / "heatmap_per_instance.png",
                "No data available",
            )

        # Gather all unique instance IDs across cells
        instance_ids: list[str] = []
        instance_set: set[str] = set()
        for cell in cells:
            for inst in cell.get("instances", []):
                iid = inst.get("instance_id", "")
                if iid and iid not in instance_set:
                    instance_ids.append(iid)
                    instance_set.add(iid)

        if not instance_ids:
            return self._generate_empty_chart(
                output_path / "heatmap_per_instance.png",
                "No instance data",
            )

        topologies = [c.get("topology", "unknown") for c in cells]

        # Build the heatmap data matrix
        data = np.full((len(instance_ids), len(topologies)), np.nan)
        for j, cell in enumerate(cells):
            for inst in cell.get("instances", []):
                iid = inst.get("instance_id", "")
                if iid in instance_set:
                    row_idx = instance_ids.index(iid)
                    data[row_idx, j] = inst.get("success_rate", 0.0)

        # Dynamic figure size based on data dimensions
        fig_height = max(4, len(instance_ids) * 0.4 + 2)
        fig, ax = plt.subplots(figsize=(8, fig_height), dpi=_DPI)

        im = ax.imshow(
            data,
            cmap=_HEATMAP_CMAP,
            vmin=0,
            vmax=1,
            aspect="auto",
        )

        # Add text annotations
        for i in range(len(instance_ids)):
            for j in range(len(topologies)):
                val = data[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 0.4 or val > 0.8 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.0%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                    )

        # Shorten instance IDs for display
        short_ids: list[str] = []
        for iid in instance_ids:
            # Take last part after __ for readability
            parts = iid.split("__")
            short = parts[-1] if len(parts) > 1 else iid
            # Further truncate if too long
            if len(short) > 20:
                short = short[:17] + "..."
            short_ids.append(short)

        ax.set_xticks(np.arange(len(topologies)))
        ax.set_yticks(np.arange(len(instance_ids)))
        ax.set_xticklabels(topologies, fontsize=_TICK_SIZE, rotation=30, ha="right")
        ax.set_yticklabels(short_ids, fontsize=max(7, 10 - len(instance_ids) // 10))
        ax.set_title("Per-Instance Outcomes by Topology", fontsize=_TITLE_SIZE)

        # Color bar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Success Rate", fontsize=_FONT_SIZE)

        plt.tight_layout()
        chart_path = output_path / "heatmap_per_instance.png"
        fig.savefig(chart_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)

        logger.info("Heatmap saved to %s", chart_path)
        return chart_path

    # ── Private helpers ──────────────────────────────────────────────

    def _generate_empty_chart(self, path: Path, message: str) -> Path:
        """Generate a placeholder chart with a message.

        Used when no data is available.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, dpi=_DPI)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=16,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_axis_off()

        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)

        return path
