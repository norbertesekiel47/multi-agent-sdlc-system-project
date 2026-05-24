"""SWE-bench harness — drives topologies against SWE-bench-Lite.

Architecture reference: §2.11 SWE-bench Harness.

Components:
- Instance loader: pulls dataset rows from HuggingFace, filters to configurable slice
- Per-instance runner: spins up SWE-bench Docker image, invokes orchestrator, captures patch
- Evaluator wrapper: runs official swebench evaluator on captured patches
- Aggregator: computes mean/variance/95% CI across runs, persists JSON + Markdown

CLI entry point: ``python -m sdlc_swarm.benchmarks.swebench``
"""

from src.benchmarks.swebench.aggregator import Aggregator
from src.benchmarks.swebench.loader import InstanceLoader
from src.benchmarks.swebench.models import SweBenchInstance, SweBenchResult

__all__ = ["Aggregator", "InstanceLoader", "SweBenchInstance", "SweBenchResult"]
