"""SWE-bench harness — drives topologies against SWE-bench-Lite.

Architecture reference: §2.11 SWE-bench Harness.

Components:
- Instance loader: pulls dataset rows from HuggingFace, filters to configurable slice
- Per-instance runner: spins up SWE-bench Docker image, invokes orchestrator, captures patch
- Evaluator wrapper: runs official swebench evaluator on captured patches
- Aggregator: computes mean/variance/95% CI across runs, persists JSON + Markdown

CLI entry point: ``python -m src.benchmarks.swebench``

Matrix mode (M6): ``python -m src.benchmarks.swebench --matrix --slice 30 --runs 3``
Runs all 3 topologies × instances × N=3 and captures comprehensive metrics.
"""

from src.benchmarks.swebench.aggregator import Aggregator
from src.benchmarks.swebench.loader import InstanceLoader
from src.benchmarks.swebench.models import RunConfig, SweBenchInstance, SweBenchResult

__all__ = [
    "Aggregator",
    "InstanceLoader",
    "RunConfig",
    "SweBenchInstance",
    "SweBenchResult",
]
