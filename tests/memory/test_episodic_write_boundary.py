"""VAL-EPISODIC-002: Only orchestrator writes to episodic tables.

No agent module imports the episodic-store writer; only ``src/orchestrator/``
writes to tasks, decisions, outcomes, repo_facts.
"""

from __future__ import annotations

import pkgutil
from pathlib import Path

import pytest

# Root of the source tree
_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src"


def _iter_modules(package_path: Path, prefix: str = "") -> list[str]:
    """Recursively find all module names under a package path."""
    modules: list[str] = []
    if not package_path.exists():
        return modules
    for _finder, name, is_pkg in pkgutil.walk_packages([str(package_path)], prefix):
        modules.append(name)
        if is_pkg:
            sub = package_path / name.split(".")[-1]
            modules.extend(_iter_modules(sub, name + "."))
    return modules


class TestEpisodicWriteBoundary:
    """Agents must not import episodic write operations."""

    def test_no_agent_imports_episodic_writer(self) -> None:
        """No module under src/agents/ imports EpisodicStore write methods."""
        agents_path = _SRC_ROOT / "agents"
        if not agents_path.exists():
            # Agents don't exist yet in M1 — skip gracefully
            pytest.skip("src/agents/ does not exist yet (expected in M1)")

        # Check for imports of episodic store write operations
        forbidden_patterns = [
            "from src.memory.episodic.store import EpisodicStore",
            "from src.memory.episodic import EpisodicStore",
            "from src.memory.episodic.store import",
        ]

        for py_file in agents_path.rglob("*.py"):
            content = py_file.read_text()
            for pattern in forbidden_patterns:
                assert pattern not in content, (
                    f"{py_file.relative_to(_SRC_ROOT)} imports EpisodicStore — "
                    f"agents must not write to episodic tables"
                )

    def test_no_agent_sql_writes(self) -> None:
        """No agent module contains raw INSERT/UPDATE/DELETE against episodic tables."""
        agents_path = _SRC_ROOT / "agents"
        if not agents_path.exists():
            pytest.skip("src/agents/ does not exist yet (expected in M1)")

        table_names = ["tasks", "decisions", "outcomes", "repo_facts"]
        for py_file in agents_path.rglob("*.py"):
            content = py_file.read_text()
            for table in table_names:
                for op in ["INSERT INTO", "UPDATE", "DELETE FROM"]:
                    # Simple substring check (not regex — avoids false positives)
                    assert f"{op} {table}" not in content, (
                        f"{py_file.relative_to(_SRC_ROOT)} contains "
                        f"'{op} {table}' — agents must not write to episodic tables"
                    )

    def test_orchestrator_can_write(self) -> None:
        """At least one module under src/orchestrator/ writes to episodic tables.

        This test is skipped if src/orchestrator/ doesn't exist yet (M1).
        """
        orchestrator_path = _SRC_ROOT / "orchestrator"
        if not orchestrator_path.exists():
            pytest.skip("src/orchestrator/ does not exist yet (expected in M2+)")

        # Check that at least one module imports EpisodicStore or uses episodic writes
        found_writer = False
        for py_file in orchestrator_path.rglob("*.py"):
            content = py_file.read_text()
            if "EpisodicStore" in content or "episodic" in content.lower():
                found_writer = True
                break

        assert found_writer, (
            "No module under src/orchestrator/ imports or uses EpisodicStore — "
            "the orchestrator must be able to write to episodic tables"
        )

    def test_episodic_module_provides_read_and_write(self) -> None:
        """The episodic store module provides both read and write methods."""
        from src.memory.episodic.store import EpisodicStore

        # Write methods
        write_methods = [
            "create_task",
            "update_task_status",
            "update_task_totals",
            "finish_task",
            "create_decision",
            "create_outcome",
            "upsert_repo_fact",
        ]
        # Read methods
        read_methods = [
            "get_task",
            "query_recent_decisions",
            "query_recent_outcomes",
            "get_outcomes_for_task",
            "query_repo_facts",
            "get_planner_context",
            "list_tasks",
        ]

        for method in write_methods + read_methods:
            assert hasattr(EpisodicStore, method), (
                f"EpisodicStore missing method: {method}"
            )
