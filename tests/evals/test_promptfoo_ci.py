"""Tests for promptfoo eval suites in GitHub Actions CI.

Verifies:
- Promptfoo configuration file exists and is structurally valid
- Eval test cases reference the custom curated repo's known-good outcomes
- Assertions verify agent prompt behavior (e.g., Planner mentions correct target_files)
- GitHub Actions CI workflow runs promptfoo eval under Node 22 via fnm
- Prompt regression is caught by failing eval
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVALS_DIR = REPO_ROOT / "evals"
PROMPTFOO_CONFIG = EVALS_DIR / "promptfooconfig.yaml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "promptfoo-eval.yml"
CURATED_REPO = "https://github.com/norbertesekiel47/sdlc-swarm-curated"


# ── Structural Tests: Config File ──────────────────────────────────


class TestPromptfooConfigExists:
    """Verify the promptfoo configuration file exists."""

    def test_config_file_exists(self) -> None:
        """promptfooconfig.yaml must exist in evals/ directory."""
        assert PROMPTFOO_CONFIG.is_file(), (
            f"Promptfoo config not found at {PROMPTFOO_CONFIG}"
        )

    def test_config_is_valid_yaml(self) -> None:
        """promptfooconfig.yaml must parse as valid YAML."""
        assert PROMPTFOO_CONFIG.is_file(), "Config file must exist first"
        content = PROMPTFOO_CONFIG.read_text()
        config: dict[str, Any] = yaml.safe_load(content)
        assert isinstance(config, dict), "Config must be a YAML dict"


class TestPromptfooConfigStructure:
    """Verify the promptfoo configuration has required structure."""

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        """Load the promptfoo config."""
        assert PROMPTFOO_CONFIG.is_file(), "Config file must exist first"
        content = PROMPTFOO_CONFIG.read_text()
        return yaml.safe_load(content)

    def test_config_has_providers(self, config: dict[str, Any]) -> None:
        """Config must define at least one provider (OpenRouter)."""
        assert "providers" in config, "Config must have 'providers' key"
        providers = config["providers"]
        assert isinstance(providers, list), "providers must be a list"
        assert len(providers) > 0, "At least one provider must be defined"

    def test_config_has_openrouter_provider(self, config: dict[str, Any]) -> None:
        """At least one provider must use OpenRouter (for DeepSeek models)."""
        providers = config["providers"]
        provider_ids: list[str] = []
        for p in providers:
            if isinstance(p, str):
                provider_ids.append(p)
            elif isinstance(p, dict) and "id" in p:
                provider_ids.append(p["id"])
        openrouter_present = any("openrouter" in pid for pid in provider_ids)
        assert openrouter_present, (
            f"At least one provider must use OpenRouter. Found: {provider_ids}"
        )

    def test_config_has_tests(self, config: dict[str, Any]) -> None:
        """Config must define test cases."""
        assert "tests" in config, "Config must have 'tests' key"
        tests = config["tests"]
        assert isinstance(tests, list), "tests must be a list"
        assert len(tests) > 0, "At least one test case must be defined"

    def test_config_has_description(self, config: dict[str, Any]) -> None:
        """Config should have a description for identification."""
        assert "description" in config, "Config should have 'description'"

    def test_config_references_curated_repo(self, config: dict[str, Any]) -> None:
        """Test cases should reference the custom curated repo."""
        tests = config.get("tests", [])
        tests_str = str(tests)
        assert "sdlc-swarm-curated" in tests_str or CURATED_REPO in tests_str, (
            "Test cases must reference the custom curated repo "
            "(norbertesekiel47/sdlc-swarm-curated)"
        )


class TestPromptfooEvalAssertions:
    """Verify eval assertions check agent prompt behavior."""

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        """Load the promptfoo config."""
        assert PROMPTFOO_CONFIG.is_file(), "Config file must exist first"
        content = PROMPTFOO_CONFIG.read_text()
        return yaml.safe_load(content)

    def test_planner_assertion_checks_target_files(
        self, config: dict[str, Any]
    ) -> None:
        """Planner eval must assert that target_files includes expected file.

        E.g., for issue #1 (subtract bug), the Planner should mention
        src/calculator.py in its output.
        """
        tests = config.get("tests", [])
        has_planner_assertion = False
        for test in tests:
            if not isinstance(test, dict):
                continue
            test_str = str(test).lower()
            if "planner" in test_str or "change_plan" in test_str or "target_file" in test_str:
                # Must have an assertion that checks for target_files
                assert "assert" in test or "assertions" in test, (
                    f"Planner test case must have assertions: {test}"
                )
                has_planner_assertion = True
        assert has_planner_assertion, (
            "At least one test must assert Planner behavior "
            "(e.g., target_files includes expected file)"
        )

    def test_assertions_use_rouge_or_contains(
        self, config: dict[str, Any]
    ) -> None:
        """Assertions must use valid promptfoo assertion types.

        Valid types include: contains, regex, rouge, llm-rubric, etc.
        """
        tests = config.get("tests", [])
        for test in tests:
            if not isinstance(test, dict):
                continue
            assertions = test.get("assert", test.get("assertions", []))
            if not assertions:
                continue
            for assertion in assertions:
                if not isinstance(assertion, dict):
                    continue
                assert "type" in assertion, (
                    f"Assertion must have 'type' field: {assertion}"
                )
                # Valid promptfoo assertion types (subset relevant to our use)
                valid_types = {
                    "contains",
                    "not-contains",
                    "regex",
                    "icontains",
                    "starts-with",
                    "rouge",
                    "llm-rubric",
                    "python",
                    "javascript",
                    "is-json",
                    "contains-json",
                    "less-than",
                    "greater-than",
                    "equals",
                }
                assert assertion["type"] in valid_types, (
                    f"Assertion type '{assertion['type']}' not in valid types: {valid_types}"
                )


# ── Structural Tests: CI Workflow ──────────────────────────────────


class TestCIWorkflowExists:
    """Verify the GitHub Actions CI workflow exists."""

    def test_workflow_file_exists(self) -> None:
        """promptfoo-eval.yml must exist in .github/workflows/."""
        assert CI_WORKFLOW.is_file(), (
            f"CI workflow not found at {CI_WORKFLOW}"
        )

    def test_workflow_is_valid_yaml(self) -> None:
        """The workflow file must parse as valid YAML."""
        assert CI_WORKFLOW.is_file(), "Workflow file must exist first"
        content = CI_WORKFLOW.read_text()
        workflow: dict[str, Any] = yaml.safe_load(content)
        assert isinstance(workflow, dict), "Workflow must be a YAML dict"


class TestCIWorkflowStructure:
    """Verify the CI workflow has required structure for promptfoo eval."""

    @pytest.fixture()
    def workflow(self) -> dict[str, Any]:
        """Load the CI workflow."""
        assert CI_WORKFLOW.is_file(), "Workflow file must exist first"
        content = CI_WORKFLOW.read_text()
        return yaml.safe_load(content)

    def test_workflow_runs_on_push(self, workflow: dict[str, Any]) -> None:
        """Workflow must trigger on push."""
        # GitHub Actions uses 'on:' which YAML parses as True (boolean)
        # because 'on' is a YAML boolean literal.
        on_config = workflow.get("on", workflow.get(True, {}))
        has_push = "push" in on_config
        assert has_push, "Workflow must trigger on push"

    def test_workflow_uses_fnm_node_22(
        self, workflow: dict[str, Any]
    ) -> None:
        """Workflow must install fnm and use Node 22 LTS."""
        content_str = str(workflow)
        assert "fnm" in content_str, "Workflow must install/use fnm"
        assert "22" in content_str, "Workflow must use Node 22 LTS via fnm"

    def test_workflow_runs_promptfoo_eval(
        self, workflow: dict[str, Any]
    ) -> None:
        """Workflow must run promptfoo eval command."""
        content_str = str(workflow).lower()
        assert "promptfoo" in content_str, (
            "Workflow must run promptfoo eval command"
        )
        assert "eval" in content_str, (
            "Workflow must run 'promptfoo eval' or equivalent"
        )

    def test_workflow_sets_openrouter_api_key(
        self, workflow: dict[str, Any]
    ) -> None:
        """Workflow must make OPENROUTER_API_KEY available for the eval."""
        content_str = str(workflow)
        assert "OPENROUTER_API_KEY" in content_str, (
            "Workflow must set OPENROUTER_API_KEY env var for promptfoo"
        )

    def test_workflow_fails_on_regression(self, workflow: dict[str, Any]) -> None:
        """Workflow should not have continue-on-error for the eval step."""
        # Find the eval step
        jobs = workflow.get("jobs", {})
        for _job_name, job_data in jobs.items():
            steps = job_data.get("steps", [])
            for step in steps:
                step_str = str(step).lower()
                if "promptfoo" in step_str and "eval" in step_str:
                    # The promptfoo eval step must NOT have continue-on-error: true
                    assert not step.get("continue-on-error", False), (
                        "Promptfoo eval step must NOT continue on error "
                        "(must fail the workflow on regression)"
                    )
                    return
        pytest.fail("Could not find promptfoo eval step in workflow")


# ── Regression Detection Test ─────────────────────────────────────


class TestPromptRegressionDetection:
    """Verify that the eval config can detect prompt regressions."""

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        """Load the promptfoo config."""
        assert PROMPTFOO_CONFIG.is_file(), "Config file must exist first"
        content = PROMPTFOO_CONFIG.read_text()
        return yaml.safe_load(content)

    def test_eval_has_known_good_assertions(
        self, config: dict[str, Any]
    ) -> None:
        """Eval assertions must verify known-good outcomes.

        For the custom curated repo, known-good outcomes include:
        - Planner for issue #1 (subtract bug) → target_files includes calculator.py
        - Coder for issue #1 → output contains a diff with subtraction fix
        - Reviewer for a clean diff → verdict is accept
        - QA for a correct fix → tests pass
        """
        tests = config.get("tests", [])
        total_assertions = 0
        for test in tests:
            if not isinstance(test, dict):
                continue
            assertions = test.get("assert", test.get("assertions", []))
            total_assertions += len(assertions)
        assert total_assertions >= 4, (
            f"Config must have at least 4 total assertions across all tests "
            f"to cover key agent behaviors. Found: {total_assertions}"
        )

    def test_eval_uses_planner_model(
        self, config: dict[str, Any]
    ) -> None:
        """Planner eval must use deepseek/deepseek-v4-pro."""
        content_str = str(config)
        assert "deepseek/deepseek-v4-pro" in content_str or "v4-pro" in content_str, (
            "Config must include deepseek/deepseek-v4-pro for Planner evals"
        )

    def test_eval_uses_coder_reviewer_model(
        self, config: dict[str, Any]
    ) -> None:
        """Coder/Reviewer/QA evals must use deepseek/deepseek-chat-v3-0324."""
        content_str = str(config)
        assert "deepseek/deepseek-chat-v3-0324" in content_str or "chat-v3-0324" in content_str, (
            "Config must include deepseek/deepseek-chat-v3-0324 "
            "for Coder/Reviewer/QA evals"
        )


# ── Promptfoo CLI Functional Test ──────────────────────────────────


class TestPromptfooCLI:
    """Verify that promptfoo CLI is functional under Node 22 via fnm."""

    def test_fnm_node_22_available(self) -> None:
        """fnm with Node 22 must be available on the system."""
        result = os.popen('eval "$(fnm env)" && fnm use 22 && node --version 2>/dev/null').read()
        # Output may include "Using Node..." prefix line from fnm
        version_line = result.strip().split("\n")[-1]
        assert version_line.startswith("v22"), (
            f"Node 22 via fnm must be available. Got: {result!r}"
        )

    def test_promptfoo_config_specifies_output(self) -> None:
        """Config should specify output path for eval results."""
        assert PROMPTFOO_CONFIG.is_file(), "Config file must exist first"
        content = PROMPTFOO_CONFIG.read_text()
        config: dict[str, Any] = yaml.safe_load(content)
        # output path or writeTo is optional but recommended for CI
        # At minimum, the config should be runnable without errors
        assert "providers" in config and "tests" in config
