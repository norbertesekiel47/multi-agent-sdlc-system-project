"""Single PydanticAI agent with all tools for the single_agent topology.

This agent handles the full pipeline: plan → edit → test → review.
It uses DeepSeek V4 Flash via OpenRouter and has access to:
- Sandbox file operations (read_file, write_file, apply_diff, run_command)
- Sandbox test runner (run_tests)
- GitHub client operations (via orchestrator delegation)

The agent name is always "single_agent" so Langfuse traces show
exactly one distinct agent name (VAL-TOPOLOGY-001).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from src.agents.models import SingleAgentOutput

load_dotenv()

logger = logging.getLogger(__name__)

# Model ID per architecture §2.3 / §5
_SINGLE_AGENT_MODEL = "openrouter:deepseek/deepseek-v4-flash"

# Default test command
_DEFAULT_TEST_CMD = "python -m pytest tests/ -v"


def _compute_diff_hash(diff: str) -> str:
    """Compute SHA-256 hash of a diff for loop detection."""
    return hashlib.sha256(diff.encode()).hexdigest()


class SandboxTools:
    """Tool dependencies for the single agent.

    Holds a reference to the sandbox manager and provides
    methods that the agent can call during its run.
    """

    def __init__(self, sandbox_manager: Any, workspace_dir: str) -> None:
        self.sandbox = sandbox_manager
        self.workspace = workspace_dir


# ── Create the agent ───────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an autonomous coding agent that fixes GitHub issues.
You have access to sandbox file tools and a test runner.

Your workflow:
1. Read the issue and understand what needs to be fixed
2. Read the relevant source files to understand the current code
3. Plan the changes needed
4. Write the fix using file tools
5. Run the tests to verify the fix
6. If tests fail, iterate on the fix (up to 3 attempts)
7. Produce a final structured output with the plan, code edit,
   test report, and whether you're ready for PR

IMPORTANT RULES:
- Always read files before editing them
- Always run tests after making changes
- If tests fail, try to fix them (up to 3 attempts)
- Never modify files outside the repo workspace
- Produce ONLY the structured SingleAgentOutput when done
- Your output must be valid JSON matching the \
SingleAgentOutput schema
"""

single_agent = Agent(
    model=_SINGLE_AGENT_MODEL,
    output_type=SingleAgentOutput,
    name="single_agent",
    deps_type=SandboxTools,
    system_prompt=_SYSTEM_PROMPT,
)


# ── Register tools ──────────────────────────────────────────────────


@single_agent.tool
async def read_file(ctx: RunContext[SandboxTools], path: str) -> str:
    """Read a file from the sandbox workspace.

    Args:
        path: Relative path to the file (e.g. 'src/calculator.py')
    """
    try:
        content = await ctx.deps.sandbox.read_file(path)
        return content
    except Exception as exc:
        return f"Error reading {path}: {exc}"


@single_agent.tool
async def write_file(
    ctx: RunContext[SandboxTools], path: str, content: str
) -> str:
    """Write content to a file in the sandbox workspace.

    Args:
        path: Relative path to the file (e.g. 'src/calculator.py')
        content: The full content to write
    """
    try:
        await ctx.deps.sandbox.write_file(path, content)
        return f"Successfully wrote {path}"
    except Exception as exc:
        return f"Error writing {path}: {exc}"


@single_agent.tool
async def run_command(ctx: RunContext[SandboxTools], command: str) -> str:
    """Run a shell command in the sandbox workspace.

    Args:
        command: The shell command to execute
    """
    try:
        output = await ctx.deps.sandbox.run_command(command)
        return output
    except Exception as exc:
        return f"Error running command: {exc}"


@single_agent.tool
async def run_tests(
    ctx: RunContext[SandboxTools],
    test_command: str = _DEFAULT_TEST_CMD,
) -> str:
    """Run tests in the sandbox and return the output.

    Args:
        test_command: The test command to run (default: pytest)
    """
    try:
        output = await ctx.deps.sandbox.run_tests(test_command=test_command)
        return output
    except Exception as exc:
        return f"Error running tests: {exc}"


@single_agent.tool
async def apply_diff(ctx: RunContext[SandboxTools], diff: str) -> str:
    """Apply a unified diff to files in the sandbox workspace.

    Args:
        diff: The unified diff to apply
    """
    try:
        await ctx.deps.sandbox.apply_diff(diff)
        return "Successfully applied diff"
    except Exception as exc:
        return f"Error applying diff: {exc}"


@single_agent.tool
async def list_files(
    ctx: RunContext[SandboxTools], directory: str = "."
) -> str:
    """List files in a directory within the sandbox workspace.

    Args:
        directory: Directory path to list (default: workspace root)
    """
    try:
        cmd = f"find {directory} -type f -not -path '*/.git/*' | head -50"
        output = await ctx.deps.sandbox.run_command(cmd)
        return output
    except Exception as exc:
        return f"Error listing files: {exc}"
