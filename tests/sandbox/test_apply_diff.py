"""Tests for apply_diff functionality using the patch binary.

VAL-SANDBOX-ISO-017 (feature scope): The sandbox base image must include
the `patch` binary so that apply_diff works correctly. The default
python:3.12-slim image lacks `patch`, causing exit code 127.

These tests verify:
1. The `patch` binary is available inside the sandbox container
2. apply_diff successfully applies a unified diff
3. The file is correctly modified after applying the diff
4. apply_diff works with multi-file diffs
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from src.sandbox.config import SANDBOX_CONTAINER_PREFIX
from src.sandbox.errors import SandboxToolError
from src.sandbox.manager import SandboxManager

# ──────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────

TEST_TIMEOUT = 30
TEST_MEMORY = "4g"
TEST_CPU = 2.0
TEST_UID = 1000


def _unique_task_id() -> str:
    """Generate a unique task ID for test isolation."""
    return f"test-patch-{uuid.uuid4().hex[:12]}"


def _cleanup_task(task_id: str) -> None:
    """Force cleanup any leftover Docker resources for a task."""
    for name in [
        f"{SANDBOX_CONTAINER_PREFIX}{task_id}",
        f"sdlc-swarm-proxy-{task_id}",
    ]:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    net_name = f"sdlc-swarm-net-{task_id}"
    subprocess.run(["docker", "network", "rm", net_name], capture_output=True)
    ws_dir = Path(f"/var/sdlc-swarm/work/{task_id}")
    if ws_dir.exists():
        shutil.rmtree(ws_dir, ignore_errors=True)


@pytest.fixture
async def sandbox():
    """Provide a fresh sandbox that is cleaned up after the test."""
    task_id = _unique_task_id()
    sb = SandboxManager(
        task_id=task_id,
        memory_limit=TEST_MEMORY,
        cpu_limit=TEST_CPU,
        timeout_seconds=TEST_TIMEOUT,
        user_uid=TEST_UID,
    )
    try:
        await sb.setup()
        await asyncio.sleep(2)
        yield sb
    finally:
        await sb.teardown()
        _cleanup_task(task_id)


# ──────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────

@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_patch_binary_available_in_sandbox(sandbox: SandboxManager) -> None:
    """The `patch` binary must be available inside the sandbox container.

    The default python:3.12-slim image does not include `patch`,
    causing apply_diff to fail with exit code 127. The custom
    sandbox-base image must install `patch` via apt-get.
    """
    output = await sandbox.run_command("which patch", timeout=10)
    assert output.strip(), "patch binary not found in sandbox container"

    # Also verify patch --version works
    version_output = await sandbox.run_command("patch --version", timeout=10)
    assert version_output.strip(), "patch --version returned empty output"


@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_apply_diff_simple_modification(sandbox: SandboxManager) -> None:
    """apply_diff must successfully apply a unified diff to modify a file.

    End-to-end test:
    1. Create a file in the sandbox
    2. Apply a unified diff via apply_diff
    3. Confirm the file is modified correctly
    """
    # Create a simple Python file
    original_content = """def hello():
    return "world"

def goodbye():
    return "farewell"
"""
    await sandbox.write_file("src/greetings.py", original_content)

    # Verify the file was written correctly
    content_before = await sandbox.read_file("src/greetings.py")
    assert "world" in content_before

    # Apply a unified diff that changes "world" to "universe"
    diff = """--- a/src/greetings.py
+++ b/src/greetings.py
@@ -1,4 +1,4 @@
 def hello():
-    return "world"
+    return "universe"

 def goodbye():
"""
    await sandbox.apply_diff(diff)

    # Verify the file was modified correctly
    content_after = await sandbox.read_file("src/greetings.py")
    assert "universe" in content_after, (
        f"apply_diff did not modify the file. Content: {content_after}"
    )
    assert '"world"' not in content_after, (
        f"Original text still present after apply_diff. Content: {content_after}"
    )
    assert "goodbye" in content_after, "Unchanged lines were modified"


@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_apply_diff_adds_new_file(sandbox: SandboxManager) -> None:
    """apply_diff must handle diffs that create new files."""
    # Create a subdirectory
    await sandbox.run_command("mkdir -p src/utils", timeout=5)

    # Apply a diff that adds a new file
    diff = (
        "--- /dev/null\n"
        "+++ b/src/utils/helpers.py\n"
        "@@ -0,0 +1,5 @@\n"
        "+def helper():\n"
        '+    """A helper function."""\n'
        "+    return True\n"
        "+\n"
        "+EXPORT = helper\n"
    )
    await sandbox.apply_diff(diff)

    # Verify the new file exists and has correct content
    content = await sandbox.read_file("src/utils/helpers.py")
    assert "def helper" in content, f"New file not created correctly: {content}"
    assert "EXPORT" in content, f"New file content incomplete: {content}"


@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_apply_diff_with_multiple_hunks(sandbox: SandboxManager) -> None:
    """apply_diff must handle diffs with multiple hunks in the same file."""
    # Create a file with multiple sections
    original_content = """# Header
VALUE_A = 1
VALUE_B = 2
VALUE_C = 3
# Middle
VALUE_D = 4
VALUE_E = 5
# Footer
VALUE_F = 6
"""
    await sandbox.write_file("config.py", original_content)

    # Apply a diff with two separate hunks
    diff = """--- a/config.py
+++ b/config.py
@@ -1,4 +1,4 @@
 # Header
-VALUE_A = 1
+VALUE_A = 10
 VALUE_B = 2
 VALUE_C = 3
@@ -6,4 +6,4 @@
 VALUE_E = 5
 # Footer
-VALUE_F = 6
+VALUE_F = 60
"""
    await sandbox.apply_diff(diff)

    # Verify both hunks were applied
    content = await sandbox.read_file("config.py")
    assert "VALUE_A = 10" in content, f"First hunk not applied: {content}"
    assert "VALUE_F = 60" in content, f"Second hunk not applied: {content}"
    # Verify unchanged lines remain intact
    assert "VALUE_B = 2" in content, f"Unchanged lines modified: {content}"


@pytest.mark.sandbox_isolation
@pytest.mark.asyncio
async def test_apply_diff_exit_code_127_without_patch_binary(sandbox: SandboxManager) -> None:
    """Verify that the sandbox image includes patch — the original python:3.12-slim
    would return exit code 127 for the patch command.

    This test confirms that our custom sandbox-base image fixes the issue.
    If this test fails with exit code 127, the custom image was not built
    or not used as the default sandbox image.
    """
    # Simply verify `patch` is on PATH — if it returns 127, patch is missing
    try:
        output = await sandbox.run_command("patch --version", timeout=10)
        # patch --version succeeded — good, the binary is present
        assert output.strip(), "patch --version returned empty output"
    except SandboxToolError as e:
        # Check if the error indicates exit code 127 (command not found)
        error_msg = str(e)
        assert "127" not in error_msg, (
            f"patch command returned exit code 127 (not found). "
            f"Is the custom sandbox-base image being used? Error: {error_msg}"
        )
        raise
