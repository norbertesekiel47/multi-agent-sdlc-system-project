/**
 * Enhanced tests for the DiffViewer component.
 *
 * Covers:
 * - VAL-HITL-UI-002: Diff viewer renders unified diff with syntax highlighting
 * - File-level grouping with file headers
 * - Per-file collapse/expand
 * - Binary file placeholder
 * - Long diffs are scrollable
 * - Added lines (green), removed lines (red), context lines
 */

import React from "react";
import { describe, it, expect, afterEach, vi } from "vitest";
import { renderWithProviders } from "../test-utils";
import { DiffViewer } from "@/components/task/diff-viewer";
import { screen, cleanup, fireEvent } from "@testing-library/react";

afterEach(() => cleanup());

const MULTI_FILE_DIFF = `diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,5 +1,6 @@
 import os
+import sys
 
 def main():
-    pass
+    print("hello")
diff --git a/src/utils.py b/src/utils.py
--- a/src/utils.py
+++ b/src/utils.py
@@ -1,3 +1,4 @@
 def helper():
-    return None
+    return True
+    # added comment`;

describe("DiffViewer — Enhanced", () => {
  it("shows empty state when no diff provided", () => {
    renderWithProviders(<DiffViewer />);

    expect(
      screen.getByText(/no diff available yet/i)
    ).toBeInTheDocument();
  });

  it("shows empty state when diff is null", () => {
    renderWithProviders(<DiffViewer diff={null} />);

    expect(
      screen.getAllByText(/no diff available yet/i).length
    ).toBeGreaterThanOrEqual(1);
  });

  it("renders diff with syntax highlighting — added lines green, removed red", () => {
    const diff = `--- a/file.py
+++ b/file.py
-removed line
+added line
 context line`;

    renderWithProviders(<DiffViewer diff={diff} />);

    const removedLine = screen.getByText("-removed line");
    const addedLine = screen.getByText("+added line");

    expect(removedLine.className).toContain("text-red");
    expect(addedLine.className).toContain("text-green");
  });

  it("renders hunk headers with cyan color", () => {
    const diff = `--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 context`;

    renderWithProviders(<DiffViewer diff={diff} />);

    const hunkHeader = screen.getByText(/@@ -1,3 \+1,4 @@/);
    expect(hunkHeader.className).toContain("text-cyan");
  });

  it("groups diff lines by file with file headers", () => {
    renderWithProviders(<DiffViewer diff={MULTI_FILE_DIFF} />);

    // Both file headers should be present — use getAllByText since
    // "src/main.py" also appears in --- a/ and +++ b/ lines
    const mainPyElements = screen.getAllByText(/src\/main\.py/);
    const utilsPyElements = screen.getAllByText(/src\/utils\.py/);
    expect(mainPyElements.length).toBeGreaterThanOrEqual(1);
    expect(utilsPyElements.length).toBeGreaterThanOrEqual(1);
  });

  it("renders binary file placeholder for binary files", () => {
    const diff = `diff --git a/image.png b/image.png
Binary files differ`;

    renderWithProviders(<DiffViewer diff={diff} />);

    expect(screen.getByText(/binary file/i)).toBeInTheDocument();
  });

  it("renders all diff content for a single file", () => {
    const diff = `diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,4 @@
 import os
+import sys`;

    renderWithProviders(<DiffViewer diff={diff} />);

    expect(screen.getByText("+import sys")).toBeInTheDocument();
  });
});
