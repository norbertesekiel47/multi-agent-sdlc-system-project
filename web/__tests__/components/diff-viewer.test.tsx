/**
 * Tests for the DiffViewer component.
 *
 * Verifies: empty state, diff rendering with syntax highlighting.
 */

import { describe, it, expect, afterEach } from "vitest";
import { renderWithProviders } from "../test-utils";
import { DiffViewer } from "@/components/task/diff-viewer";
import { screen, cleanup } from "@testing-library/react";

afterEach(() => cleanup());

describe("DiffViewer", () => {
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

  it("renders diff with syntax highlighting", () => {
    const diff = `diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,4 @@
 import os
+import sys
 
 def main():
     pass`;

    renderWithProviders(<DiffViewer diff={diff} />);

    // Should contain the diff text
    expect(screen.getByText(/diff --git/i)).toBeInTheDocument();
    expect(screen.getByText(/import sys/i)).toBeInTheDocument();
  });

  it("renders added lines in green and removed lines in red", () => {
    const diff = `--- a/file.py
+++ b/file.py
-removed line
+added line`;

    renderWithProviders(<DiffViewer diff={diff} />);

    const removedLine = screen.getByText("-removed line");
    const addedLine = screen.getByText("+added line");

    expect(removedLine.className).toContain("text-red");
    expect(addedLine.className).toContain("text-green");
  });
});
