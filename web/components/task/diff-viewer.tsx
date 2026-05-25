"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface DiffViewerProps {
  diff?: string | null;
}

/** Represents a single file's diff within a multi-file diff */
interface FileDiff {
  /** File path (from --- a/ or +++ b/ lines) */
  filePath: string;
  /** All lines for this file including headers, hunk markers, and content */
  lines: string[];
  /** Whether this appears to be a binary file */
  isBinary: boolean;
}

/** Parse a unified diff string into per-file sections */
function parseDiff(diff: string): FileDiff[] {
  const lines = diff.split("\n");
  const files: FileDiff[] = [];
  let currentFile: FileDiff | null = null;

  for (const line of lines) {
    // New file starts with "diff --git"
    if (line.startsWith("diff --git ")) {
      // Extract file path from "diff --git a/path b/path"
      const match = line.match(/^diff --git a\/(.+?) b\//);
      if (match) {
        if (currentFile) {
          files.push(currentFile);
        }
        currentFile = {
          filePath: match[1],
          lines: [line],
          isBinary: false,
        };
      }
      continue;
    }

    // Detect binary files
    if (line.startsWith("Binary files") || line === "Binary files differ") {
      if (currentFile) {
        currentFile.isBinary = true;
        currentFile.lines.push(line);
      }
      continue;
    }

    // If we haven't seen a "diff --git" yet, create a synthetic file
    if (!currentFile) {
      // For diffs without "diff --git" headers, create a synthetic single-file diff
      currentFile = {
        filePath: "changes",
        lines: [],
        isBinary: false,
      };
    }

    // Extract file path from --- / +++ lines if we haven't found it yet
    if (currentFile.filePath === "changes") {
      const pathMatch = line.match(/^--- a\/(.+)$/);
      if (pathMatch) {
        currentFile.filePath = pathMatch[1];
      }
    }

    currentFile.lines.push(line);
  }

  if (currentFile) {
    files.push(currentFile);
  }

  return files;
}

/** Get CSS class for a diff line based on its prefix */
function getLineClass(line: string): string {
  if (line.startsWith("+++") || line.startsWith("---")) {
    return "font-bold text-foreground";
  }
  if (line.startsWith("+")) {
    return "text-green-400";
  }
  if (line.startsWith("-")) {
    return "text-red-400";
  }
  if (line.startsWith("@@")) {
    return "text-cyan-400";
  }
  return "text-muted-foreground";
}

/** Single file diff section with collapse/expand */
function FileDiffSection({ file }: { file: FileDiff }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="border border-border rounded-md overflow-hidden">
      {/* File header — clickable to toggle collapse */}
      <button
        type="button"
        className="w-full flex items-center gap-2 px-3 py-2 bg-muted/50 hover:bg-muted/80 text-left text-sm font-mono"
        onClick={() => setCollapsed(!collapsed)}
        aria-expanded={!collapsed}
        aria-label={`Toggle diff for ${file.filePath}`}
      >
        <span className="text-muted-foreground select-none">
          {collapsed ? "▶" : "▼"}
        </span>
        <span className="font-semibold text-foreground">{file.filePath}</span>
        {file.isBinary && (
          <Badge variant="outline" className="text-xs ml-2">
            Binary
          </Badge>
        )}
      </button>

      {/* Diff content */}
      {!collapsed && (
        <div className="overflow-x-auto bg-muted/20 p-2">
          {file.isBinary ? (
            <div className="py-4 text-center text-sm text-muted-foreground">
              Binary file — cannot display content
            </div>
          ) : (
            <pre className="text-xs leading-relaxed">
              {file.lines.map((line, idx) => (
                <div
                  key={idx}
                  className={`${getLineClass(line)} whitespace-pre hover:bg-muted/30`}
                >
                  {line || " "}
                </div>
              ))}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export function DiffViewer({ diff }: DiffViewerProps) {
  const files = useMemo(() => (diff ? parseDiff(diff) : []), [diff]);

  if (!diff) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Diff</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No diff available yet. The diff will appear after the Coder agent
            produces changes.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Diff</CardTitle>
          <Badge variant="outline" className="text-xs">
            {files.length} {files.length === 1 ? "file" : "files"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {files.map((file, idx) => (
          <FileDiffSection key={`${file.filePath}-${idx}`} file={file} />
        ))}
      </CardContent>
    </Card>
  );
}
