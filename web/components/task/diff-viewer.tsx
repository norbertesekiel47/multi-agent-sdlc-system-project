"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface DiffViewerProps {
  diff?: string | null;
}

export function DiffViewer({ diff }: DiffViewerProps) {
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

  const lines = diff.split("\n");

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Diff</CardTitle>
      </CardHeader>
      <CardContent>
        <pre className="overflow-x-auto rounded-md bg-muted p-4 text-xs leading-relaxed">
          {lines.map((line, idx) => {
            let lineClass = "text-foreground";
            if (line.startsWith("+++") || line.startsWith("---")) {
              lineClass = "font-bold text-foreground";
            } else if (line.startsWith("+")) {
              lineClass = "text-green-400";
            } else if (line.startsWith("-")) {
              lineClass = "text-red-400";
            } else if (line.startsWith("@@")) {
              lineClass = "text-cyan-400";
            }

            return (
              <div key={idx} className={`${lineClass} whitespace-pre`}>
                {line || " "}
              </div>
            );
          })}
        </pre>
      </CardContent>
    </Card>
  );
}
