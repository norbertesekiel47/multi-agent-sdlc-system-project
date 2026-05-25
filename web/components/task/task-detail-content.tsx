"use client";

import Link from "next/link";
import { useTraceStream } from "@/hooks/use-trace-stream";
import { TracePanel } from "@/components/task/trace-panel";
import { CostPanel } from "@/components/task/cost-panel";
import { DiffViewer } from "@/components/task/diff-viewer";
import { TopologyTransition } from "@/components/task/topology-transition";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useTask } from "@/hooks/use-tasks";

const statusColors: Record<string, string> = {
  running: "bg-blue-500/15 text-blue-500",
  awaiting_hitl: "bg-yellow-500/15 text-yellow-500",
  approved: "bg-green-500/15 text-green-500",
  rejected: "bg-red-500/15 text-red-500",
  completed: "bg-green-500/15 text-green-500",
  failed: "bg-red-500/15 text-red-500",
};

/** Inner component that uses the trace stream hook (needs taskId from loaded task) */
export function TaskDetailContent({
  id,
  task,
}: {
  id: string;
  task: NonNullable<ReturnType<typeof useTask>["data"]>;
}) {
  const { events } = useTraceStream(id);

  return (
    <div className="space-y-6">
      {/* Task header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <h1 className="text-xl font-bold">
            {task.repo_url.replace("https://github.com/", "")}
          </h1>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>#{task.issue_number ?? "—"}</span>
            <span>·</span>
            <span>{task.topology}</span>
          </div>
        </div>
        <Badge
          variant="outline"
          className={statusColors[task.status] ?? ""}
        >
          {task.status}
        </Badge>
      </div>

      {/* HITL link if awaiting */}
      {task.status === "awaiting_hitl" && (
        <Link href={`/tasks/${id}/hitl`}>
          <Button variant="outline" className="w-full">
            Review Pending Decision →
          </Button>
        </Link>
      )}

      {/* PR link if completed */}
      {task.pr_url && (
        <a
          href={task.pr_url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
        >
          View Pull Request →
        </a>
      )}

      {/* Topology transition visualization */}
      <TopologyTransition topology={task.topology} events={events} />

      {/* Two-column layout: trace + cost */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <TracePanel taskId={id} />
        </div>
        <div className="space-y-6">
          <CostPanel task={task} />
        </div>
      </div>

      {/* Diff viewer */}
      <DiffViewer />
    </div>
  );
}
