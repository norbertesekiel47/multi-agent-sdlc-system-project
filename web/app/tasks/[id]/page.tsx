"use client";

import { use } from "react";
import Link from "next/link";
import { useTask } from "@/hooks/use-tasks";
import { useTraceStream } from "@/hooks/use-trace-stream";
import { LayoutShell } from "@/components/layout/layout-shell";
import { TracePanel } from "@/components/task/trace-panel";
import { CostPanel } from "@/components/task/cost-panel";
import { DiffViewer } from "@/components/task/diff-viewer";
import { TopologyTransition } from "@/components/task/topology-transition";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError } from "@/lib/api-client";

const statusColors: Record<string, string> = {
  running: "bg-blue-500/15 text-blue-500",
  awaiting_hitl: "bg-yellow-500/15 text-yellow-500",
  approved: "bg-green-500/15 text-green-500",
  rejected: "bg-red-500/15 text-red-500",
  completed: "bg-green-500/15 text-green-500",
  failed: "bg-red-500/15 text-red-500",
};

export default function TaskDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data: task, isLoading, error } = useTask(id);

  return (
    <LayoutShell>
      {error && error instanceof ApiError && error.status >= 500 ? (
        <div className="flex flex-col items-center gap-4 py-12">
          <p className="text-lg text-muted-foreground">
            Couldn&apos;t load this task
          </p>
          <Button
            variant="outline"
            onClick={() => window.location.reload()}
          >
            Retry
          </Button>
        </div>
      ) : isLoading ? (
        <TaskDetailSkeleton />
      ) : error ? (
        <div className="flex flex-col items-center gap-4 py-12">
          <p className="text-lg text-muted-foreground">
            Couldn&apos;t load this task
          </p>
          <Button
            variant="outline"
            onClick={() => window.location.reload()}
          >
            Retry
          </Button>
        </div>
      ) : task ? (
        <TaskDetailContent id={id} task={task} />
      ) : (
        <div className="flex flex-col items-center gap-4 py-12">
          <p className="text-lg text-muted-foreground">Task not found</p>
        </div>
      )}
    </LayoutShell>
  );
}

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

function TaskDetailSkeleton() {
  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <div className="h-6 w-48 animate-pulse rounded bg-muted" />
          <div className="h-4 w-32 animate-pulse rounded bg-muted" />
        </div>
        <div className="h-6 w-24 animate-pulse rounded bg-muted" />
      </div>
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <div className="h-64 w-full animate-pulse rounded bg-muted" />
        </div>
        <div className="h-48 w-full animate-pulse rounded bg-muted" />
      </div>
    </div>
  );
}
