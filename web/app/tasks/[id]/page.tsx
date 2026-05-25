"use client";

import { use } from "react";
import { useTask } from "@/hooks/use-tasks";
import { LayoutShell } from "@/components/layout/layout-shell";
import { TaskDetailContent } from "@/components/task/task-detail-content";
import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api-client";

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
