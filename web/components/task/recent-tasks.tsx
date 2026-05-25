"use client";

import Link from "next/link";
import { useTasks } from "@/hooks/use-tasks";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { TaskStatus, Topology } from "@/types/api";

const statusColors: Record<TaskStatus, string> = {
  running: "bg-blue-500/15 text-blue-500",
  awaiting_hitl: "bg-yellow-500/15 text-yellow-500",
  approved: "bg-green-500/15 text-green-500",
  rejected: "bg-red-500/15 text-red-500",
  completed: "bg-green-500/15 text-green-500",
  failed: "bg-red-500/15 text-red-500",
};

function formatCost(cost: number | string | null): string {
  if (cost === null || cost === undefined) return "—";
  const numCost = typeof cost === "string" ? parseFloat(cost) : cost;
  if (isNaN(numCost)) return "—";
  return `$${numCost.toFixed(4)}`;
}

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function RecentTasks() {
  const { data, isLoading, error } = useTasks({ limit: 10 });

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Recent Tasks</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Unable to load tasks. Please try again.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Tasks</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : !data?.tasks.length ? (
          <p className="text-sm text-muted-foreground">
            No tasks yet. Submit your first task above.
          </p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Repo</TableHead>
                <TableHead>Issue</TableHead>
                <TableHead>Topology</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Cost</TableHead>
                <TableHead>Started</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.tasks.map((task) => (
                <TableRow key={task.id}>
                  <TableCell className="font-medium">
                    <Link
                      href={`/tasks/${task.id}`}
                      className="text-primary hover:underline"
                    >
                      {task.repo_url.replace("https://github.com/", "")}
                    </Link>
                  </TableCell>
                  <TableCell>#{task.issue_number ?? "—"}</TableCell>
                  <TableCell>
                    <span className="text-xs">{task.topology}</span>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant="outline"
                      className={statusColors[task.status]}
                    >
                      {task.status}
                    </Badge>
                  </TableCell>
                  <TableCell>{formatCost(task.total_cost_usd)}</TableCell>
                  <TableCell>{formatTime(task.started_at)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
