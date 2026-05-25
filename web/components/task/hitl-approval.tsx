"use client";

import { useState } from "react";
import { useTask, useHitlDecision } from "@/hooks/use-tasks";
import { DiffViewer } from "@/components/task/diff-viewer";
import { CostPanel } from "@/components/task/cost-panel";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { ApiError } from "@/lib/api-client";

/** Map of HITL cause values to human-readable labels */
const HITL_CAUSE_LABELS: Record<string, string> = {
  loop_detected: "Loop Detected",
  uncertainty_escalation: "Uncertainty Escalation",
  retry_budget_exhausted: "Retry Budget Exhausted",
  guardrail_block: "Guardrail Block",
  cost_budget_exhausted: "Cost Budget Exhausted",
};

/** Map of HITL cause values to badge color classes */
const HITL_CAUSE_COLORS: Record<string, string> = {
  loop_detected: "bg-orange-500/15 text-orange-400",
  uncertainty_escalation: "bg-purple-500/15 text-purple-400",
  retry_budget_exhausted: "bg-red-500/15 text-red-400",
  guardrail_block: "bg-red-500/15 text-red-400",
  cost_budget_exhausted: "bg-red-500/15 text-red-400",
};

/** Safely escape HTML to prevent XSS when rendering user-provided text */
function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

interface HitlApprovalProps {
  taskId: string;
}

export function HitlApproval({ taskId }: HitlApprovalProps) {
  const { data: task, isLoading, error } = useTask(taskId);
  const hitlDecision = useHitlDecision();
  const [rejectReason, setRejectReason] = useState("");
  const [decisionError, setDecisionError] = useState<string | null>(null);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="h-8 w-48 animate-pulse rounded bg-muted" />
        <div className="h-64 animate-pulse rounded bg-muted" />
      </div>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="py-8">
          <p className="text-center text-muted-foreground">
            Unable to load task data. Please try again.
          </p>
        </CardContent>
      </Card>
    );
  }

  if (!task) {
    return (
      <Card>
        <CardContent className="py-8">
          <p className="text-center text-muted-foreground">
            Task not found.
          </p>
        </CardContent>
      </Card>
    );
  }

  const isAwaitingHitl = task.status === "awaiting_hitl";
  const decisionAlreadyMade = task.hitl_decision !== null;
  const isPending = hitlDecision.isPending;

  async function handleApprove() {
    setDecisionError(null);
    try {
      await hitlDecision.mutateAsync({
        taskId,
        data: { decision: "approve" },
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setDecisionError("A decision has already been made for this task.");
      } else if (err instanceof ApiError) {
        setDecisionError(err.message);
      } else {
        setDecisionError("An unexpected error occurred.");
      }
    }
  }

  async function handleReject() {
    setDecisionError(null);
    try {
      await hitlDecision.mutateAsync({
        taskId,
        data: {
          decision: "reject",
          reason: rejectReason.trim() || null,
        },
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setDecisionError("A decision has already been made for this task.");
      } else if (err instanceof ApiError) {
        setDecisionError(err.message);
      } else {
        setDecisionError("An unexpected error occurred.");
      }
    }
  }

  // Not awaiting HITL and no decision made — show appropriate message
  if (!isAwaitingHitl && !decisionAlreadyMade) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Human-in-the-Loop Review</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-muted-foreground">
            No pending decision. This task is currently{" "}
            <Badge variant="outline">{task.status}</Badge>.
          </div>
        </CardContent>
      </Card>
    );
  }

  // Decision already made — show final state
  if (decisionAlreadyMade) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Decision Made</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              This task has already been{" "}
              <Badge
                variant="outline"
                className={
                  task.hitl_decision === "approve"
                    ? "text-green-400"
                    : "text-red-400"
                }
              >
                {task.hitl_decision === "approve" ? "Approved" : "Rejected"}
              </Badge>
              .
            </div>
            {task.pr_url && (
              <a
                href={task.pr_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-primary hover:underline"
              >
                View Pull Request →
              </a>
            )}
            {task.reject_reason && (
              <div className="space-y-1">
                <span className="text-sm text-muted-foreground">Reject reason:</span>
                {/* Render escaped text to prevent XSS */}
                <p
                  className="text-sm italic"
                  dangerouslySetInnerHTML={{
                    __html: escapeHtml(task.reject_reason),
                  }}
                />
              </div>
            )}
          </CardContent>
        </Card>
        <CostPanel task={task} />
      </div>
    );
  }

  // Awaiting HITL — show approval UI
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle>Human-in-the-Loop Review</CardTitle>
            <Badge variant="outline" className="text-yellow-400">
              Awaiting Decision
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Task info */}
          <div className="text-sm">
            <span className="text-muted-foreground">Task: </span>
            <span className="font-mono">{task.repo_url}</span>
            <span className="text-muted-foreground"> #{task.issue_number}</span>
          </div>

          {/* HITL cause (for non-PR escalations: loop_detected, uncertainty, etc.) */}
          {task.hitl_cause && (
            <div className="space-y-2">
              <Separator />
              <div className="flex items-center gap-2">
                <Badge
                  variant="outline"
                  className={HITL_CAUSE_COLORS[task.hitl_cause] ?? "bg-yellow-500/15 text-yellow-400"}
                >
                  {HITL_CAUSE_LABELS[task.hitl_cause] ?? task.hitl_cause}
                </Badge>
                <span className="text-sm text-muted-foreground">
                  This review was triggered by an escalation event.
                </span>
              </div>
              {task.hitl_cause_detail && (
                <div className="rounded-md bg-muted/50 p-3 text-xs font-mono space-y-1">
                  {Object.entries(task.hitl_cause_detail).map(([key, value]) => (
                    <div key={key}>
                      <span className="text-muted-foreground">{key}: </span>
                      <span className="text-foreground">{String(value)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Review summary */}
          {task.review_summary && (
            <div className="space-y-1">
              <h3 className="text-sm font-semibold">Review Summary</h3>
              <pre className="whitespace-pre-wrap text-sm text-muted-foreground rounded-md bg-muted/30 p-3">
                {task.review_summary}
              </pre>
            </div>
          )}

          {/* Test summary */}
          {task.test_summary && (
            <div className="space-y-1">
              <h3 className="text-sm font-semibold">Test Summary</h3>
              <pre className="whitespace-pre-wrap text-sm text-muted-foreground rounded-md bg-muted/30 p-3">
                {task.test_summary}
              </pre>
            </div>
          )}

          {/* Diff viewer */}
          <DiffViewer diff={task.pending_diff} />

          {/* Decision buttons */}
          <div className="space-y-3 pt-2">
            <div className="flex gap-3">
              <Button
                onClick={handleApprove}
                disabled={isPending}
                className="bg-green-600 hover:bg-green-700"
              >
                {isPending ? "Submitting…" : "Approve"}
              </Button>
              <Button
                variant="destructive"
                onClick={handleReject}
                disabled={isPending}
              >
                {isPending ? "Submitting…" : "Reject"}
              </Button>
            </div>

            {/* Reject reason */}
            <div className="space-y-2">
              <Label htmlFor="reject-reason">
                Reject reason (optional)
              </Label>
              <Textarea
                id="reject-reason"
                placeholder="Explain why you are rejecting this change…"
                value={rejectReason}
                onChange={(e) => setRejectReason(e.target.value)}
                disabled={isPending}
                rows={3}
              />
            </div>

            {/* Error message */}
            {decisionError && (
              <div className="rounded-md bg-destructive/10 p-3">
                <p className="text-sm text-destructive">{decisionError}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Frozen cost panel during HITL */}
      <CostPanel task={task} />
    </div>
  );
}
