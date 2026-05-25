"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useCreateTask } from "@/hooks/use-tasks";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Topology } from "@/types/api";
import { ApiError } from "@/lib/api-client";

const TOPOLOGIES: { value: Topology; label: string }[] = [
  { value: "single_agent", label: "Single Agent" },
  { value: "supervisor_only", label: "Supervisor Only" },
  { value: "hybrid", label: "Hybrid" },
];

interface FormErrors {
  repo_url?: string;
  issue_number?: string;
  general?: string;
}

export function TaskSubmitForm() {
  const router = useRouter();
  const createTask = useCreateTask();

  const [repoUrl, setRepoUrl] = useState("");
  const [issueNumber, setIssueNumber] = useState("");
  const [topology, setTopology] = useState<Topology>("hybrid");
  const [errors, setErrors] = useState<FormErrors>({});

  function validate(): boolean {
    const newErrors: FormErrors = {};

    const trimmedUrl = repoUrl.trim();
    if (!trimmedUrl) {
      newErrors.repo_url = "Repository URL is required";
    } else if (
      !trimmedUrl.startsWith("https://") &&
      !trimmedUrl.startsWith("http://")
    ) {
      newErrors.repo_url = "Please enter a valid URL starting with https://";
    }

    const issueNum = issueNumber.trim();
    if (!issueNum) {
      newErrors.issue_number = "Issue number is required";
    } else if (!/^\d+$/.test(issueNum) || parseInt(issueNum, 10) <= 0) {
      newErrors.issue_number = "Issue number must be a positive integer";
    } else if (parseInt(issueNum, 10) > 2147483647) {
      newErrors.issue_number = "Issue number is too large";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    if (!validate()) return;

    try {
      const result = await createTask.mutateAsync({
        repo_url: repoUrl.trim(),
        issue_number: parseInt(issueNumber.trim(), 10),
        issue_text: `Issue #${issueNumber.trim()}`,
        topology,
      });

      router.push(`/tasks/${result.id}`);
    } catch (err) {
      if (err instanceof ApiError) {
        setErrors((prev) => ({
          ...prev,
          general: err.message,
        }));
      } else {
        setErrors((prev) => ({
          ...prev,
          general: "An unexpected error occurred. Please try again.",
        }));
      }
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Submit New Task</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="repo-url">Repository URL</Label>
            <Input
              id="repo-url"
              type="text"
              placeholder="https://github.com/org/repo"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              disabled={createTask.isPending}
              aria-describedby={
                errors.repo_url ? "repo-url-error" : undefined
              }
            />
            {errors.repo_url && (
              <p id="repo-url-error" className="text-sm text-destructive">
                {errors.repo_url}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="issue-number">Issue Number</Label>
            <Input
              id="issue-number"
              type="text"
              inputMode="numeric"
              placeholder="e.g. 42"
              value={issueNumber}
              onChange={(e) => setIssueNumber(e.target.value)}
              disabled={createTask.isPending}
              aria-describedby={
                errors.issue_number ? "issue-number-error" : undefined
              }
            />
            {errors.issue_number && (
              <p id="issue-number-error" className="text-sm text-destructive">
                {errors.issue_number}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="topology">Topology</Label>
            <Select
              value={topology}
              onValueChange={(v) => setTopology(v as Topology)}
              disabled={createTask.isPending}
            >
              <SelectTrigger id="topology" className="w-full">
                <SelectValue placeholder="Select topology" />
              </SelectTrigger>
              <SelectContent>
                {TOPOLOGIES.map((t) => (
                  <SelectItem key={t.value} value={t.value}>
                    {t.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {errors.general && (
            <div className="rounded-md bg-destructive/10 p-3">
              <p className="text-sm text-destructive">{errors.general}</p>
            </div>
          )}

          <Button
            type="submit"
            className="w-full"
            disabled={createTask.isPending}
          >
            {createTask.isPending ? "Submitting…" : "Submit Task"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
