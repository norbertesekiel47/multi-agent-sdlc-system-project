"use client";

import { use } from "react";
import Link from "next/link";
import { LayoutShell } from "@/components/layout/layout-shell";
import { HitlApproval } from "@/components/task/hitl-approval";
import { Button } from "@/components/ui/button";

export default function HitlPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);

  return (
    <LayoutShell>
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <Link href={`/tasks/${id}`}>
            <Button variant="ghost" size="sm">
              ← Back to task
            </Button>
          </Link>
          <h1 className="text-lg font-semibold">HITL Review</h1>
        </div>
        <HitlApproval taskId={id} />
      </div>
    </LayoutShell>
  );
}
