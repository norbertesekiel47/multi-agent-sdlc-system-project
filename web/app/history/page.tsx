import { Suspense } from "react";
import { LayoutShell } from "@/components/layout/layout-shell";
import { HistoryList } from "@/components/task/history-list";
import { Skeleton } from "@/components/ui/skeleton";

function HistoryFallback() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 5 }).map((_, i) => (
        <Skeleton key={i} className="h-12 w-full" />
      ))}
    </div>
  );
}

export default function HistoryPage() {
  return (
    <LayoutShell>
      <Suspense fallback={<HistoryFallback />}>
        <HistoryList />
      </Suspense>
    </LayoutShell>
  );
}
