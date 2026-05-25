import { LayoutShell } from "@/components/layout/layout-shell";
import { HistoryList } from "@/components/task/history-list";

export default function HistoryPage() {
  return (
    <LayoutShell>
      <HistoryList />
    </LayoutShell>
  );
}
