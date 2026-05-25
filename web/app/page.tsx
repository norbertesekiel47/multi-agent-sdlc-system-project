import { LayoutShell } from "@/components/layout/layout-shell";
import { TaskSubmitForm } from "@/components/task/task-submit-form";
import { RecentTasks } from "@/components/task/recent-tasks";

export default function HomePage() {
  return (
    <LayoutShell>
      <div className="space-y-6">
        <TaskSubmitForm />
        <RecentTasks />
      </div>
    </LayoutShell>
  );
}
