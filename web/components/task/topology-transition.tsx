"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Topology, TraceEvent } from "@/types/api";

interface TopologyTransitionProps {
  topology: Topology;
  events: TraceEvent[];
}

/** Agent names in execution order for each topology */
const AGENT_SEQUENCES: Record<Topology, string[]> = {
  single_agent: ["agent"],
  supervisor_only: ["planner", "coder", "reviewer", "qa"],
  hybrid: ["planner", "coder", "reviewer", "qa"],
};

/** Back-edge pairs for hybrid topology (Coder ⇄ Reviewer) */
const HYBRID_BACK_EDGES: Array<[number, number]> = [[2, 1]]; // reviewer → coder

function getAgentColor(agent: string): string {
  const colors: Record<string, string> = {
    planner: "bg-blue-500",
    coder: "bg-purple-500",
    reviewer: "bg-yellow-500",
    qa: "bg-green-500",
    agent: "bg-blue-500",
  };
  return colors[agent] ?? "bg-gray-500";
}

export function TopologyTransition({
  topology,
  events,
}: TopologyTransitionProps) {
  const agents = AGENT_SEQUENCES[topology];
  const backEdges = topology === "hybrid" ? HYBRID_BACK_EDGES : [];

  // Determine which agents have been seen in the trace events
  const seenAgents = new Set<string>();
  for (const event of events) {
    if (event.agent) {
      seenAgents.add(event.agent);
    }
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Agent Flow</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-2 overflow-x-auto py-2">
          {agents.map((agent, idx) => {
            const isActive = seenAgents.has(agent);
            const isHybridBackEdgeSource =
              topology === "hybrid" &&
              backEdges.some(([from]) => from === idx);

            return (
              <div key={`${agent}-${idx}`} className="flex items-center">
                <div className="flex flex-col items-center gap-1">
                  <div
                    className={`flex h-10 w-10 items-center justify-center rounded-full text-xs font-bold text-white ${
                      isActive
                        ? getAgentColor(agent)
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {agent.slice(0, 2).toUpperCase()}
                  </div>
                  <span className="text-[10px] text-muted-foreground capitalize">
                    {agent}
                  </span>
                  {isActive && (
                    <Badge
                      variant="outline"
                      className="text-[9px] text-green-400"
                    >
                      ✓
                    </Badge>
                  )}
                </div>
                {/* Arrow to next agent */}
                {idx < agents.length - 1 && (
                  <div className="mx-1 flex items-center text-muted-foreground">
                    →
                  </div>
                )}
              </div>
            );
          })}
        </div>
        {/* Show back-edge indicator for hybrid topology */}
        {topology === "hybrid" && (
          <div className="mt-2 text-xs text-muted-foreground">
            <span className="text-yellow-400">↻</span> Coder ⇄ Reviewer
            peer handoff enabled
          </div>
        )}
        {topology === "single_agent" && (
          <div className="mt-2 text-xs text-muted-foreground">
            Single agent handles all steps
          </div>
        )}
      </CardContent>
    </Card>
  );
}
