"use client";

import { useEffect, useRef, useState } from "react";
import { useTraceStream } from "@/hooks/use-trace-stream";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { TraceEvent, TraceEventType } from "@/types/api";

const eventTypeLabels: Record<TraceEventType, string> = {
  node_start: "▶",
  node_end: "■",
  tool_call: "🔧",
  llm_completion: "🤖",
  task_complete: "✅",
  ping: "📡",
  error: "❌",
};

const eventTypeColors: Record<TraceEventType, string> = {
  node_start: "text-blue-400",
  node_end: "text-blue-300",
  tool_call: "text-yellow-400",
  llm_completion: "text-purple-400",
  task_complete: "text-green-400",
  ping: "text-muted-foreground",
  error: "text-destructive",
};

function formatLatency(startedAt: string | null, endedAt: string | null): string {
  if (!startedAt || !endedAt) return "—";
  const ms = new Date(endedAt).getTime() - new Date(startedAt).getTime();
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatTokens(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString();
}

interface TracePanelProps {
  taskId: string;
}

export function TracePanel({ taskId }: TracePanelProps) {
  const { events, status, error: wsError } = useTraceStream(taskId);
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [events, autoScroll]);

  // Detect when user scrolls up (pause auto-scroll)
  function handleScroll() {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 50;
    setAutoScroll(isNearBottom);
  }

  function jumpToLatest() {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
      setAutoScroll(true);
    }
  }

  const taskComplete = events.some((e) => e.type === "task_complete");

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-base">Trace Stream</CardTitle>
        <div className="flex items-center gap-2">
          {status === "connected" && (
            <Badge variant="outline" className="text-xs text-green-400">
              Live
            </Badge>
          )}
          {status === "connecting" && (
            <Badge variant="outline" className="text-xs text-yellow-400">
              Connecting…
            </Badge>
          )}
          {status === "disconnected" && (
            <Badge variant="outline" className="text-xs text-red-400">
              Disconnected
            </Badge>
          )}
          {taskComplete && (
            <Badge variant="outline" className="text-xs text-green-400">
              Complete
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {wsError && (
          <p className="mb-2 text-sm text-destructive">{wsError}</p>
        )}

        <div
          ref={containerRef}
          onScroll={handleScroll}
          className="max-h-96 space-y-1 overflow-y-auto"
        >
          {events.length === 0 && status === "connected" && (
            <p className="py-4 text-center text-sm text-muted-foreground">
              Waiting for first agent span…
            </p>
          )}

          {events.map((event, idx) => (
            <TraceEventRow key={`${event.span_id}-${idx}`} event={event} />
          ))}
        </div>

        {!autoScroll && events.length > 0 && (
          <div className="mt-2 flex justify-center">
            <button
              type="button"
              onClick={jumpToLatest}
              className="rounded-full bg-primary px-3 py-1 text-xs text-primary-foreground hover:opacity-90"
            >
              ↓ Jump to latest
            </button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function TraceEventRow({ event }: { event: TraceEvent }) {
  const [expanded, setExpanded] = useState(false);
  const hasChildren = event.type === "node_start" || event.type === "node_end";
  const icon = eventTypeLabels[event.type] ?? "•";
  const color = eventTypeColors[event.type] ?? "text-foreground";

  return (
    <div
      className="rounded px-2 py-1 text-sm hover:bg-muted/50"
      onClick={() => hasChildren && setExpanded(!expanded)}
      role={hasChildren ? "button" : undefined}
      tabIndex={hasChildren ? 0 : undefined}
    >
      <div className="flex items-center gap-2">
        <span className={color}>{icon}</span>
        <span className="font-medium">{event.name}</span>
        {event.agent && (
          <Badge variant="outline" className="text-[10px]">
            {event.agent}
          </Badge>
        )}
        <span className="ml-auto text-xs text-muted-foreground">
          {formatLatency(event.started_at, event.ended_at)}
        </span>
      </div>
      {expanded && (event.tokens_in !== null || event.tokens_out !== null) && (
        <div className="mt-1 ml-6 flex gap-3 text-xs text-muted-foreground">
          {event.tokens_in !== null && (
            <span>In: {formatTokens(event.tokens_in)}</span>
          )}
          {event.tokens_out !== null && (
            <span>Out: {formatTokens(event.tokens_out)}</span>
          )}
          {event.cached_tokens !== null && event.cached_tokens !== undefined && (
            <span>Cached: {formatTokens(event.cached_tokens)}</span>
          )}
          {event.cost_usd !== null && (
            <span>Cost: ${event.cost_usd.toFixed(4)}</span>
          )}
        </div>
      )}
    </div>
  );
}
