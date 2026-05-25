"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useTraceStream } from "@/hooks/use-trace-stream";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
  if (ms < 0) return "—";
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

/** Build a tree from flat trace events using parent_span_id */
interface SpanNode {
  event: TraceEvent;
  children: SpanNode[];
}

function buildSpanTree(events: TraceEvent[]): SpanNode[] {
  const nodeMap = new Map<string, SpanNode>();
  const roots: SpanNode[] = [];

  // Create nodes for all events
  for (const event of events) {
    nodeMap.set(event.span_id, { event, children: [] });
  }

  // Build parent-child relationships
  for (const event of events) {
    const node = nodeMap.get(event.span_id);
    if (!node) continue;

    if (event.parent_span_id && nodeMap.has(event.parent_span_id)) {
      const parent = nodeMap.get(event.parent_span_id);
      parent?.children.push(node);
    } else {
      roots.push(node);
    }
  }

  return roots;
}

export function TracePanel({ taskId }: TracePanelProps) {
  const { events, status, error: wsError } = useTraceStream(taskId);
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Build span tree from events
  const spanTree = useMemo(() => buildSpanTree(events), [events]);

  const taskComplete = events.some((e) => e.type === "task_complete");

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

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-base">Trace Stream</CardTitle>
        <div className="flex items-center gap-2">
          {status === "connected" && !taskComplete && (
            <Badge variant="outline" className="text-xs text-green-400">
              Live
            </Badge>
          )}
          {status === "connecting" && (
            <Badge variant="outline" className="text-xs text-yellow-400">
              Connecting…
            </Badge>
          )}
          {status === "disconnected" && !taskComplete && (
            <Badge variant="outline" className="text-xs text-red-400">
              Reconnecting
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

          {spanTree.map((node) => (
            <SpanNodeRow key={node.event.span_id} node={node} depth={0} />
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

function SpanNodeRow({ node, depth }: { node: SpanNode; depth: number }) {
  const [expanded, setExpanded] = useState(true);
  const hasChildren = node.children.length > 0;
  const event = node.event;
  const icon = eventTypeLabels[event.type] ?? "•";
  const color = eventTypeColors[event.type] ?? "text-foreground";

  // Determine if this is a parent span (node_start/node_end can have children)
  const isParentSpan =
    event.type === "node_start" ||
    event.type === "node_end" ||
    hasChildren;

  const showTokenInfo =
    event.tokens_in !== null ||
    event.tokens_out !== null ||
    event.cached_tokens !== null ||
    event.cost_usd !== null;

  return (
    <div data-span-id={event.span_id} data-parent-span-id={event.parent_span_id}>
      <div
        className={`rounded px-2 py-1 text-sm hover:bg-muted/50 cursor-pointer`}
        style={{ paddingLeft: `${depth * 20 + 8}px` }}
        onClick={() => hasChildren && setExpanded(!expanded)}
        role={isParentSpan ? "button" : undefined}
        tabIndex={isParentSpan ? 0 : undefined}
        aria-expanded={isParentSpan ? expanded : undefined}
      >
        <div className="flex items-center gap-2">
          {hasChildren && (
            <span className="text-xs text-muted-foreground w-3 shrink-0">
              {expanded ? "▾" : "▸"}
            </span>
          )}
          {!hasChildren && <span className="w-3 shrink-0" />}
          <span className={color}>{icon}</span>
          <span className="font-medium truncate">{event.name}</span>
          {event.agent && (
            <Badge variant="outline" className="text-[10px] shrink-0">
              {event.agent}
            </Badge>
          )}
          <span className="ml-auto text-xs text-muted-foreground shrink-0">
            {formatLatency(event.started_at, event.ended_at)}
          </span>
        </div>
        {/* Token/cost info always visible for spans that have it */}
        {showTokenInfo && (
          <div
            className="mt-0.5 flex gap-3 text-xs text-muted-foreground"
            style={{ paddingLeft: "23px" }}
          >
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
      {/* Render children when expanded */}
      {expanded &&
        hasChildren &&
        node.children.map((child) => (
          <SpanNodeRow
            key={child.event.span_id}
            node={child}
            depth={depth + 1}
          />
        ))}
    </div>
  );
}
