"use client";

import { useEffect, useMemo, useState } from "react";
import { connectSocket } from "@/lib/websocket";
import type { AgentState } from "@/lib/types";

const BASE_AGENTS: AgentState[] = [
  { id: "architect", name: "Architect", status: "idle", message: "Ready", logs: [] },
  { id: "rewards", name: "Rewards", status: "idle", message: "Ready", logs: [] },
  { id: "spaces", name: "Spaces", status: "idle", message: "Ready", logs: [] },
  { id: "validator", name: "Validator", status: "idle", message: "Ready", logs: [] },
  { id: "docs", name: "Docs", status: "idle", message: "Ready", logs: [] },
  { id: "trainer", name: "Trainer", status: "idle", message: "Ready", logs: [] }
];

export function useAgentStatus() {
  const [agents, setAgents] = useState<AgentState[]>(BASE_AGENTS);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);

  useEffect(() => {
    const onRunStart = (event: Event) => {
      const custom = event as CustomEvent<{ runId?: string }>;
      const runId = custom.detail?.runId ?? null;
      setActiveRunId(runId);
      setAgents(
        BASE_AGENTS.map((agent) => ({
          ...agent,
          status: "idle",
          message: "Ready",
          logs: []
        }))
      );
    };
    window.addEventListener("zerorl:run-start", onRunStart as EventListener);
    const disconnect = connectSocket("/ws/agents", (payload) => {
      if (typeof payload !== "object" || payload === null) {
        return;
      }

      const event = payload as { agent_id?: string; status?: AgentState["status"]; message?: string; run_id?: string | null };
      if (!event.agent_id || !event.status) {
        return;
      }
      if (!activeRunId || !event.run_id || event.run_id !== activeRunId) {
        return;
      }

      setAgents((prev) =>
        prev.map((agent) =>
          agent.id === event.agent_id
            ? {
                ...agent,
                status: event.status ?? agent.status,
                message: event.message ?? agent.message,
                logs: event.message
                  ? [...(agent.logs ?? []), event.message].slice(-6)
                  : (agent.logs ?? [])
              }
            : agent
        )
      );
    });
    return () => {
      window.removeEventListener("zerorl:run-start", onRunStart as EventListener);
      disconnect();
    };
  }, [activeRunId]);

  const summary = useMemo(
    () => ({
      active: agents.filter((agent) => agent.status === "working").length,
      done: agents.filter((agent) => agent.status === "complete").length,
      error: agents.filter((agent) => agent.status === "error").length
    }),
    [agents]
  );

  return { agents, summary };
}
