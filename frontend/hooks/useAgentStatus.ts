"use client";

import { useEffect, useMemo, useState } from "react";
import { connectSocket } from "@/lib/websocket";
import type { AgentState } from "@/lib/types";

const BASE_AGENTS: AgentState[] = [
  { id: "architect", name: "Architect", status: "idle", message: "Ready" },
  { id: "rewards", name: "Rewards", status: "idle", message: "Ready" },
  { id: "spaces", name: "Spaces", status: "idle", message: "Ready" },
  { id: "validator", name: "Validator", status: "idle", message: "Ready" },
  { id: "docs", name: "Docs", status: "idle", message: "Ready" },
  { id: "trainer", name: "Trainer", status: "idle", message: "Ready" }
];

export function useAgentStatus() {
  const [agents, setAgents] = useState<AgentState[]>(BASE_AGENTS);

  useEffect(() => {
    return connectSocket("/ws/agents", (payload) => {
      if (typeof payload !== "object" || payload === null) {
        return;
      }

      const event = payload as { agent_id?: string; status?: AgentState["status"]; message?: string };
      if (!event.agent_id || !event.status) {
        return;
      }

      setAgents((prev) =>
        prev.map((agent) =>
          agent.id === event.agent_id
            ? {
                ...agent,
                status: event.status ?? agent.status,
                message: event.message ?? agent.message
              }
            : agent
        )
      );
    });
  }, []);

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
