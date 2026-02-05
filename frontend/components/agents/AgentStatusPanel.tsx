"use client";

import { AgentCard } from "@/components/agents/AgentCard";
import { useAgentStatus } from "@/hooks/useAgentStatus";

export function AgentStatusPanel() {
  const { agents, summary } = useAgentStatus();

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h2 className="text-sm text-[var(--ink)]">Parallel Agents</h2>
        <p className="text-[10px] uppercase tracking-[0.18em] text-[var(--soft-ink)]">
          Active {summary.active} | Done {summary.done} | Error {summary.error}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-2 xl:grid-cols-3">
        {agents.map((agent) => (
          <AgentCard key={agent.id} agent={agent} />
        ))}
      </div>
    </div>
  );
}
