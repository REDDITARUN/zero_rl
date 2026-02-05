import type { AgentState } from "@/lib/types";

const STATUS_STYLES: Record<AgentState["status"], string> = {
  idle: "bg-zinc-300",
  working: "bg-amber-500 pulse-active",
  complete: "bg-emerald-600",
  error: "bg-red-600"
};

export function AgentCard({ agent }: { agent: AgentState }) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] px-3 py-2">
      <div className="mb-1 flex items-center gap-2">
        <span className={`inline-block h-2.5 w-2.5 rounded-full ${STATUS_STYLES[agent.status]}`} />
        <p className="text-xs font-medium text-[var(--ink)]">{agent.name}</p>
      </div>
      <p className="line-clamp-2 text-[11px] text-[var(--soft-ink)]">{agent.message}</p>
    </div>
  );
}
