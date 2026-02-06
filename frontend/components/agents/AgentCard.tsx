import type { AgentState } from "@/lib/types";

const STATUS_STYLES: Record<AgentState["status"], string> = {
  idle: "bg-zinc-300",
  working: "bg-amber-500 pulse-active",
  complete: "bg-emerald-600",
  error: "bg-red-600"
};

export function AgentCard({ agent }: { agent: AgentState }) {
  const logs = agent.logs ?? [];

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] px-3 py-2">
      <div className="mb-1 flex items-center gap-2">
        <span className={`inline-block h-2.5 w-2.5 rounded-full ${STATUS_STYLES[agent.status]}`} />
        <p className="text-xs font-medium text-[var(--ink)]">{agent.name}</p>
      </div>
      <p className="line-clamp-2 text-[11px] text-[var(--soft-ink)]">{agent.message}</p>
      {logs.length > 1 ? (
        <div className="mt-2 border-l-2 border-[var(--border)] pl-2">
          <p className="mb-1 text-[10px] uppercase tracking-[0.14em] text-[var(--soft-ink)]">Steps</p>
          <ul className="space-y-1">
            {logs.slice(-4).map((entry, idx) => (
              <li key={`${agent.id}-${idx}`} className="line-clamp-1 text-[10px] text-[var(--soft-ink)]">
                {entry}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
