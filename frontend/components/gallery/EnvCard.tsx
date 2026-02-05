import type { EnvironmentSummary } from "@/lib/types";

interface EnvCardProps {
  env: EnvironmentSummary;
  onSelect: (envId: string) => void;
}

export function EnvCard({ env, onSelect }: EnvCardProps) {
  return (
    <button
      onClick={() => onSelect(env.env_id)}
      type="button"
      className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3 text-left hover:border-[var(--accent)]"
    >
      <p className="text-sm font-semibold text-[var(--ink)]">{env.name}</p>
      <p className="mt-1 line-clamp-2 text-xs text-[var(--soft-ink)]">{env.prompt}</p>
      <p className="mt-2 text-[10px] uppercase tracking-[0.16em] text-[var(--soft-ink)]">{new Date(env.created_at).toLocaleString()}</p>
    </button>
  );
}
