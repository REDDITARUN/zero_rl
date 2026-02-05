export function PlanCard() {
  return (
    <div className="mb-3 rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-3 text-xs text-[var(--soft-ink)]">
      <p className="mb-1 text-[10px] uppercase tracking-[0.18em]">Pipeline</p>
      <p>Phase 1: Architect, then Rewards + Spaces in parallel</p>
      <p>Phase 2: Validator with up to 3 targeted fixes</p>
      <p>Phase 3: Docs + Trainer generation in parallel</p>
    </div>
  );
}
