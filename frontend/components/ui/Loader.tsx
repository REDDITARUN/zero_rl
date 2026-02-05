export function Loader({ label = "Working" }: { label?: string }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 py-1 text-xs text-[var(--soft-ink)]">
      <span className="pulse-active inline-block h-2 w-2 rounded-full bg-[var(--accent)]" />
      {label}
    </div>
  );
}
