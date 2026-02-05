interface QuickActionsProps {
  disabled?: boolean;
  onSelect: (prompt: string) => void;
}

const ACTIONS = [
  "Create a maze where a robot finds the exit",
  "Create a collector world where the agent gathers 5 items",
  "Create a chase environment with a moving target",
  "Make rewards sparse, only on goal reach"
];

export function QuickActions({ disabled = false, onSelect }: QuickActionsProps) {
  return (
    <div className="mb-3 flex flex-wrap gap-2">
      {ACTIONS.map((action) => (
        <button
          key={action}
          disabled={disabled}
          onClick={() => onSelect(action)}
          className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 py-1 text-xs text-[var(--soft-ink)] transition hover:border-[var(--accent)] disabled:opacity-50"
          type="button"
        >
          {action}
        </button>
      ))}
    </div>
  );
}
