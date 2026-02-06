import type { TrainingConfig } from "@/lib/types";

interface HyperparamControlsProps {
  config: TrainingConfig;
  onChange: (next: TrainingConfig) => void;
}

export function HyperparamControls({ config, onChange }: HyperparamControlsProps) {
  return (
    <div className="space-y-3 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3">
      <div className="grid gap-2 text-xs text-[var(--soft-ink)] md:grid-cols-2">
        <div className="rounded-lg border border-[var(--border)] bg-[#fbf6ed] px-3 py-2">
          <p className="uppercase tracking-[0.16em]">Algorithm</p>
          <p className="mt-1 text-sm text-[var(--ink)]">PPO</p>
        </div>
        <div className="rounded-lg border border-[var(--border)] bg-[#fbf6ed] px-3 py-2">
          <p className="uppercase tracking-[0.16em]">Learning Rate</p>
          <p className="mt-1 text-sm text-[var(--ink)]">{config.learning_rate}</p>
        </div>
        <div className="rounded-lg border border-[var(--border)] bg-[#fbf6ed] px-3 py-2">
          <p className="uppercase tracking-[0.16em]">Gamma</p>
          <p className="mt-1 text-sm text-[var(--ink)]">{config.gamma}</p>
        </div>
        <div className="rounded-lg border border-[var(--border)] bg-[#fbf6ed] px-3 py-2">
          <p className="uppercase tracking-[0.16em]">Batch Size</p>
          <p className="mt-1 text-sm text-[var(--ink)]">{config.batch_size}</p>
        </div>
      </div>

      <label className="space-y-1 text-xs text-[var(--soft-ink)]">
        <span className="uppercase tracking-[0.16em]">Timesteps</span>
        <input
          type="number"
          min={100}
          step={100}
          value={config.timesteps}
          onChange={(event) => onChange({ ...config, timesteps: Number(event.target.value) })}
          className="w-full rounded-lg border border-[var(--border)] bg-[#fffaf0] px-3 py-2 text-sm text-[var(--ink)]"
        />
      </label>
    </div>
  );
}
