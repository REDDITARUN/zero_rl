import type { TrainingConfig } from "@/lib/types";

interface HyperparamControlsProps {
  config: TrainingConfig;
  onChange: (next: TrainingConfig) => void;
}

export function HyperparamControls({ config, onChange }: HyperparamControlsProps) {
  return (
    <div className="grid gap-3 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3 md:grid-cols-2">
      <label className="space-y-1 text-xs text-[var(--soft-ink)]">
        <span className="uppercase tracking-[0.16em]">Algorithm</span>
        <select
          value={config.algorithm}
          onChange={(event) => onChange({ ...config, algorithm: event.target.value as TrainingConfig["algorithm"] })}
          className="w-full rounded-lg border border-[var(--border)] bg-[#fffaf0] px-3 py-2 text-sm text-[var(--ink)]"
        >
          <option value="PPO">PPO</option>
          <option value="DQN">DQN</option>
          <option value="A2C">A2C</option>
        </select>
      </label>

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

      <label className="space-y-1 text-xs text-[var(--soft-ink)]">
        <span className="uppercase tracking-[0.16em]">Learning Rate</span>
        <input
          type="number"
          min={0.00001}
          step={0.00001}
          value={config.learning_rate}
          onChange={(event) => onChange({ ...config, learning_rate: Number(event.target.value) })}
          className="w-full rounded-lg border border-[var(--border)] bg-[#fffaf0] px-3 py-2 text-sm text-[var(--ink)]"
        />
      </label>

      <label className="space-y-1 text-xs text-[var(--soft-ink)]">
        <span className="uppercase tracking-[0.16em]">Gamma</span>
        <input
          type="number"
          min={0.1}
          max={0.999}
          step={0.001}
          value={config.gamma}
          onChange={(event) => onChange({ ...config, gamma: Number(event.target.value) })}
          className="w-full rounded-lg border border-[var(--border)] bg-[#fffaf0] px-3 py-2 text-sm text-[var(--ink)]"
        />
      </label>

      <label className="space-y-1 text-xs text-[var(--soft-ink)]">
        <span className="uppercase tracking-[0.16em]">Batch Size</span>
        <input
          type="number"
          min={8}
          step={8}
          value={config.batch_size}
          onChange={(event) => onChange({ ...config, batch_size: Number(event.target.value) })}
          className="w-full rounded-lg border border-[var(--border)] bg-[#fffaf0] px-3 py-2 text-sm text-[var(--ink)]"
        />
      </label>

      <label className="space-y-1 text-xs text-[var(--soft-ink)]">
        <span className="uppercase tracking-[0.16em]">n_steps</span>
        <input
          type="number"
          min={16}
          step={16}
          value={config.n_steps}
          onChange={(event) => onChange({ ...config, n_steps: Number(event.target.value) })}
          className="w-full rounded-lg border border-[var(--border)] bg-[#fffaf0] px-3 py-2 text-sm text-[var(--ink)]"
        />
      </label>

      <label className="space-y-1 text-xs text-[var(--soft-ink)] md:col-span-2">
        <span className="uppercase tracking-[0.16em]">Epsilon (for DQN)</span>
        <input
          type="number"
          min={0}
          max={1}
          step={0.01}
          value={config.epsilon}
          onChange={(event) => onChange({ ...config, epsilon: Number(event.target.value) })}
          className="w-full rounded-lg border border-[var(--border)] bg-[#fffaf0] px-3 py-2 text-sm text-[var(--ink)]"
        />
      </label>
    </div>
  );
}
