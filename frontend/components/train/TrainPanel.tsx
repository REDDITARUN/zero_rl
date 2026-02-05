"use client";

import { useMemo } from "react";
import { HyperparamControls } from "@/components/train/HyperparamControls";
import { ProgressChart } from "@/components/train/ProgressChart";
import { useTraining } from "@/hooks/useTraining";

export function TrainPanel({ envId }: { envId: string | null }) {
  const { events, latest, busy, config, setConfig, runTraining } = useTraining(envId);

  const title = useMemo(() => {
    if (!envId) {
      return "No environment selected";
    }
    return `Training ${envId.slice(0, 8)}`;
  }, [envId]);

  return (
    <div className="space-y-3 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg text-[var(--ink)]">{title}</h3>
        <button
          disabled={!envId || busy}
          onClick={() => void runTraining()}
          className="rounded-lg bg-[var(--accent)] px-4 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-50"
          type="button"
        >
          {busy ? "Training..." : "Start Training"}
        </button>
      </div>

      <HyperparamControls config={config} onChange={setConfig} />

      {latest ? (
        <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3 text-sm text-[var(--soft-ink)]">
          <p>Status: {latest.status}</p>
          <p>
            Episode {latest.episode} | Reward {latest.reward.toFixed(3)} | Avg100 {latest.avg_reward_100.toFixed(3)}
          </p>
          <p>{latest.message}</p>
        </div>
      ) : null}

      <ProgressChart events={events} />
    </div>
  );
}
