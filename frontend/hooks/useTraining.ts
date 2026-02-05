"use client";

import { useEffect, useMemo, useState } from "react";
import { startTraining } from "@/lib/api";
import type { TrainingConfig, TrainingPoint } from "@/lib/types";
import { connectSocket } from "@/lib/websocket";

const DEFAULT_CONFIG: TrainingConfig = {
  algorithm: "PPO",
  timesteps: 5000,
  learning_rate: 0.0003,
  gamma: 0.99,
  batch_size: 64,
  n_steps: 512,
  epsilon: 0.05
};

export function useTraining(envId: string | null) {
  const [events, setEvents] = useState<TrainingPoint[]>([]);
  const [busy, setBusy] = useState(false);
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);

  useEffect(() => {
    return connectSocket("/ws/train", (payload) => {
      if (typeof payload !== "object" || payload === null) {
        return;
      }
      const point = payload as TrainingPoint;
      if (!point.env_id || !point.status) {
        return;
      }
      if (envId && point.env_id !== envId) {
        return;
      }
      setEvents((prev) => [...prev, point]);
      if (point.status === "complete" || point.status === "error") {
        setBusy(false);
      }
    });
  }, [envId]);

  async function runTraining() {
    if (!envId || busy) {
      return;
    }
    setBusy(true);
    await startTraining(envId, config);
  }

  const latest = useMemo(() => events[events.length - 1] ?? null, [events]);

  return {
    events,
    latest,
    busy,
    config,
    setConfig,
    runTraining
  };
}
