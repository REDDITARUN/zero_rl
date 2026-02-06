"use client";

import { useEffect, useMemo } from "react";
import { startTraining } from "@/lib/api";
import type { TrainingConfig, TrainingPoint } from "@/lib/types";
import { useSessionStore } from "@/lib/sessionStore";
import { connectSocket } from "@/lib/websocket";

export function useTraining(envId: string | null) {
  const trainingByEnv = useSessionStore((state) => state.trainingByEnv);
  const ensureTraining = useSessionStore((state) => state.ensureTraining);
  const appendTrainingEvent = useSessionStore((state) => state.appendTrainingEvent);
  const setTrainingBusy = useSessionStore((state) => state.setTrainingBusy);
  const setTrainingConfig = useSessionStore((state) => state.setTrainingConfig);

  useEffect(() => {
    if (envId) {
      ensureTraining(envId);
    }
  }, [envId, ensureTraining]);

  useEffect(() => {
    return connectSocket("/ws/train", (payload) => {
      if (typeof payload !== "object" || payload === null) {
        return;
      }
      const point = payload as TrainingPoint;
      if (!point.env_id || !point.status) {
        return;
      }
      if (!envId || point.env_id !== envId) {
        return;
      }
      appendTrainingEvent(point.env_id, point);
    });
  }, [appendTrainingEvent, envId]);

  const session = envId ? trainingByEnv[envId] : undefined;
  const events = session?.events ?? [];
  const busy = session?.busy ?? false;
  const config: TrainingConfig =
    session?.config ?? {
      algorithm: "PPO",
      timesteps: 5000,
      learning_rate: 0.0003,
      gamma: 0.99,
      batch_size: 64,
      n_steps: 512,
      epsilon: 0.05
    };

  async function runTraining() {
    if (!envId || busy) {
      return;
    }
    setTrainingBusy(envId, true);
    await startTraining(envId, { ...config, algorithm: "PPO" });
  }

  function setConfig(next: TrainingConfig) {
    if (!envId) {
      return;
    }
    setTrainingConfig(envId, next);
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
