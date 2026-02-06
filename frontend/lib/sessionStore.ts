"use client";

import { create } from "zustand";

import type { EvalPoint, TrainingConfig, TrainingPoint } from "@/lib/types";

interface TrainingSession {
  events: TrainingPoint[];
  busy: boolean;
  config: TrainingConfig;
}

interface EvalSession {
  events: EvalPoint[];
  busy: boolean;
  episodes: number;
  maxSteps: number;
}

interface SessionStore {
  trainingByEnv: Record<string, TrainingSession>;
  evalByEnv: Record<string, EvalSession>;
  ensureTraining: (envId: string) => void;
  appendTrainingEvent: (envId: string, event: TrainingPoint) => void;
  setTrainingBusy: (envId: string, busy: boolean) => void;
  setTrainingConfig: (envId: string, config: TrainingConfig) => void;
  ensureEval: (envId: string) => void;
  appendEvalEvent: (envId: string, event: EvalPoint) => void;
  setEvalBusy: (envId: string, busy: boolean) => void;
  setEvalParams: (envId: string, episodes: number, maxSteps: number) => void;
  clearEnvSession: (envId: string) => void;
}

const DEFAULT_TRAINING: TrainingSession = {
  events: [],
  busy: false,
  config: {
    algorithm: "PPO",
    timesteps: 5000,
    learning_rate: 0.0003,
    gamma: 0.99,
    batch_size: 64,
    n_steps: 512,
    epsilon: 0.05
  }
};

const DEFAULT_EVAL: EvalSession = {
  events: [],
  busy: false,
  episodes: 1,
  maxSteps: 250
};

function cloneTraining(session?: TrainingSession): TrainingSession {
  if (!session) {
    return { ...DEFAULT_TRAINING, events: [], config: { ...DEFAULT_TRAINING.config } };
  }
  return {
    ...session,
    events: [...session.events],
    config: { ...session.config }
  };
}

function cloneEval(session?: EvalSession): EvalSession {
  if (!session) {
    return { ...DEFAULT_EVAL, events: [] };
  }
  return {
    ...session,
    events: [...session.events]
  };
}

export const useSessionStore = create<SessionStore>((set, get) => ({
  trainingByEnv: {},
  evalByEnv: {},
  ensureTraining: (envId) => {
    if (get().trainingByEnv[envId]) {
      return;
    }
    set((state) => ({
      trainingByEnv: {
        ...state.trainingByEnv,
        [envId]: cloneTraining()
      }
    }));
  },
  appendTrainingEvent: (envId, event) => {
    const session = cloneTraining(get().trainingByEnv[envId]);
    session.events.push(event);
    if (session.events.length > 400) {
      session.events = session.events.slice(-400);
    }
    if (event.status === "complete" || event.status === "error") {
      session.busy = false;
    }
    set((state) => ({
      trainingByEnv: {
        ...state.trainingByEnv,
        [envId]: session
      }
    }));
  },
  setTrainingBusy: (envId, busy) => {
    const session = cloneTraining(get().trainingByEnv[envId]);
    session.busy = busy;
    set((state) => ({
      trainingByEnv: {
        ...state.trainingByEnv,
        [envId]: session
      }
    }));
  },
  setTrainingConfig: (envId, config) => {
    const session = cloneTraining(get().trainingByEnv[envId]);
    session.config = { ...config };
    set((state) => ({
      trainingByEnv: {
        ...state.trainingByEnv,
        [envId]: session
      }
    }));
  },
  ensureEval: (envId) => {
    if (get().evalByEnv[envId]) {
      return;
    }
    set((state) => ({
      evalByEnv: {
        ...state.evalByEnv,
        [envId]: cloneEval()
      }
    }));
  },
  appendEvalEvent: (envId, event) => {
    const session = cloneEval(get().evalByEnv[envId]);
    session.events.push(event);
    if (session.events.length > 600) {
      session.events = session.events.slice(-600);
    }
    if (event.status === "complete" || event.status === "error") {
      session.busy = false;
    }
    set((state) => ({
      evalByEnv: {
        ...state.evalByEnv,
        [envId]: session
      }
    }));
  },
  setEvalBusy: (envId, busy) => {
    const session = cloneEval(get().evalByEnv[envId]);
    session.busy = busy;
    set((state) => ({
      evalByEnv: {
        ...state.evalByEnv,
        [envId]: session
      }
    }));
  },
  setEvalParams: (envId, episodes, maxSteps) => {
    const session = cloneEval(get().evalByEnv[envId]);
    session.episodes = episodes;
    session.maxSteps = maxSteps;
    set((state) => ({
      evalByEnv: {
        ...state.evalByEnv,
        [envId]: session
      }
    }));
  },
  clearEnvSession: (envId) => {
    const nextTraining = { ...get().trainingByEnv };
    const nextEval = { ...get().evalByEnv };
    delete nextTraining[envId];
    delete nextEval[envId];
    set({
      trainingByEnv: nextTraining,
      evalByEnv: nextEval
    });
  }
}));
