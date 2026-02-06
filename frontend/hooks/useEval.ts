"use client";

import { useEffect, useMemo } from "react";

import { startEvaluation } from "@/lib/api";
import { useSessionStore } from "@/lib/sessionStore";
import { connectSocket } from "@/lib/websocket";
import type { EvalPoint } from "@/lib/types";

export function useEval(envId: string | null) {
  const evalByEnv = useSessionStore((state) => state.evalByEnv);
  const ensureEval = useSessionStore((state) => state.ensureEval);
  const appendEvalEvent = useSessionStore((state) => state.appendEvalEvent);
  const setEvalBusy = useSessionStore((state) => state.setEvalBusy);
  const setEvalParams = useSessionStore((state) => state.setEvalParams);

  useEffect(() => {
    if (envId) {
      ensureEval(envId);
    }
  }, [envId, ensureEval]);

  useEffect(() => {
    return connectSocket("/ws/eval", (payload) => {
      if (typeof payload !== "object" || payload === null) {
        return;
      }
      const event = payload as EvalPoint;
      if (!event.env_id || !envId || event.env_id !== envId) {
        return;
      }
      appendEvalEvent(event.env_id, event);
    });
  }, [appendEvalEvent, envId]);

  const session = envId ? evalByEnv[envId] : undefined;
  const events = session?.events ?? [];
  const busy = session?.busy ?? false;
  const episodes = session?.episodes ?? 1;
  const maxSteps = session?.maxSteps ?? 250;

  async function start() {
    if (!envId || busy) {
      return;
    }
    setEvalBusy(envId, true);
    await startEvaluation(envId, episodes, maxSteps);
  }

  function setEpisodes(next: number) {
    if (!envId) {
      return;
    }
    setEvalParams(envId, Math.max(1, next), maxSteps);
  }

  function setMaxSteps(next: number) {
    if (!envId) {
      return;
    }
    setEvalParams(envId, episodes, Math.max(10, next));
  }

  const latest = useMemo(() => events[events.length - 1] ?? null, [events]);

  return {
    events,
    latest,
    busy,
    episodes,
    setEpisodes,
    maxSteps,
    setMaxSteps,
    start
  };
}
