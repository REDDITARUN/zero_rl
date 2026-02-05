"use client";

import { useEffect, useMemo, useState } from "react";

import { startEvaluation } from "@/lib/api";
import { connectSocket } from "@/lib/websocket";
import type { EvalPoint } from "@/lib/types";

export function useEval(envId: string | null) {
  const [events, setEvents] = useState<EvalPoint[]>([]);
  const [busy, setBusy] = useState(false);
  const [episodes, setEpisodes] = useState(1);
  const [maxSteps, setMaxSteps] = useState(250);

  useEffect(() => {
    return connectSocket("/ws/eval", (payload) => {
      if (typeof payload !== "object" || payload === null) {
        return;
      }
      const event = payload as EvalPoint;
      if (!event.env_id || (envId && event.env_id !== envId)) {
        return;
      }
      setEvents((prev) => [...prev, event]);
      if (event.status === "complete" || event.status === "error") {
        setBusy(false);
      }
    });
  }, [envId]);

  async function start() {
    if (!envId || busy) {
      return;
    }
    setBusy(true);
    setEvents([]);
    await startEvaluation(envId, episodes, maxSteps);
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
