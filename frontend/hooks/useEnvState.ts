"use client";

import { useCallback, useEffect, useState } from "react";
import {
  getEnvironment,
  getEnvironmentState,
  resetEnvironment,
  stepEnvironment
} from "@/lib/api";
import type { EnvironmentMeta, RuntimeState } from "@/lib/types";

export function useEnvState(envId: string | null) {
  const [meta, setMeta] = useState<EnvironmentMeta | null>(null);
  const [runtime, setRuntime] = useState<RuntimeState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refreshMeta = useCallback(async () => {
    if (!envId) {
      setMeta(null);
      setRuntime(null);
      return;
    }
    try {
      const next = await getEnvironment(envId);
      setMeta(next);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load environment metadata");
    }
  }, [envId]);

  const refreshState = useCallback(async () => {
    if (!envId) {
      return;
    }
    try {
      const next = await getEnvironmentState(envId);
      setRuntime(next);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load environment state");
    }
  }, [envId]);

  useEffect(() => {
    void refreshMeta();
  }, [refreshMeta]);

  useEffect(() => {
    void refreshState();
  }, [refreshState]);

  const reset = useCallback(async () => {
    if (!envId) {
      return;
    }
    setBusy(true);
    try {
      const state = await resetEnvironment(envId);
      setRuntime(state);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reset environment");
    } finally {
      setBusy(false);
    }
  }, [envId]);

  const step = useCallback(
    async (action: number | string) => {
      if (!envId) {
        return;
      }
      setBusy(true);
      try {
        const state = await stepEnvironment(envId, action);
        setRuntime(state);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to step environment");
      } finally {
        setBusy(false);
      }
    },
    [envId]
  );

  return {
    meta,
    runtime,
    error,
    busy,
    refreshMeta,
    refreshState,
    reset,
    step
  };
}
