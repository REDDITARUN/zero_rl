"use client";

import { useCallback, useEffect, useState } from "react";
import type { EnvironmentMeta, EnvDetails } from "@/types";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useEnvs() {
  const [envs, setEnvs] = useState<EnvironmentMeta[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/envs`);
      if (res.ok) {
        const data = await res.json();
        setEnvs(data);
      }
    } catch { /* backend offline */ }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const getEnv = useCallback(async (id: string): Promise<EnvDetails | null> => {
    try {
      const res = await fetch(`${API}/api/envs/${id}`);
      if (res.ok) return res.json();
    } catch { /* */ }
    return null;
  }, []);

  const deleteEnv = useCallback(
    async (id: string) => {
      try {
        await fetch(`${API}/api/envs/${id}`, { method: "DELETE" });
        setEnvs((prev) => prev.filter((e) => e.id !== id));
      } catch { /* */ }
    },
    []
  );

  return { envs, loading, refresh, getEnv, deleteEnv };
}
