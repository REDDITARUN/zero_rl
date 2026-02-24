"use client";

import { useCallback, useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type AssetMeta = {
  id: string;
  name: string;
  description: string;
  file_types: string[];
  files: string[];
  has_params: boolean;
};

export type AssetDetails = {
  id: string;
  description: string;
  files: Record<string, string>;
  binary_files: string[];
  parameters: AssetParameter[];
};

export type AssetParameter = {
  name: string;
  displayName?: string;
  value: number | string | boolean;
  defaultValue: number | string | boolean;
  type?: "number" | "string" | "boolean";
  range?: { min?: number; max?: number; step?: number };
};

export function useAssets() {
  const [assets, setAssets] = useState<AssetMeta[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/assets`);
      if (r.ok) setAssets(await r.json());
    } catch {
      /* silent */
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const getAsset = useCallback(async (id: string): Promise<AssetDetails | null> => {
    try {
      const r = await fetch(`${API}/api/assets/${encodeURIComponent(id)}`);
      if (r.ok) return await r.json();
    } catch {
      /* silent */
    }
    return null;
  }, []);

  return { assets, loading, refresh, getAsset };
}
