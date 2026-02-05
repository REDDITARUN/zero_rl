"use client";

import { useEffect, useState } from "react";
import { EnvCard } from "@/components/gallery/EnvCard";
import { listEnvironments } from "@/lib/api";
import type { EnvironmentSummary } from "@/lib/types";

interface GalleryGridProps {
  onSelect: (envId: string) => void;
}

export function GalleryGrid({ onSelect }: GalleryGridProps) {
  const [items, setItems] = useState<EnvironmentSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setItems(await listEnvironments());
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not load environments");
      }
    }
    void load();
  }, []);

  return (
    <div className="space-y-3 p-4">
      <h3 className="text-lg text-[var(--ink)]">Gallery</h3>
      {error ? <p className="text-xs text-red-700">{error}</p> : null}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((item) => (
          <EnvCard key={item.env_id} env={item} onSelect={onSelect} />
        ))}
      </div>
      {items.length === 0 ? <p className="text-sm text-[var(--soft-ink)]">No environments yet.</p> : null}
    </div>
  );
}
