"use client";

import { useMemo, useState } from "react";

import { CodeViewer } from "@/components/code/CodeViewer";
import { ActionSpaceTab } from "@/components/env/ActionSpaceTab";
import { EnvViewer } from "@/components/env/EnvViewer";
import { ObsSpaceTab } from "@/components/env/ObsSpaceTab";
import { RewardTab } from "@/components/env/RewardTab";
import { EvalPanel } from "@/components/eval/EvalPanel";
import { GalleryGrid } from "@/components/gallery/GalleryGrid";
import { TrainPanel } from "@/components/train/TrainPanel";
import { saveEnvironment } from "@/lib/api";
import { useEnvState } from "@/hooks/useEnvState";

interface TabPanelProps {
  envId: string | null;
  onEnvSelected: (envId: string | null) => void;
  onResetWorkspace: () => void;
}

type TabKey = "env" | "actions" | "obs" | "rewards" | "train" | "eval" | "code" | "gallery";

const TABS: { key: TabKey; label: string }[] = [
  { key: "env", label: "Env" },
  { key: "actions", label: "Actions" },
  { key: "obs", label: "Obs" },
  { key: "rewards", label: "Rewards" },
  { key: "train", label: "Train" },
  { key: "eval", label: "Eval" },
  { key: "code", label: "Code" },
  { key: "gallery", label: "Gallery" }
];

export function TabPanel({ envId, onEnvSelected, onResetWorkspace }: TabPanelProps) {
  const [active, setActive] = useState<TabKey>("env");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const { meta, runtime, error, busy, refreshMeta, refreshState, reset, step } = useEnvState(envId);

  const files = useMemo(() => {
    if (!meta?.files) {
      return [];
    }
    return Object.keys(meta.files);
  }, [meta]);

  const actions = meta?.action_space?.actions ?? [];
  const runtimeActions = runtime?.available_actions?.length ? runtime.available_actions : actions;
  const needsEnv = active !== "gallery";

  async function handleSave() {
    if (!envId || saving) {
      return;
    }
    setSaving(true);
    setSaveError(null);
    try {
      await saveEnvironment(envId);
      await refreshMeta();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Failed to save environment");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="scrollbar-thin flex items-center justify-between gap-2 overflow-x-auto border-b border-[var(--border)] p-3">
        <div className="flex gap-2">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActive(tab.key)}
              className={`rounded-full px-3 py-1 text-xs transition ${
                tab.key === active
                  ? "bg-[var(--accent)] text-white"
                  : "border border-[var(--border)] bg-[var(--surface)] text-[var(--soft-ink)]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          {envId ? (
            <button
              type="button"
              onClick={() => void handleSave()}
              disabled={saving || !!meta?.saved}
              className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-1 text-xs text-[var(--soft-ink)] disabled:opacity-50"
            >
              {meta?.saved ? "Saved" : saving ? "Saving..." : "Save To Gallery"}
            </button>
          ) : null}
          <button
            type="button"
            onClick={onResetWorkspace}
            className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-1 text-xs text-[var(--soft-ink)]"
          >
            Reset Workspace
          </button>
        </div>
      </div>

      {saveError ? <p className="px-3 pt-2 text-xs text-red-700">{saveError}</p> : null}

      <div className="min-h-0 flex-1 overflow-auto">
        {needsEnv && !envId ? (
          <div className="grid h-full place-items-center p-6 text-center">
            <div>
              <h3 className="text-xl text-[var(--ink)]">Create an environment to begin</h3>
              <p className="mt-1 text-sm text-[var(--soft-ink)]">Use the left chat panel to generate and iterate with Codex agents.</p>
            </div>
          </div>
        ) : null}

        {active === "env" && envId ? (
          <EnvViewer
            meta={meta}
            runtime={runtime}
            actions={runtimeActions}
            busy={busy}
            error={error}
            onRefresh={() => void refreshState()}
            onReset={() => void reset()}
            onAction={(action) => void step(action)}
          />
        ) : null}
        {active === "actions" && envId ? <ActionSpaceTab meta={meta} /> : null}
        {active === "obs" && envId ? <ObsSpaceTab meta={meta} runtime={runtime} /> : null}
        {active === "rewards" && envId ? <RewardTab meta={meta} runtime={runtime} /> : null}
        {active === "train" ? <TrainPanel envId={envId} /> : null}
        {active === "eval" ? <EvalPanel envId={envId} /> : null}
        {active === "code" ? <CodeViewer envId={envId} files={files} /> : null}
        {active === "gallery" ? <GalleryGrid onSelect={onEnvSelected} /> : null}
      </div>
    </div>
  );
}
