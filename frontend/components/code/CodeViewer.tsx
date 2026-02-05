"use client";

import { useEffect, useMemo, useState } from "react";
import { FileTree } from "@/components/code/FileTree";
import { CodeBlock } from "@/components/ui/CodeBlock";
import { downloadEnvZip, getEnvironmentFile } from "@/lib/api";

interface CodeViewerProps {
  envId: string | null;
  files: string[];
}

export function CodeViewer({ envId, files }: CodeViewerProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const [code, setCode] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setSelected(files[0] ?? null);
  }, [files]);

  useEffect(() => {
    async function load() {
      if (!envId || !selected) {
        setCode("");
        return;
      }
      try {
        setError(null);
        const next = await getEnvironmentFile(envId, selected);
        setCode(next);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not load file");
      }
    }
    void load();
  }, [envId, selected]);

  const empty = useMemo(() => !envId || files.length === 0, [envId, files.length]);

  async function handleDownload() {
    if (!envId) {
      return;
    }
    try {
      const blob = await downloadEnvZip(envId);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${envId}.zip`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed");
    }
  }

  if (empty) {
    return <p className="p-4 text-sm text-[var(--soft-ink)]">Generate an environment to inspect code.</p>;
  }

  return (
    <div className="grid h-full grid-cols-1 gap-3 p-4 md:grid-cols-[260px_1fr]">
      <FileTree files={files} selected={selected} onSelect={setSelected} />
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <p className="text-xs uppercase tracking-[0.16em] text-[var(--soft-ink)]">{selected}</p>
          <button
            type="button"
            onClick={() => void handleDownload()}
            className="rounded-lg border border-[var(--border)] px-2 py-1 text-xs text-[var(--soft-ink)]"
          >
            Download zip
          </button>
        </div>
        {error ? <p className="text-xs text-red-700">{error}</p> : null}
        <CodeBlock code={code || "No content"} />
      </div>
    </div>
  );
}
