"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { RefreshCcw, Download } from "lucide-react";
import type { AssetParameter } from "@/hooks/useAssets";

function calcRange(defaultVal: number): { min: number; max: number; step: number } {
  const abs = Math.abs(defaultVal) || 1;
  const mag = Math.pow(10, Math.floor(Math.log10(abs)));
  const min = defaultVal >= 0 ? 0 : -abs * 3;
  const max = Math.max(abs * 3, mag * 10);
  const step = mag / 10 || 0.1;
  return { min: Math.round(min * 100) / 100, max: Math.round(max * 100) / 100, step };
}

function ParameterSlider({
  param,
  onChange,
}: {
  param: AssetParameter;
  onChange: (val: number) => void;
}) {
  const range = param.range || calcRange(Number(param.defaultValue) || 0);
  const val = Number(param.value) || 0;

  return (
    <div className="grid grid-cols-[72px_1fr_48px] items-center gap-2">
      <span className="truncate text-[10px] text-muted-foreground/60 font-mono">
        {param.displayName || param.name}
      </span>
      <input
        type="range"
        min={range.min}
        max={range.max}
        step={range.step}
        value={val}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-1 w-full cursor-pointer appearance-none rounded bg-border accent-primary"
      />
      <input
        type="number"
        value={val}
        step={range.step}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="w-full rounded border border-border bg-secondary px-1.5 py-0.5 text-[10px] font-mono text-foreground focus:border-primary focus:outline-none"
      />
    </div>
  );
}

function ParameterSwitch({
  param,
  onChange,
}: {
  param: AssetParameter;
  onChange: (val: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-[10px] text-muted-foreground/60 font-mono">
        {param.displayName || param.name}
      </span>
      <button
        onClick={() => onChange(!param.value)}
        className={`h-5 w-9 rounded-full transition-colors ${
          param.value ? "bg-primary" : "bg-border"
        }`}
      >
        <div
          className={`h-3.5 w-3.5 rounded-full bg-white shadow transition-transform ${
            param.value ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </button>
    </div>
  );
}

function ParameterText({
  param,
  onChange,
}: {
  param: AssetParameter;
  onChange: (val: string) => void;
}) {
  return (
    <div className="grid grid-cols-[72px_1fr] items-center gap-2">
      <span className="truncate text-[10px] text-muted-foreground/60 font-mono">
        {param.displayName || param.name}
      </span>
      <input
        type="text"
        value={String(param.value)}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded border border-border bg-secondary px-2 py-1 text-[10px] font-mono text-foreground focus:border-primary focus:outline-none"
      />
    </div>
  );
}

interface AssetParameterPanelProps {
  parameters: AssetParameter[];
  onParameterChange?: (params: AssetParameter[]) => void;
  onDownload?: () => void;
  downloadDisabled?: boolean;
}

export function AssetParameterPanel({
  parameters,
  onParameterChange,
  onDownload,
  downloadDisabled,
}: AssetParameterPanelProps) {
  const [params, setParams] = useState<AssetParameter[]>(parameters);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    setParams(parameters);
  }, [parameters]);

  const handleChange = useCallback(
    (name: string, value: AssetParameter["value"]) => {
      const updated = params.map((p) =>
        p.name === name ? { ...p, value } : p
      );
      setParams(updated);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        onParameterChange?.(updated);
      }, 300);
    },
    [params, onParameterChange]
  );

  const handleReset = () => {
    const reset = params.map((p) => ({ ...p, value: p.defaultValue }));
    setParams(reset);
    onParameterChange?.(reset);
  };

  if (params.length === 0) return null;

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card/50 p-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-widest text-muted-foreground/50 font-mono">
          parameters
        </span>
        <button
          onClick={handleReset}
          className="rounded p-1 text-muted-foreground/40 hover:text-foreground transition-colors"
          title="Reset all"
        >
          <RefreshCcw size={10} />
        </button>
      </div>

      <div className="space-y-2.5 max-h-[300px] overflow-y-auto">
        {params.map((p) => {
          if (p.type === "boolean") {
            return (
              <ParameterSwitch
                key={p.name}
                param={p}
                onChange={(v) => handleChange(p.name, v)}
              />
            );
          }
          if (p.type === "string") {
            return (
              <ParameterText
                key={p.name}
                param={p}
                onChange={(v) => handleChange(p.name, v)}
              />
            );
          }
          return (
            <ParameterSlider
              key={p.name}
              param={p}
              onChange={(v) => handleChange(p.name, v)}
            />
          );
        })}
      </div>

      {onDownload && (
        <button
          onClick={onDownload}
          disabled={downloadDisabled}
          className="flex items-center justify-center gap-1.5 rounded-md border border-border bg-secondary px-3 py-2 text-[10px] font-mono text-muted-foreground hover:text-foreground transition-colors disabled:opacity-30"
        >
          <Download size={10} /> download stl
        </button>
      )}
    </div>
  );
}
