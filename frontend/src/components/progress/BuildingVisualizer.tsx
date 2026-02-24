"use client";

import { useEffect, useRef, useMemo, useState } from "react";
import { animate, stagger } from "animejs";
import { CheckCircle, XCircle, Loader2 } from "lucide-react";
import type { ToolEvent, PipelineStage } from "@/hooks/useChat";

const STAGE_META: Record<
  string,
  { label: string; sub: string; color: string; glow: string }
> = {
  plan: {
    label: "planning",
    sub: "analyzing prompt and designing environment",
    color: "#67e8f9",
    glow: "rgba(103,232,249,0.12)",
  },
  write: {
    label: "generating",
    sub: "writing environment code and assets",
    color: "#3b82f6",
    glow: "rgba(59,130,246,0.12)",
  },
  validate: {
    label: "validating",
    sub: "running checks and simulations",
    color: "#f97316",
    glow: "rgba(249,115,22,0.12)",
  },
  render: {
    label: "rendering",
    sub: "capturing environment frames",
    color: "#a78bfa",
    glow: "rgba(167,139,250,0.12)",
  },
  error: {
    label: "fixing",
    sub: "diagnosing and repairing errors",
    color: "#ef4444",
    glow: "rgba(239,68,68,0.12)",
  },
  done: {
    label: "complete",
    sub: "environment is ready",
    color: "#34d399",
    glow: "rgba(52,211,153,0.12)",
  },
};

function GearTeeth({
  cx,
  cy,
  r,
  teeth,
  toothDepth,
  color,
  opacity = 0.15,
}: {
  cx: number;
  cy: number;
  r: number;
  teeth: number;
  toothDepth: number;
  color: string;
  opacity?: number;
}) {
  const d = useMemo(() => {
    const pts: string[] = [];
    for (let i = 0; i < teeth; i++) {
      const a1 = (i / teeth) * Math.PI * 2;
      const a2 = ((i + 0.35) / teeth) * Math.PI * 2;
      const a3 = ((i + 0.65) / teeth) * Math.PI * 2;
      const a4 = ((i + 1) / teeth) * Math.PI * 2;
      const inner = r - toothDepth;
      pts.push(
        `${cx + Math.cos(a1) * inner},${cy + Math.sin(a1) * inner}`,
        `${cx + Math.cos(a2) * r},${cy + Math.sin(a2) * r}`,
        `${cx + Math.cos(a3) * r},${cy + Math.sin(a3) * r}`,
        `${cx + Math.cos(a4) * inner},${cy + Math.sin(a4) * inner}`
      );
    }
    return `M${pts.join("L")}Z`;
  }, [cx, cy, r, teeth, toothDepth]);

  return <path d={d} fill="none" stroke={color} strokeWidth="0.6" opacity={opacity} />;
}

function MechanicalCore({ color, progress }: { color: string; progress: number }) {
  const C = 100;
  const arcR = 88;
  const arcLen = 2 * Math.PI * arcR;
  const filled = arcLen * Math.min(progress, 1);

  return (
    <svg viewBox="0 0 200 200" className="h-44 w-44" style={{ filter: `drop-shadow(0 0 20px ${color}15)` }}>
      <defs>
        <radialGradient id="core-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={color} stopOpacity="0.15" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Ambient glow */}
      <circle cx={C} cy={C} r="90" fill="url(#core-glow)">
        <animate attributeName="r" values="85;95;85" dur="4s" repeatCount="indefinite" />
      </circle>

      {/* Outer gear ring — slow CW */}
      <g>
        <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="360 100 100" dur="60s" repeatCount="indefinite" />
        <GearTeeth cx={C} cy={C} r={92} teeth={48} toothDepth={3} color={color} opacity={0.1} />
        <circle cx={C} cy={C} r={89} fill="none" stroke={color} strokeWidth="0.3" opacity="0.08" />
      </g>

      {/* Progress arc */}
      <circle
        cx={C} cy={C} r={arcR}
        fill="none" stroke={color} strokeWidth="1.5"
        strokeDasharray={`${filled} ${arcLen - filled}`}
        strokeDashoffset={arcLen * 0.25}
        strokeLinecap="round"
        opacity="0.5"
        style={{ transition: "stroke-dasharray 0.6s ease" }}
      />
      <circle
        cx={C} cy={C} r={arcR}
        fill="none" stroke={color} strokeWidth="0.3" opacity="0.1"
        strokeDasharray={`${arcLen}`}
        strokeDashoffset={arcLen * 0.25}
      />

      {/* Middle gear — CCW */}
      <g>
        <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="-360 100 100" dur="30s" repeatCount="indefinite" />
        <GearTeeth cx={C} cy={C} r={72} teeth={32} toothDepth={3.5} color={color} opacity={0.12} />
      </g>

      {/* Inner ring 1 — dashed, CW fast */}
      <circle cx={C} cy={C} r={56} fill="none" stroke={color} strokeWidth="0.4" opacity="0.15"
        strokeDasharray="6 8"
      >
        <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="360 100 100" dur="12s" repeatCount="indefinite" />
      </circle>

      {/* Inner ring 2 — dashed, CCW */}
      <circle cx={C} cy={C} r={42} fill="none" stroke={color} strokeWidth="0.4" opacity="0.12"
        strokeDasharray="3 10"
      >
        <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="-360 100 100" dur="8s" repeatCount="indefinite" />
      </circle>

      {/* Inner gear ring */}
      <g>
        <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="360 100 100" dur="20s" repeatCount="indefinite" />
        <GearTeeth cx={C} cy={C} r={32} teeth={16} toothDepth={2.5} color={color} opacity={0.15} />
      </g>

      {/* Orbital particles on middle ring */}
      {[0, 1, 2, 3].map((i) => (
        <circle key={`p-${i}`} cx={C + 56} cy={C} r="1.5" fill={color} opacity="0.5">
          <animateTransform
            attributeName="transform" type="rotate"
            from={`${i * 90} 100 100`} to={`${i * 90 + 360} 100 100`}
            dur={`${5 + i * 0.7}s`} repeatCount="indefinite"
          />
          <animate attributeName="opacity" values="0.3;0.7;0.3" dur={`${2 + i * 0.3}s`} repeatCount="indefinite" />
        </circle>
      ))}

      {/* Core dot */}
      <circle cx={C} cy={C} r="6" fill={color} opacity="0.15">
        <animate attributeName="r" values="5;8;5" dur="2.4s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.1;0.2;0.1" dur="2.4s" repeatCount="indefinite" />
      </circle>
      <circle cx={C} cy={C} r="3" fill={color} opacity="0.6">
        <animate attributeName="r" values="2.5;3.5;2.5" dur="1.8s" repeatCount="indefinite" />
      </circle>

      {/* Tick marks on outer ring */}
      {Array.from({ length: 24 }).map((_, i) => {
        const a = (i / 24) * Math.PI * 2 - Math.PI / 2;
        const r1 = 84;
        const r2 = i % 3 === 0 ? 80 : 82;
        return (
          <line
            key={`tick-${i}`}
            x1={C + Math.cos(a) * r1} y1={C + Math.sin(a) * r1}
            x2={C + Math.cos(a) * r2} y2={C + Math.sin(a) * r2}
            stroke={color}
            strokeWidth={i % 3 === 0 ? "0.8" : "0.4"}
            opacity={i % 3 === 0 ? "0.25" : "0.1"}
          />
        );
      })}
    </svg>
  );
}

interface BuildingVisualizerProps {
  stage: PipelineStage;
  toolEvents: ToolEvent[];
}

export function BuildingVisualizer({ stage, toolEvents }: BuildingVisualizerProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const meta = STAGE_META[stage] || STAGE_META.plan;
  const [elapsed, setElapsed] = useState(0);

  const successCount = useMemo(
    () => toolEvents.filter((t) => t.status === "success").length,
    [toolEvents]
  );

  const total = Math.max(toolEvents.length, 1);
  const progress = successCount / total;

  const lastActiveColor = useMemo(() => {
    const running = [...toolEvents].reverse().find((t) => t.status === "running");
    return running?.color;
  }, [toolEvents]);

  // Elapsed timer
  useEffect(() => {
    const start = Date.now();
    const interval = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(interval);
  }, []);

  // Stagger-in tool lines
  useEffect(() => {
    if (!listRef.current) return;
    const items = listRef.current.querySelectorAll(".tool-line-item");
    if (items.length === 0) return;
    animate(items, {
      opacity: [0, 1],
      translateX: [-12, 0],
      duration: 400,
      ease: "outCubic",
      delay: stagger(60),
    });
  }, [toolEvents.length]);

  const accentColor = lastActiveColor || meta.color;
  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  const timeStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;

  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 px-8">
      <MechanicalCore color={accentColor} progress={progress} />

      {/* Stage label */}
      <div className="text-center">
        <div className="flex items-center justify-center gap-2">
          <p
            className="text-sm font-mono font-semibold tracking-widest uppercase"
            style={{ color: accentColor }}
          >
            {meta.label}
          </p>
          <span className="text-[9px] font-mono text-muted-foreground/30 tabular-nums">
            {timeStr}
          </span>
        </div>
        <p className="mt-0.5 text-[10px] text-muted-foreground/40 font-mono">
          {meta.sub}
        </p>
      </div>

      {/* Tool event log */}
      {toolEvents.length > 0 && (
        <div ref={listRef} className="w-full max-w-xs space-y-0.5">
          {toolEvents.slice(-8).map((t) => (
            <div
              key={t.id}
              className="tool-line-item flex items-center gap-2 rounded-md px-2 py-1 text-[10px] font-mono transition-colors"
              style={{
                background:
                  t.status === "running"
                    ? `${t.color || accentColor}08`
                    : "transparent",
              }}
            >
              {t.status === "running" ? (
                <Loader2
                  size={10}
                  className="animate-spin shrink-0"
                  style={{ color: t.color || accentColor }}
                />
              ) : t.status === "success" ? (
                <CheckCircle
                  size={10}
                  className="shrink-0"
                  style={{ color: t.color ? `${t.color}99` : "#34d399b3" }}
                />
              ) : (
                <XCircle size={10} className="text-red-400/70 shrink-0" />
              )}
              <span
                className="font-medium"
                style={{
                  color: t.status === "error" ? "#f87171" : t.color ? `${t.color}cc` : undefined,
                }}
              >
                {t.label}
              </span>
              {t.detail && t.status !== "running" && (
                <span className="truncate text-muted-foreground/25 ml-auto text-[9px]">
                  {t.detail.slice(0, 40)}
                </span>
              )}
            </div>
          ))}

          {/* Progress bar + counter */}
          <div className="flex items-center gap-2 pt-1.5">
            <div className="flex-1 h-[2px] rounded-full bg-white/5 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500 ease-out"
                style={{
                  width: `${progress * 100}%`,
                  background: `linear-gradient(90deg, ${accentColor}66, ${accentColor})`,
                }}
              />
            </div>
            <span className="text-[9px] font-mono text-muted-foreground/30 tabular-nums shrink-0">
              {successCount}/{toolEvents.length}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
