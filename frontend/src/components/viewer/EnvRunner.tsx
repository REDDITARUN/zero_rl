"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Play, Square, RotateCcw, Loader2, Keyboard, Move3d } from "lucide-react";

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL?.replace("/ws/chat", "") || "ws://localhost:8000";

type ActionSpaceDesc =
  | { type: "discrete"; n: number }
  | { type: "box"; shape: number[]; low: number[]; high: number[] }
  | { type: string };

type FrameData = {
  type?: string;
  step: number;
  frame: string;
  reward: number;
  done: boolean;
  actions: number[];
  info: Record<string, number | string>;
  action_space?: ActionSpaceDesc;
  error?: string;
};

const DISCRETE_KEY_MAP: Record<string, number> = {
  ArrowRight: 0,
  ArrowUp: 1,
  ArrowLeft: 2,
  ArrowDown: 3,
  d: 0,
  w: 1,
  a: 2,
  s: 3,
};

interface EnvRunnerProps {
  envId: string;
  autoStart?: boolean;
}

export function EnvRunner({ envId, autoStart = true }: EnvRunnerProps) {
  const [status, setStatus] = useState<"idle" | "connecting" | "running" | "done" | "error">("idle");
  const [frame, setFrame] = useState<string>("");
  const [step, setStep] = useState(0);
  const [reward, setReward] = useState(0);
  const [totalReward, setTotalReward] = useState(0);
  const [actions, setActions] = useState<number[]>([]);
  const [info, setInfo] = useState<Record<string, number | string>>({});
  const [error, setError] = useState("");
  const [fps, setFps] = useState(0);
  const [actionSpace, setActionSpace] = useState<ActionSpaceDesc | null>(null);
  const [manualMode, setManualMode] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const frameCountRef = useRef(0);
  const fpsTimerRef = useRef<ReturnType<typeof setInterval>>();
  const containerRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);
  const lastMouseRef = useRef({ x: 0, y: 0 });

  const sendWs = useCallback((data: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data));
    }
  }, []);

  const sendAction = useCallback((action: number | number[]) => {
    sendWs({ action });
  }, [sendWs]);

  const sendCamera = useCallback((camera: Record<string, unknown>) => {
    sendWs({ camera });
  }, [sendWs]);

  /** Normalise pixel delta to fraction of viewport (0-1 range). */
  const normDelta = useCallback((pxX: number, pxY: number): [number, number] => {
    const el = frameRef.current;
    const w = el?.clientWidth || 600;
    const h = el?.clientHeight || 400;
    return [pxX / w, pxY / h];
  }, []);

  const sendReset = useCallback(() => {
    sendWs({ cmd: "reset" });
    setTotalReward(0);
  }, [sendWs]);

  const start = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    setStatus("connecting");
    setStep(0);
    setTotalReward(0);
    setFrame("");
    setError("");
    setActionSpace(null);
    frameCountRef.current = 0;

    const ws = new WebSocket(`${WS_BASE}/ws/env/${envId}/frames`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("running");
      fpsTimerRef.current = setInterval(() => {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
      }, 1000);
    };

    ws.onmessage = (event) => {
      try {
        const data: FrameData = JSON.parse(event.data);
        if (data.error) {
          setError(data.error);
          setStatus("error");
          return;
        }
        if (data.action_space) {
          setActionSpace(data.action_space);
        }
        setStep(data.step);
        setReward(data.reward ?? 0);
        setTotalReward((prev) => prev + (data.reward ?? 0));
        setActions(Array.isArray(data.actions) ? data.actions : typeof data.actions === "number" ? [data.actions] : []);
        setInfo(data.info || {});
        if (data.frame) {
          setFrame(data.frame);
          frameCountRef.current++;
        }
        if (data.done) {
          setTimeout(() => setTotalReward(0), 1500);
        }
      } catch {
        /* ignore malformed */
      }
    };

    ws.onclose = () => {
      setStatus((prev) => (prev === "running" ? "done" : prev));
      if (fpsTimerRef.current) clearInterval(fpsTimerRef.current);
    };

    ws.onerror = () => {
      setStatus("error");
      setError("WebSocket connection failed");
    };
  }, [envId]);

  const stop = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setStatus("done");
    if (fpsTimerRef.current) clearInterval(fpsTimerRef.current);
  }, []);

  // Always-on turntable camera: left-drag = orbit, right/middle-drag = pan, scroll = zoom
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (status !== "running") return;
    isDraggingRef.current = true;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
    if (e.button === 0 || e.button === 2) e.preventDefault();
  }, [status]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDraggingRef.current || status !== "running") return;
    const dx = e.clientX - lastMouseRef.current.x;
    const dy = e.clientY - lastMouseRef.current.y;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
    if (dx === 0 && dy === 0) return;

    const [nx, ny] = normDelta(dx, dy);

    if (e.buttons === 1) {
      // Left-drag: turntable orbit
      sendCamera({ orbit: [nx, ny] });
    } else if (e.buttons === 2 || e.buttons === 4) {
      // Right-drag or middle-drag: pan
      sendCamera({ pan: [nx, ny] });
    }
  }, [status, normDelta, sendCamera]);

  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = false;
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (status !== "running") return;
    e.preventDefault();
    const dz = e.deltaY > 0 ? -1 : 1;
    sendCamera({ zoom: dz });
  }, [status, sendCamera]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
  }, []);

  // Keyboard handler: camera shortcuts always active, action keys only in manual mode
  useEffect(() => {
    if (status !== "running") return;

    const handleKey = (e: KeyboardEvent) => {
      // Camera keyboard shortcuts (always active, no toggle needed)
      const PAN = 0.04;
      const key = e.key.toLowerCase();
      if (!manualMode) {
        switch (key) {
          case "w": sendCamera({ pan: [0, PAN] }); e.preventDefault(); return;
          case "s": sendCamera({ pan: [0, -PAN] }); e.preventDefault(); return;
          case "a": sendCamera({ pan: [-PAN, 0] }); e.preventDefault(); return;
          case "d": sendCamera({ pan: [PAN, 0] }); e.preventDefault(); return;
          case "q": sendCamera({ zoom: 1 }); e.preventDefault(); return;
          case "e": sendCamera({ zoom: -1 }); e.preventDefault(); return;
          case "r": sendCamera({ reset: true }); e.preventDefault(); return;
        }
      }

      // Manual action mode
      if (!manualMode || !actionSpace) return;

      if (actionSpace.type === "discrete") {
        const mapped = DISCRETE_KEY_MAP[e.key];
        if (mapped !== undefined && mapped < (actionSpace as { n: number }).n) {
          e.preventDefault();
          sendAction(mapped);
        }
      } else if (actionSpace.type === "box") {
        const shape = (actionSpace as { shape: number[] }).shape;
        const dim = shape[0] || 1;
        const act = new Array(dim).fill(0);

        if (e.key === "w" || e.key === "ArrowUp") { act[0] = 0.5; e.preventDefault(); }
        else if (e.key === "s" || e.key === "ArrowDown") { act[0] = -0.5; e.preventDefault(); }
        else if (e.key === "a" || e.key === "ArrowLeft") { if (dim > 1) act[1] = -0.5; e.preventDefault(); }
        else if (e.key === "d" || e.key === "ArrowRight") { if (dim > 1) act[1] = 0.5; e.preventDefault(); }
        else return;

        sendAction(act);
      }

      if (key === "r") {
        sendReset();
      }
    };

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [status, manualMode, actionSpace, sendAction, sendCamera, sendReset]);

  useEffect(() => {
    if (autoStart) {
      start();
    }
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
      if (fpsTimerRef.current) clearInterval(fpsTimerRef.current);
    };
  }, [envId]); // eslint-disable-line react-hooks/exhaustive-deps

  const isDiscrete = actionSpace?.type === "discrete";
  const discreteN = isDiscrete ? (actionSpace as { n: number }).n : 0;

  return (
    <div ref={containerRef} className="flex h-full flex-col gap-3" tabIndex={-1}>
      {/* Controls bar */}
      <div className="flex items-center gap-2 shrink-0 flex-wrap">
        <div className="flex items-center gap-1">
          {status === "running" ? (
            <button
              onClick={stop}
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1.5 text-[10px] font-mono text-muted-foreground hover:text-foreground transition-colors"
            >
              <Square size={10} /> stop
            </button>
          ) : (
            <button
              onClick={start}
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1.5 text-[10px] font-mono text-muted-foreground hover:text-foreground transition-colors"
            >
              {status === "connecting" ? (
                <Loader2 size={10} className="animate-spin" />
              ) : (
                <Play size={10} />
              )}
              {status === "connecting" ? "loading..." : "run"}
            </button>
          )}
          <button
            onClick={sendReset}
            disabled={status !== "running"}
            className="flex items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1.5 text-[10px] font-mono text-muted-foreground hover:text-foreground transition-colors disabled:opacity-30"
          >
            <RotateCcw size={10} /> reset
          </button>
          <button
            onClick={() => setManualMode((p) => !p)}
            className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-[10px] font-mono transition-colors ${
              manualMode
                ? "border-primary/40 bg-primary/10 text-primary"
                : "border-border bg-card text-muted-foreground hover:text-foreground"
            }`}
          >
            <Keyboard size={10} /> {manualMode ? "manual" : "auto"}
          </button>
        </div>

        <div className="ml-auto flex items-center gap-3 text-[10px] font-mono text-muted-foreground/50">
          <span>step {step}</span>
          <span>{fps} fps</span>
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              status === "running"
                ? "bg-emerald-400 animate-pulse"
                : status === "error"
                ? "bg-red-400"
                : "bg-muted-foreground/30"
            }`}
          />
        </div>
      </div>

      {/* Manual mode hint */}
      {manualMode && status === "running" && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          className="shrink-0 rounded-md border border-primary/20 bg-primary/5 px-3 py-2"
        >
          <p className="text-[10px] font-mono text-primary/70">
            {isDiscrete
              ? `keyboard: W/A/S/D or arrows (${discreteN} actions) · R = reset`
              : `keyboard: W/S = axis 0 · A/D = axis 1 · R = reset`}
          </p>
        </motion.div>
      )}

      {/* Camera hint — always visible while running */}
      {status === "running" && !manualMode && (
        <div className="shrink-0 rounded-md border border-border/40 bg-card/30 px-3 py-1.5">
          <p className="text-[9px] font-mono text-muted-foreground/40">
            left-drag orbit · right-drag pan · scroll zoom · W/A/S/D pan · Q/E zoom · R reset
          </p>
        </div>
      )}

      {/* Discrete action buttons */}
      {manualMode && status === "running" && isDiscrete && (
        <div className="shrink-0 flex items-center gap-1 flex-wrap">
          {Array.from({ length: discreteN }, (_, i) => (
            <button
              key={i}
              onClick={() => sendAction(i)}
              className="rounded border border-border bg-card px-2 py-1 text-[10px] font-mono text-muted-foreground hover:bg-primary/10 hover:text-primary hover:border-primary/30 transition-colors"
            >
              {i}
            </button>
          ))}
        </div>
      )}

      {/* Frame display with interactive camera overlay */}
      <div
        ref={frameRef}
        className="relative flex-1 min-h-[200px] rounded-xl border border-border overflow-hidden bg-black/50 cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        onContextMenu={handleContextMenu}
      >
        {frame ? (
          <img
            src={`data:image/png;base64,${frame}`}
            alt="Environment render"
            className="absolute inset-0 h-full w-full object-contain select-none pointer-events-none"
            draggable={false}
          />
        ) : status === "connecting" ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <Loader2 size={20} className="mx-auto mb-2 animate-spin text-primary/40" />
              <p className="text-[10px] text-muted-foreground/40 font-mono">
                initializing environment...
              </p>
            </div>
          </div>
        ) : error ? (
          <div className="flex h-full items-center justify-center p-4">
            <pre className="text-[10px] text-red-400/70 font-mono whitespace-pre-wrap max-w-full overflow-auto">
              {error}
            </pre>
          </div>
        ) : (
          <div className="flex h-full items-center justify-center">
            <p className="text-[10px] text-muted-foreground/30 font-mono">
              {status === "running" ? "waiting for frames..." : "press run to start"}
            </p>
          </div>
        )}

        {/* Camera icon — subtle top-right indicator */}
        {status === "running" && (
          <div className="absolute top-2 right-2 flex items-center gap-1 rounded-md bg-black/40 px-1.5 py-0.5 backdrop-blur-sm pointer-events-none">
            <Move3d size={9} className="text-white/30" />
          </div>
        )}
      </div>

      {/* Reward & action stats */}
      <div className="shrink-0 grid grid-cols-2 gap-2">
        <div className="rounded-xl border border-border bg-card/50 p-2">
          <p className="text-[9px] text-muted-foreground/40 font-mono uppercase tracking-wider mb-1">
            reward
          </p>
          <p className={`text-[13px] font-mono font-medium ${reward >= 0 ? "text-emerald-400" : "text-red-400"}`}>
            {reward.toFixed(4)}
          </p>
          <p className="text-[9px] font-mono text-muted-foreground/30 mt-0.5">
            total: {totalReward.toFixed(2)}
          </p>
        </div>
        <div className="rounded-xl border border-border bg-card/50 p-2">
          <p className="text-[9px] text-muted-foreground/40 font-mono uppercase tracking-wider mb-1">
            actions
          </p>
          <p className="text-[11px] font-mono text-foreground/70 truncate">
            [{Array.isArray(actions) ? actions.map((a) => (typeof a === "number" ? a.toFixed(2) : String(a))).join(", ") : String(actions)}]
          </p>
          {actionSpace && (
            <p className="text-[9px] font-mono text-muted-foreground/30 mt-0.5">
              {actionSpace.type === "discrete" ? `discrete(${(actionSpace as {n:number}).n})` : `box${JSON.stringify((actionSpace as {shape:number[]}).shape)}`}
            </p>
          )}
        </div>
      </div>

      {/* Info breakdown */}
      {Object.keys(info).length > 0 && (
        <div className="shrink-0 rounded-xl border border-border bg-card/50 p-2">
          <p className="text-[9px] text-muted-foreground/40 font-mono uppercase tracking-wider mb-1.5">
            info
          </p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
            {Object.entries(info).map(([k, v]) => (
              <div key={k} className="flex justify-between text-[10px] font-mono">
                <span className="text-muted-foreground/50">{k}</span>
                <span className="text-foreground/60">
                  {typeof v === "number" ? v.toFixed(3) : String(v)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
