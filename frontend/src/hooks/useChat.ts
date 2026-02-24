"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { wsManager, type WSMessage } from "@/lib/websocket";
import type { ChatMessage } from "@/types";

export type ToolEvent = {
  id: string;
  tool: string;
  label: string;
  status: "running" | "success" | "error" | "pending";
  detail?: string;
  color?: string;
  timestamp: number;
};

export type PipelineStage = "plan" | "write" | "validate" | "render" | "done" | "error";

const TOOL_DISPLAY: Record<string, { label: string; color: string }> = {
  cad_generate: { label: "generating 3D model", color: "#f59e0b" },
  urdf_generate: { label: "building robot model", color: "#8b5cf6" },
  file_write: { label: "writing files", color: "#3b82f6" },
  file_read: { label: "reading files", color: "#64748b" },
  file_edit: { label: "editing code", color: "#06b6d4" },
  shell: { label: "running command", color: "#10b981" },
  eval_env: { label: "validating environment", color: "#f97316" },
  eval_tool: { label: "running validation", color: "#f97316" },
  eval_agent: { label: "diagnosing errors", color: "#ef4444" },
  code_search: { label: "searching code", color: "#64748b" },
  dir_list: { label: "browsing files", color: "#64748b" },
  doc_lookup: { label: "looking up docs", color: "#94a3b8" },
};

const HIDDEN_TOOLS = new Set(["doc_lookup"]);

function getToolDisplay(toolName: string, args?: Record<string, unknown>): { label: string; color: string } {
  const base = TOOL_DISPLAY[toolName];

  if (toolName === "cad_generate" && args) {
    const desc = String(args.description || args.name || "").toLowerCase();
    if (desc.includes("tree")) return { label: "generating trees", color: "#22c55e" };
    if (desc.includes("rock")) return { label: "sculpting rocks", color: "#78716c" };
    if (desc.includes("log")) return { label: "crafting logs", color: "#a16207" };
    if (desc.includes("desk")) return { label: "building desk", color: "#d97706" };
    if (desc.includes("robot") || desc.includes("arm")) return { label: "generating robot", color: "#8b5cf6" };
    return { label: `generating ${desc.slice(0, 30) || "asset"}`, color: "#f59e0b" };
  }

  if (toolName === "urdf_generate" && args) {
    const desc = String(args.name || args.description || "").toLowerCase();
    if (desc) return { label: `building ${desc.slice(0, 30)}`, color: "#8b5cf6" };
  }

  if (toolName === "file_write" && args) {
    const path = String(args.path || "");
    if (path.includes("env.py")) return { label: "writing environment", color: "#3b82f6" };
    if (path.includes("config.py")) return { label: "writing config", color: "#06b6d4" };
    if (path.includes("rewards.py")) return { label: "designing rewards", color: "#eab308" };
    if (path.includes("train.py")) return { label: "setting up training", color: "#10b981" };
    if (path.includes(".scad")) return { label: "generating 3D model", color: "#f59e0b" };
  }

  if (toolName === "file_edit" && args) {
    const path = String(args.path || "");
    if (path.includes("env.py")) return { label: "patching environment", color: "#06b6d4" };
    if (path.includes("config")) return { label: "updating config", color: "#06b6d4" };
    if (path.includes("rewards")) return { label: "tuning rewards", color: "#eab308" };
  }

  return base || { label: toolName, color: "#67e8f9" };
}

let _msgIdSeq = 0;
function nextId(prefix: string): string {
  return `${prefix}-${++_msgIdSeq}-${Date.now()}`;
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [connected, setConnected] = useState(false);
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
  const [currentEnvId, setCurrentEnvId] = useState<string>("");
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>("plan");
  const [validationPassed, setValidationPassed] = useState(false);
  const toolIdCounter = useRef(0);

  useEffect(() => {
    wsManager.connect();

    const unsub = wsManager.subscribe((msg: WSMessage) => {
      switch (msg.type) {
        case "status":
          if (msg.status === "connected") setConnected(true);
          else if (msg.status === "disconnected") setConnected(false);
          else if (msg.status === "processing") {
            setIsProcessing(true);
            setPipelineStage("plan");
            setValidationPassed(false);
          }
          break;

        case "agent_event": {
          const evt = msg.event;
          if (evt.type === "tool_start") {
            const toolName = String(evt.data.tool || "unknown");

            if (HIDDEN_TOOLS.has(toolName)) break;

            const display = getToolDisplay(toolName, evt.data.args as Record<string, unknown>);
            const id = `t-${++toolIdCounter.current}`;
            setToolEvents((prev) => [
              ...prev,
              {
                id,
                tool: toolName,
                label: display.label,
                color: display.color,
                status: "running",
                timestamp: Date.now(),
              },
            ]);
            if (toolName === "file_write") setPipelineStage("write");
            if (toolName === "eval_env" || toolName === "eval_tool") setPipelineStage("validate");
          } else if (evt.type === "tool_end") {
            const toolName = String(evt.data.tool || "");
            if (HIDDEN_TOOLS.has(toolName)) break;

            const resultStr = String(evt.data.result || "");
            const isError = resultStr.startsWith("Tool error:") || resultStr.startsWith("ERROR:");
            setToolEvents((prev) => {
              const copy = [...prev];
              const last = copy.findLastIndex(
                (t) => t.tool === toolName && t.status === "running"
              );
              if (last >= 0) {
                copy[last] = {
                  ...copy[last],
                  status: isError ? "error" : "success",
                  detail: resultStr.slice(0, 120),
                };
              }
              return copy;
            });
          } else if (evt.type === "error") {
            setPipelineStage("error");
            setToolEvents((prev) => {
              const copy = [...prev];
              const lastRunning = copy.findLastIndex((t) => t.status === "running");
              if (lastRunning >= 0) {
                copy[lastRunning] = {
                  ...copy[lastRunning],
                  status: "error",
                  detail: String(evt.data.message || "Validation failed"),
                };
              }
              return copy;
            });
            setMessages((prev) => [
              ...prev,
              {
                id: nextId("e"),
                role: "assistant",
                content: String(evt.data.message || "Validation error"),
                error: String(evt.data.details || ""),
                timestamp: Date.now(),
              },
            ]);
          } else if (evt.type === "validation") {
            setPipelineStage("validate");
            const id = `t-${++toolIdCounter.current}`;
            setToolEvents((prev) => [
              ...prev,
              {
                id,
                tool: "eval_tool",
                label: `validating ${evt.data.env_id || "env"}`,
                color: "#f97316",
                status: "running",
                detail: `Validating ${evt.data.env_id}`,
                timestamp: Date.now(),
              },
            ]);
          } else if (evt.type === "validation_passed") {
            setValidationPassed(true);
            setPipelineStage("done");
            if (evt.data.env_id) setCurrentEnvId(String(evt.data.env_id));
            setToolEvents((prev) => {
              const copy = [...prev];
              const last = copy.findLastIndex(
                (t) => t.tool === "eval_tool" && t.status === "running"
              );
              if (last >= 0) {
                copy[last] = { ...copy[last], status: "success", detail: "All stages passed" };
              }
              return copy;
            });
          }
          break;
        }

        case "response":
          setIsProcessing(false);
          setMessages((prev) => [
            ...prev,
            { id: nextId("a"), role: "assistant", content: msg.content, timestamp: Date.now() },
          ]);
          if (msg.env_id) setCurrentEnvId(msg.env_id);
          setToolEvents([]);
          break;

        case "error":
          setIsProcessing(false);
          setPipelineStage("error");
          setMessages((prev) => [
            ...prev,
            {
              id: nextId("e"),
              role: "assistant",
              content: `Error: ${msg.message}`,
              error: msg.traceback,
              timestamp: Date.now(),
            },
          ]);
          setToolEvents([]);
          break;

        case "session_cleared":
          break;
      }
    });

    return () => { unsub(); };
  }, []);

  const sendMessage = useCallback(
    (content: string, images?: string[]) => {
      const userMsg: ChatMessage = {
        id: nextId("u"),
        role: "user",
        content,
        images: images?.length ? images : undefined,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsProcessing(true);
      setToolEvents([]);
      setPipelineStage("plan");
      setValidationPassed(false);

      wsManager.send({
        type: "message",
        content,
        images: images || [],
        session_id: "default",
      });
    },
    []
  );

  const clearSession = useCallback(() => {
    wsManager.send({ type: "clear_session", session_id: "default" });
    setMessages([]);
    setToolEvents([]);
    setIsProcessing(false);
    setPipelineStage("plan");
    setValidationPassed(false);
    setCurrentEnvId("");
  }, []);

  const retryLast = useCallback(() => {
    setMessages((prev) => {
      const lastUser = [...prev].reverse().find((m) => m.role === "user");
      if (lastUser) {
        setIsProcessing(true);
        setToolEvents([]);
        setPipelineStage("plan");
        setValidationPassed(false);
        wsManager.send({
          type: "message",
          content: lastUser.content,
          images: lastUser.images || [],
          session_id: "default",
        });
      }
      return prev;
    });
  }, []);

  return {
    messages,
    isProcessing,
    connected,
    toolEvents,
    currentEnvId,
    setCurrentEnvId,
    pipelineStage,
    validationPassed,
    sendMessage,
    clearSession,
    retryLast,
  };
}
