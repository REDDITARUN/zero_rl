"use client";

import { useState, useCallback, useRef, useEffect, useMemo, type DragEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputActions,
  PromptInputAction,
} from "@/components/prompt-kit/prompt-input";
import {
  Paperclip,
  Send,
  Settings,
  FolderOpen,
  Trash2,
  ChevronRight,
  Check,
  X,
  Terminal,
  CheckCircle,
  Loader2,
  RotateCcw,
  Edit3,
  Code,
  Eye,
  Box,
  Grip,
  Cpu,
  ChevronDown,
} from "lucide-react";
import { useChat, type ToolEvent, type PipelineStage } from "@/hooks/useChat";
import { useEnvs } from "@/hooks/useEnvs";
import { useAssets, type AssetMeta, type AssetDetails } from "@/hooks/useAssets";
import { EnvRunner } from "@/components/viewer/EnvRunner";
import { AssetViewer } from "@/components/viewer/AssetViewer";
import { AssetParameterPanel } from "@/components/asset/AssetParameterPanel";
import { BuildingVisualizer } from "@/components/progress/BuildingVisualizer";
import type { ChatMessage, EnvDetails } from "@/types";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Right Panel State Machine ──────────────────────────────────────────────

type RightPanelView =
  | { kind: "empty" }
  | { kind: "building"; stage: PipelineStage; toolEvents: ToolEvent[] }
  | { kind: "env-viewer"; envId: string }
  | { kind: "env-code"; env: EnvDetails }
  | { kind: "asset-viewer"; asset: AssetDetails; modelFile: string }
  | { kind: "asset-code"; asset: AssetDetails };

// ── Main Page ──────────────────────────────────────────────────────────────

export default function Home() {
  const {
    messages, isProcessing, connected, toolEvents,
    currentEnvId, setCurrentEnvId, pipelineStage, validationPassed,
    sendMessage, clearSession, retryLast,
  } = useChat();
  const { envs, refresh: refreshEnvs, getEnv, deleteEnv } = useEnvs();
  const { assets, refresh: refreshAssets, getAsset } = useAssets();
  const [input, setInput] = useState("");
  const [showGallery, setShowGallery] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [selectedEnv, setSelectedEnv] = useState<EnvDetails | null>(null);
  const [selectedAsset, setSelectedAsset] = useState<AssetDetails | null>(null);
  const [sideTab, setSideTab] = useState<"viewer" | "code">("viewer");
  const [envsOpen, setEnvsOpen] = useState(true);
  const [assetsOpen, setAssetsOpen] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const [attachedAssets, setAttachedAssets] = useState<string[]>([]);
  const [attachedImages, setAttachedImages] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, toolEvents]);

  // On new env created, select it
  useEffect(() => {
    if (currentEnvId) {
      setSelectedAsset(null);
      getEnv(currentEnvId).then((d) => {
        if (d) { setSelectedEnv(d); refreshEnvs(); refreshAssets(); setSideTab("viewer"); }
      });
    }
  }, [currentEnvId, getEnv, refreshEnvs, refreshAssets]);

  // Derive right panel view
  const rightPanel = useMemo((): RightPanelView => {
    if (isProcessing) {
      return { kind: "building", stage: pipelineStage, toolEvents };
    }
    if (selectedAsset) {
      const modelFile = selectedAsset.binary_files.find(
        (f) => f.endsWith(".stl") || f.endsWith(".urdf")
      );
      if (sideTab === "viewer" && modelFile) {
        return { kind: "asset-viewer", asset: selectedAsset, modelFile };
      }
      return { kind: "asset-code", asset: selectedAsset };
    }
    if ((validationPassed || selectedEnv) && currentEnvId) {
      if (sideTab === "code" && selectedEnv) {
        return { kind: "env-code", env: selectedEnv };
      }
      return { kind: "env-viewer", envId: currentEnvId };
    }
    return { kind: "empty" };
  }, [isProcessing, pipelineStage, toolEvents, selectedAsset, selectedEnv, validationPassed, currentEnvId, sideTab]);

  const handleSubmit = useCallback(() => {
    if ((!input.trim() && attachedImages.length === 0) || isProcessing) return;
    let msg = input || (attachedImages.length > 0 ? "Analyze this image" : "");
    if (attachedAssets.length > 0) {
      msg += `\n\n[Attached assets: ${attachedAssets.join(", ")}]`;
    }
    if (selectedEnv && !msg.toLowerCase().includes("create")) {
      msg += `\n\n[Active environment: envs/${selectedEnv.id}/ — modify this env rather than creating a new one]`;
    }
    sendMessage(msg, attachedImages.length > 0 ? attachedImages : undefined);
    setInput("");
    setAttachedAssets([]);
    setAttachedImages([]);
  }, [input, isProcessing, sendMessage, attachedAssets, attachedImages, selectedEnv]);

  const handleSampleClick = useCallback(
    (sample: string) => { sendMessage(sample); },
    [sendMessage]
  );

  const handleDeleteEnv = useCallback(async (envId: string) => {
    await deleteEnv(envId);
    if (selectedEnv?.id === envId) setSelectedEnv(null);
  }, [deleteEnv, selectedEnv]);

  // Drag-and-drop for assets and images
  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);
  const handleDragLeave = useCallback(() => setDragOver(false), []);
  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);

    // Asset drag from sidebar
    const assetId = e.dataTransfer.getData("application/x-asset-id");
    const assetFile = e.dataTransfer.getData("application/x-asset-file");
    if (assetId) {
      const ref = assetFile ? `assets/${assetId}/${assetFile}` : `assets/${assetId}`;
      setAttachedAssets((prev) => prev.includes(ref) ? prev : [...prev, ref]);
      return;
    }

    // Image file drop
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      Array.from(files).forEach((file) => {
        if (!file.type.startsWith("image/")) return;
        const reader = new FileReader();
        reader.onload = () => {
          setAttachedImages((prev) => [...prev, reader.result as string]);
        };
        reader.readAsDataURL(file);
      });
    }
  }, []);

  const hasContent = rightPanel.kind !== "empty";
  const showTabs = !isProcessing && (selectedEnv || selectedAsset);

  return (
    <div className="h-screen w-screen overflow-hidden bg-black/80 p-2 font-mono">
      <div className="flex h-full w-full overflow-hidden rounded-2xl border border-border/50 bg-background shadow-2xl">
      {/* ── Gallery Sidebar ───────────────────────────────────────── */}
      <AnimatePresence>
        {showGallery && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 220, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="relative z-10 flex flex-col overflow-hidden border-r border-border bg-background"
          >
            <div className="flex items-center justify-between px-3 py-2.5 text-[10px] uppercase tracking-widest text-muted-foreground">
              <span>browser</span>
              <button onClick={() => setShowGallery(false)} className="rounded p-0.5 hover:bg-white/5">
                <ChevronRight size={10} className="rotate-180" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto px-1.5 pb-4">
              {/* Envs section */}
              <button
                onClick={() => setEnvsOpen((p) => !p)}
                className="flex w-full items-center gap-1.5 px-2 py-1.5 text-[10px] uppercase tracking-wider text-emerald-400/50 hover:text-emerald-400/80 transition-colors"
              >
                <ChevronDown size={9} className={`transition-transform ${envsOpen ? "" : "-rotate-90"}`} />
                <Terminal size={9} />
                <span>environments</span>
                <span className="ml-auto text-emerald-400/25">{envs.length}</span>
              </button>
              {envsOpen && (
                <div className="mb-2">
                  {envs.map((env) => (
                    <div
                      key={env.id}
                      role="button"
                      tabIndex={0}
                      onClick={async () => {
                        setSelectedAsset(null);
                        const d = await getEnv(env.id);
                        if (d) {
                          setSelectedEnv(d);
                          setCurrentEnvId(env.id);
                          setSideTab("viewer");
                        }
                      }}
                      onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLElement).click(); }}
                      className={`group flex w-full cursor-pointer items-center gap-1.5 rounded px-2 py-1.5 text-left text-[11px] transition-colors hover:bg-white/5 ${
                        selectedEnv?.id === env.id && !selectedAsset ? "bg-emerald-500/10 text-emerald-400" : "text-muted-foreground"
                      }`}
                    >
                      <Cpu size={9} className="shrink-0 text-emerald-400/40" />
                      <span className="flex-1 truncate">{env.name || env.id}</span>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDeleteEnv(env.id); }}
                        className="rounded p-0.5 opacity-0 group-hover:opacity-50 hover:!opacity-100 hover:text-destructive"
                      >
                        <Trash2 size={8} />
                      </button>
                    </div>
                  ))}
                  {envs.length === 0 && (
                    <p className="px-2 py-3 text-center text-[10px] text-muted-foreground/30">no envs yet</p>
                  )}
                </div>
              )}

              {/* Assets section */}
              <button
                onClick={() => setAssetsOpen((p) => !p)}
                className="flex w-full items-center gap-1.5 px-2 py-1.5 text-[10px] uppercase tracking-wider text-violet-400/50 hover:text-violet-400/80 transition-colors"
              >
                <ChevronDown size={9} className={`transition-transform ${assetsOpen ? "" : "-rotate-90"}`} />
                <Box size={9} />
                <span>assets</span>
                <span className="ml-auto text-violet-400/25">{assets.length}</span>
              </button>
              {assetsOpen && (
                <div className="mb-2">
                  {assets.map((asset) => (
                    <AssetSidebarItem
                      key={asset.id}
                      asset={asset}
                      selected={selectedAsset?.id === asset.id}
                      onSelect={async () => {
                        setSelectedEnv(null);
                        const d = await getAsset(asset.id);
                        if (d) {
                          setSelectedAsset(d);
                          setSideTab("viewer");
                        }
                      }}
                    />
                  ))}
                  {assets.length === 0 && (
                    <p className="px-2 py-3 text-center text-[10px] text-muted-foreground/30">no assets yet</p>
                  )}
                </div>
              )}
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* ── Chat Panel (left) ─────────────────────────────────────── */}
      <div className="flex w-[440px] min-w-[360px] flex-col border-r border-border">
        <header className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2">
          <button
            onClick={() => setShowGallery((p) => !p)}
            className="rounded p-1 text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground"
          >
            <FolderOpen size={13} />
          </button>
          <div className="flex items-center">
            <span className="text-[10px] font-medium font-mono tracking-tighter leading-none">
              <span style={{ color: "#15F5BA" }}>0</span><span className="text-white/90">RL</span>
            </span>
          </div>
          <div className="flex-1" />
          <ConnectionBadge connected={connected} />
          <button
            onClick={() => setShowSettings((p) => !p)}
            className="rounded p-1 text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground"
          >
            <Settings size={12} />
          </button>
        </header>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto scroll-smooth">
          {messages.length === 0 && !isProcessing ? (
            <EmptyState onSampleClick={handleSampleClick} />
          ) : (
            <div className="flex flex-col">
              {messages.map((msg) => (
                <TerminalMessage
                  key={msg.id}
                  message={msg}
                  onAcceptPlan={() => sendMessage("Go ahead, execute the plan.")}
                  onModifyPlan={(mod) => sendMessage(`Modify the plan: ${mod}`)}
                  onRetry={retryLast}
                  onClearAndRetry={clearSession}
                />
              ))}
              <AnimatePresence>
                {isProcessing && toolEvents.length > 0 && (
                  <div className="px-4 py-1">
                    {toolEvents.slice(-6).map((t) => (
                      <ToolLine key={t.id} event={t} />
                    ))}
                  </div>
                )}
              </AnimatePresence>
              {isProcessing && toolEvents.length === 0 && (
                <div className="px-4 py-2">
                  <PulseLoader />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Quick actions + Input */}
        <div className="shrink-0 border-t border-border">
          {/* Active env context badge */}
          {selectedEnv && !isProcessing && (
            <div className="flex items-center gap-1.5 px-3 pt-2">
              <span className="inline-flex items-center gap-1 rounded-md border border-emerald-500/20 bg-emerald-500/5 px-2 py-0.5 text-[9px] font-mono text-emerald-400/70">
                <Cpu size={8} />
                editing: {selectedEnv.id}
                <button
                  onClick={() => { setSelectedEnv(null); setCurrentEnvId(""); }}
                  className="ml-1 hover:text-destructive"
                >
                  <X size={7} />
                </button>
              </span>
            </div>
          )}

          {/* Attached image previews */}
          {attachedImages.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 px-3 pt-2">
              {attachedImages.map((img, i) => (
                <div key={i} className="relative group">
                  <img
                    src={img}
                    alt={`Attached ${i + 1}`}
                    className="h-14 w-14 rounded-md border border-border object-cover"
                  />
                  <button
                    onClick={() => setAttachedImages((prev) => prev.filter((_, j) => j !== i))}
                    className="absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-destructive text-destructive-foreground opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X size={8} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Attached assets chips */}
          {attachedAssets.length > 0 && (
            <div className="flex flex-wrap items-center gap-1 px-3 pt-2">
              {attachedAssets.map((ref) => (
                <span
                  key={ref}
                  className="inline-flex items-center gap-1 rounded-md border border-primary/20 bg-primary/5 px-2 py-0.5 text-[9px] font-mono text-primary/70"
                >
                  <Box size={8} />
                  {ref.split("/").pop()}
                  <button
                    onClick={() => setAttachedAssets((p) => p.filter((a) => a !== ref))}
                    className="ml-0.5 hover:text-destructive"
                  >
                    <X size={7} />
                  </button>
                </span>
              ))}
            </div>
          )}

          <div
            className={`p-3 transition-colors ${dragOver ? "bg-primary/5 ring-1 ring-inset ring-primary/20" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(e) => {
                const files = e.target.files;
                if (!files) return;
                Array.from(files).forEach((file) => {
                  const reader = new FileReader();
                  reader.onload = () => {
                    setAttachedImages((prev) => [...prev, reader.result as string]);
                  };
                  reader.readAsDataURL(file);
                });
                e.target.value = "";
              }}
            />
            <PromptInput
              value={input}
              onValueChange={setInput}
              onSubmit={handleSubmit}
              isLoading={isProcessing}
              onPaste={(e) => {
                const items = e.clipboardData?.items;
                if (!items) return;
                for (const item of Array.from(items)) {
                  if (item.type.startsWith("image/")) {
                    e.preventDefault();
                    const blob = item.getAsFile();
                    if (!blob) continue;
                    const reader = new FileReader();
                    reader.onload = () => {
                      setAttachedImages((prev) => [...prev, reader.result as string]);
                    };
                    reader.readAsDataURL(blob);
                  }
                }
              }}
            >
              <PromptInputTextarea placeholder="describe an rl environment..." />
              <PromptInputActions>
                <PromptInputAction tooltip="Attach image" onClick={() => fileInputRef.current?.click()}>
                  <Paperclip size={14} />
                </PromptInputAction>
                <PromptInputAction
                  tooltip={isProcessing ? "Processing..." : "Send"}
                  onClick={handleSubmit}
                  disabled={isProcessing || !input.trim()}
                  className={input.trim() ? "bg-primary/20 text-primary hover:bg-primary/30" : ""}
                >
                  {isProcessing ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                </PromptInputAction>
              </PromptInputActions>
            </PromptInput>
            {dragOver && (
              <p className="mt-1.5 text-center text-[9px] text-primary/60 font-mono">
                drop image or asset to attach
              </p>
            )}
          </div>
        </div>
      </div>

      {/* ── Right Panel ───────────────────────────────────────────── */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Tab bar */}
        {showTabs && (
          <div className="shrink-0 flex items-center gap-0 border-b border-border px-2">
            <TabButton active={sideTab === "viewer"} onClick={() => setSideTab("viewer")}>
              <Eye size={11} /> viewer
            </TabButton>
            <TabButton active={sideTab === "code"} onClick={() => setSideTab("code")}>
              <Code size={11} /> code
            </TabButton>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          <AnimatePresence mode="wait">
            {rightPanel.kind === "building" ? (
              <motion.div
                key="building"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex h-full items-center justify-center"
              >
                <BuildingVisualizer stage={rightPanel.stage} toolEvents={rightPanel.toolEvents} />
              </motion.div>
            ) : rightPanel.kind === "env-viewer" ? (
              <motion.div
                key={`env-viewer-${rightPanel.envId}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="h-full flex flex-col gap-3"
              >
                <SuccessBanner envId={rightPanel.envId} />
                <EnvRunner envId={rightPanel.envId} autoStart={validationPassed} />
              </motion.div>
            ) : rightPanel.kind === "env-code" ? (
              <motion.div
                key="env-code"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <SuccessBanner envId={currentEnvId} />
                <div className="mt-3">
                  <CodeViewer files={rightPanel.env.files} />
                </div>
              </motion.div>
            ) : rightPanel.kind === "asset-viewer" ? (
              <motion.div
                key={`asset-${rightPanel.asset.id}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="h-full flex flex-col gap-3"
              >
                <AssetBanner asset={rightPanel.asset} />
                <div className="flex-1 min-h-[300px]">
                  <AssetViewer assetId={rightPanel.asset.id} filename={rightPanel.modelFile} />
                </div>
                {rightPanel.asset.parameters.length > 0 && (
                  <AssetParameterPanel parameters={rightPanel.asset.parameters} />
                )}
              </motion.div>
            ) : rightPanel.kind === "asset-code" ? (
              <motion.div
                key="asset-code"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <AssetBanner asset={selectedAsset!} />
                <div className="mt-3">
                  <CodeViewer files={selectedAsset?.files || {}} />
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex h-full items-center justify-center"
              >
                <div className="text-center">
                  <Terminal size={28} className="mx-auto mb-3 text-muted-foreground/20" />
                  <p className="text-[11px] text-muted-foreground/40 font-mono">
                    output will appear here
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      </div>

      {/* Settings modal */}
      <AnimatePresence>
        {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} />}
      </AnimatePresence>
    </div>
  );
}

// ── Sub-components ──────────────────────────────────────────────────────────

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-2 text-[10px] font-mono transition-colors border-b-2 ${
        active
          ? "border-primary text-primary"
          : "border-transparent text-muted-foreground/50 hover:text-muted-foreground"
      }`}
    >
      {children}
    </button>
  );
}

function ConnectionBadge({ connected }: { connected: boolean }) {
  return (
    <div className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[9px] tracking-wider uppercase">
      <motion.div
        animate={{ scale: connected ? [1, 1.3, 1] : 1 }}
        transition={{ repeat: connected ? Infinity : 0, duration: 2 }}
        className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-emerald-400" : "bg-zinc-600"}`}
      />
      <span className={connected ? "text-emerald-400/70" : "text-zinc-600"}>
        {connected ? "live" : "offline"}
      </span>
    </div>
  );
}

// ── Asset Sidebar Item (draggable) ─────────────────────────────────────────

function AssetSidebarItem({
  asset,
  selected,
  onSelect,
}: {
  asset: AssetMeta;
  selected: boolean;
  onSelect: () => void;
}) {
  const handleDragStart = (e: DragEvent<HTMLDivElement>) => {
    e.dataTransfer.setData("application/x-asset-id", asset.id);
    if (asset.files[0]) {
      e.dataTransfer.setData("application/x-asset-file", asset.files[0]);
    }
    e.dataTransfer.effectAllowed = "copy";
  };

  const typeLabel = asset.file_types[0]?.toUpperCase() || "?";

  return (
    <div
      role="button"
      tabIndex={0}
      draggable
      onDragStart={handleDragStart}
      onClick={onSelect}
      onKeyDown={(e) => { if (e.key === "Enter") onSelect(); }}
      className={`group flex w-full cursor-pointer items-center gap-1.5 rounded px-2 py-1.5 text-left text-[11px] transition-colors hover:bg-white/5 ${
        selected ? "bg-violet-500/10 text-violet-400" : "text-muted-foreground"
      }`}
    >
      <Grip size={8} className="shrink-0 opacity-20 group-hover:opacity-40 cursor-grab" />
      <span className="flex-1 truncate">{asset.name}</span>
      <span className="shrink-0 rounded bg-secondary px-1 py-0.5 text-[8px] text-muted-foreground/40">
        {typeLabel}
      </span>
    </div>
  );
}

// ── Terminal Message with Markdown ──────────────────────────────────────────

function TerminalMessage({
  message,
  onAcceptPlan,
  onModifyPlan,
  onRetry,
  onClearAndRetry,
}: {
  message: ChatMessage;
  onAcceptPlan: () => void;
  onModifyPlan: (modification: string) => void;
  onRetry: () => void;
  onClearAndRetry: () => void;
}) {
  const isUser = message.role === "user";
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState("");

  const isPlan = !isUser && (
    message.content.toLowerCase().includes("ready to build") ||
    message.content.toLowerCase().includes("approve the plan") ||
    message.content.toLowerCase().includes("suggest changes") ||
    message.content.toLowerCase().includes("shall i proceed") ||
    message.content.toLowerCase().includes("want me to proceed") ||
    message.content.toLowerCase().includes("approve or") ||
    message.content.toLowerCase().includes("accept or")
  );

  return (
    <div className="border-b border-border/30 px-4 py-3">
      {isUser ? (
        <div className="flex items-start gap-2">
          <span className="shrink-0 mt-0.5 text-[11px] text-primary/50 select-none font-bold">&gt;</span>
          <div>
            <span className="text-[12px] text-foreground leading-relaxed">{message.content}</span>
            {message.images && message.images.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {message.images.map((img, i) => (
                  <img
                    key={i}
                    src={img}
                    alt={`Attached ${i + 1}`}
                    className="h-16 w-16 rounded-md border border-border/50 object-cover"
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="ml-4">
          <div className="prose-chat">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({ children }) => <h1 className="text-[13px] font-semibold text-foreground mt-3 mb-1.5">{children}</h1>,
                h2: ({ children }) => <h2 className="text-[12px] font-semibold text-foreground/90 mt-2.5 mb-1">{children}</h2>,
                h3: ({ children }) => <h3 className="text-[11px] font-semibold text-foreground/80 mt-2 mb-1">{children}</h3>,
                p: ({ children }) => <p className="text-[11px] leading-relaxed text-muted-foreground mb-1.5">{children}</p>,
                ul: ({ children }) => <ul className="list-disc list-inside space-y-0.5 text-[11px] text-muted-foreground mb-1.5 ml-1">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-inside space-y-0.5 text-[11px] text-muted-foreground mb-1.5 ml-1">{children}</ol>,
                li: ({ children }) => <li className="text-[11px] leading-relaxed">{children}</li>,
                strong: ({ children }) => <strong className="font-semibold text-foreground/80">{children}</strong>,
                em: ({ children }) => <em className="italic text-muted-foreground/80">{children}</em>,
                code: ({ className, children, ...props }) => {
                  const isBlock = className?.includes("language-");
                  if (isBlock) {
                    return (
                      <div className="my-2 rounded-md border border-border bg-secondary/50 overflow-hidden">
                        <div className="flex items-center px-3 py-1 border-b border-border/50">
                          <span className="text-[9px] text-muted-foreground/40 font-mono">
                            {className?.replace("language-", "") || "code"}
                          </span>
                        </div>
                        <pre className="p-3 overflow-x-auto">
                          <code className="text-[10px] leading-[1.6] font-mono text-foreground/80" {...props}>
                            {children}
                          </code>
                        </pre>
                      </div>
                    );
                  }
                  return (
                    <code className="rounded bg-secondary/80 px-1 py-0.5 text-[10px] font-mono text-primary/80" {...props}>
                      {children}
                    </code>
                  );
                },
                pre: ({ children }) => <>{children}</>,
                table: ({ children }) => (
                  <div className="my-2 overflow-x-auto rounded border border-border">
                    <table className="w-full text-[10px] font-mono">{children}</table>
                  </div>
                ),
                th: ({ children }) => <th className="border-b border-border bg-secondary/50 px-2 py-1 text-left text-muted-foreground">{children}</th>,
                td: ({ children }) => <td className="border-b border-border/30 px-2 py-1 text-foreground/70">{children}</td>,
                blockquote: ({ children }) => (
                  <blockquote className="my-1.5 border-l-2 border-primary/30 pl-3 text-[11px] text-muted-foreground/60 italic">
                    {children}
                  </blockquote>
                ),
                hr: () => <hr className="my-3 border-border/50" />,
                a: ({ children, href }) => (
                  <a href={href} className="text-primary/70 underline underline-offset-2 hover:text-primary" target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
          {message.error && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}>
              <pre className="mt-1.5 max-h-[200px] overflow-auto rounded border border-destructive/20 bg-destructive/5 p-2 text-[10px] text-destructive/80">
                {message.error}
              </pre>
              <div className="mt-1.5 flex gap-2">
                <button
                  onClick={onRetry}
                  className="flex items-center gap-1 rounded-md border border-border px-2 py-1 text-[10px] text-muted-foreground hover:bg-white/5"
                >
                  <RotateCcw size={9} /> retry
                </button>
                {message.error.includes("tool_use") && (
                  <button
                    onClick={onClearAndRetry}
                    className="flex items-center gap-1 rounded-md border border-amber-500/30 bg-amber-500/5 px-2 py-1 text-[10px] text-amber-400/80 hover:bg-amber-500/10"
                  >
                    <Trash2 size={9} /> clear session & retry
                  </button>
                )}
              </div>
            </motion.div>
          )}
          {isPlan && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mt-3 space-y-2"
            >
              {editing ? (
                <div className="space-y-2">
                  <textarea
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    placeholder="what should be changed..."
                    className="w-full rounded-md border border-border bg-secondary px-2.5 py-1.5 text-[11px] text-foreground font-mono placeholder:text-muted-foreground/30 focus:border-primary focus:outline-none"
                    rows={2}
                    autoFocus
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={() => { if (editText.trim()) { onModifyPlan(editText); setEditing(false); setEditText(""); } }}
                      className="flex items-center gap-1 rounded-md border border-primary/30 bg-primary/10 px-3 py-1 text-[10px] text-primary hover:bg-primary/20"
                    >
                      <Send size={9} /> send
                    </button>
                    <button onClick={() => setEditing(false)} className="text-[10px] text-muted-foreground hover:text-foreground">
                      cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex gap-2">
                  <button
                    onClick={onAcceptPlan}
                    className="flex items-center gap-1.5 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-[10px] font-medium text-emerald-400 transition-colors hover:bg-emerald-500/20"
                  >
                    <Check size={11} /> accept plan
                  </button>
                  <button
                    onClick={() => setEditing(true)}
                    className="flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-[10px] text-muted-foreground transition-colors hover:bg-white/5"
                  >
                    <Edit3 size={11} /> modify
                  </button>
                </div>
              )}
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Tool Line ───────────────────────────────────────────────────────────────

function ToolLine({ event }: { event: ToolEvent }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex items-center gap-2 py-0.5 text-[10px]"
    >
      <span className="text-muted-foreground/40 select-none">$</span>
      {event.status === "running" ? (
        <Loader2 size={10} className="animate-spin shrink-0" style={{ color: event.color || undefined }} />
      ) : event.status === "success" ? (
        <CheckCircle size={10} className="shrink-0" style={{ color: event.color ? `${event.color}b3` : "#34d399b3" }} />
      ) : (
        <X size={10} className="text-destructive/70 shrink-0" />
      )}
      <span
        className="font-medium"
        style={{ color: event.status === "error" ? undefined : event.color ? `${event.color}cc` : undefined }}
      >
        {event.label}
      </span>
      {event.detail && event.status !== "running" && (
        <span className="truncate text-muted-foreground/30">{event.detail.slice(0, 60)}</span>
      )}
    </motion.div>
  );
}

// ── Pulse Loader ────────────────────────────────────────────────────────────

function PulseLoader() {
  return (
    <div className="flex items-center gap-2">
      <span className="text-muted-foreground/40 text-[10px] select-none">$</span>
      <div className="flex items-center gap-1">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="h-1 w-1 rounded-full bg-primary/50"
            animate={{ opacity: [0.2, 1, 0.2], scale: [0.8, 1.2, 0.8] }}
            transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
          />
        ))}
      </div>
      <span className="text-[10px] text-muted-foreground/30">thinking</span>
    </div>
  );
}

// ── Success Banner ──────────────────────────────────────────────────────────

function SuccessBanner({ envId }: { envId: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/5 px-4 py-2.5">
      <CheckCircle size={14} className="text-emerald-400 shrink-0" />
      <div>
        <p className="text-[11px] font-medium text-emerald-400">environment validated</p>
        <p className="text-[10px] text-emerald-400/50 font-mono">envs/{envId}/</p>
      </div>
    </div>
  );
}

function AssetBanner({ asset }: { asset: AssetDetails }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-primary/20 bg-primary/5 px-4 py-2.5">
      <Box size={14} className="text-primary shrink-0" />
      <div>
        <p className="text-[11px] font-medium text-primary">{asset.id.replace(/_/g, " ")}</p>
        <p className="text-[10px] text-primary/50 font-mono truncate max-w-[300px]">
          {asset.description || "no description"}
        </p>
      </div>
    </div>
  );
}

// ── Code Viewer ─────────────────────────────────────────────────────────────

function CodeViewer({ files }: { files: Record<string, string> }) {
  const [activeFile, setActiveFile] = useState("");
  const fileNames = Object.keys(files);

  useEffect(() => {
    if (fileNames.length > 0 && !fileNames.includes(activeFile)) {
      setActiveFile(fileNames.includes("env.py") ? "env.py" : fileNames[0]);
    }
  }, [fileNames, activeFile]);

  if (fileNames.length === 0) return null;
  const content = files[activeFile] || "";

  return (
    <div className="flex flex-col rounded-xl border border-border overflow-hidden">
      <div className="flex items-center gap-0 border-b border-border bg-card/50 px-1">
        {fileNames.map((f) => (
          <button
            key={f}
            onClick={() => setActiveFile(f)}
            className={`px-3 py-2 text-[10px] font-mono transition-colors border-b-2 ${
              activeFile === f
                ? "border-primary text-primary bg-primary/5"
                : "border-transparent text-muted-foreground/50 hover:text-muted-foreground"
            }`}
          >
            {f}
          </button>
        ))}
      </div>
      <div className="max-h-[600px] overflow-auto bg-background p-4">
        <pre className="text-[11px] leading-[1.7] font-mono">
          {content.split("\n").map((line, i) => (
            <div key={i} className="flex">
              <span className="mr-4 inline-block w-8 shrink-0 select-none text-right text-muted-foreground/20">{i + 1}</span>
              <code className="text-foreground/80">{line}</code>
            </div>
          ))}
        </pre>
      </div>
    </div>
  );
}

// ── Empty State ─────────────────────────────────────────────────────────────

function EmptyState({ onSampleClick }: { onSampleClick: (s: string) => void }) {
  const samples = [
    "2D grid maze with moving obstacles",
    "3-DOF robotic arm reaching task",
    "Drone navigation in a warehouse",
  ];

  return (
    <div className="flex h-full flex-col items-center justify-center gap-8 px-8">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <motion.div
          className="mb-3"
          animate={{ textShadow: ["0 0 0px transparent", "0 0 24px rgba(21,245,186,0.35)", "0 0 0px transparent"] }}
          transition={{ duration: 3, repeat: Infinity }}
        >
          <span className="text-[26px] font-black font-mono tracking-tight select-none">
            <span style={{ color: "#15F5BA" }}>0</span><span className="text-white/90">RL</span>
          </span>
        </motion.div>
        <h2 className="text-[11px] font-medium text-muted-foreground/60 uppercase tracking-widest">from words to worlds</h2>
        <p className="mt-1 text-[11px] text-muted-foreground/60">
          describe an environment. i&apos;ll build it.
        </p>
      </motion.div>

      <div className="grid w-full max-w-[280px] gap-1.5">
        {samples.map((s, i) => (
          <motion.button
            key={s}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 + i * 0.08 }}
            whileHover={{ x: 4 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onSampleClick(s)}
            className="rounded-xl border border-border/50 px-3 py-2 text-left text-[11px] text-muted-foreground/60 transition-colors hover:border-primary/20 hover:bg-primary/5 hover:text-foreground"
          >
            <span className="text-primary/40 mr-1.5">&gt;</span>{s}
          </motion.button>
        ))}
      </div>
    </div>
  );
}

// ── Settings ────────────────────────────────────────────────────────────────

function SettingsPanel({ onClose }: { onClose: () => void }) {
  const [provider, setProvider] = useState("anthropic");
  const [model, setModel] = useState("claude-sonnet-4-6");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const models: Record<string, string[]> = {
    anthropic: ["claude-opus-4-6", "claude-sonnet-4-6"],
    openai: ["gpt-4o", "gpt-4o-mini", "o1-mini"],
    "google-genai": ["gemini-2.5-flash-preview-04-17", "gemini-2.0-flash"],
  };

  useEffect(() => {
    fetch(`${API}/api/settings`)
      .then((r) => r.json())
      .then((d) => { if (d.provider) setProvider(d.provider); if (d.model) setModel(d.model); })
      .catch(() => {});
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setError("");
    try {
      const r = await fetch(`${API}/api/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, model, api_key: apiKey || undefined }),
      });
      if (!r.ok) throw new Error(`Save failed (${r.status})`);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
    setSaving(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0, y: 10 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.95, opacity: 0, y: 10 }}
        transition={{ duration: 0.2 }}
        onClick={(e) => e.stopPropagation()}
        className="w-[380px] rounded-2xl border border-border bg-card p-5 shadow-2xl"
      >
        <h2 className="text-[12px] font-semibold text-foreground font-mono">settings</h2>
        <p className="mb-4 mt-1 text-[10px] text-muted-foreground">configure llm provider</p>

        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-[10px] text-muted-foreground">provider</label>
            <select
              value={provider}
              onChange={(e) => { setProvider(e.target.value); const m = models[e.target.value]; if (m?.[0]) setModel(m[0]); }}
              className="w-full rounded-md border border-border bg-secondary px-2.5 py-1.5 text-[11px] text-foreground font-mono focus:border-primary focus:outline-none"
            >
              <option value="anthropic">anthropic</option>
              <option value="openai">openai</option>
              <option value="google-genai">google-genai</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-[10px] text-muted-foreground">model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full rounded-md border border-border bg-secondary px-2.5 py-1.5 text-[11px] text-foreground font-mono focus:border-primary focus:outline-none"
            >
              {(models[provider] || []).map((m) => (<option key={m} value={m}>{m}</option>))}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-[10px] text-muted-foreground">api key</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="leave empty to use .env"
              className="w-full rounded-md border border-border bg-secondary px-2.5 py-1.5 text-[11px] text-foreground font-mono placeholder:text-muted-foreground/30 focus:border-primary focus:outline-none"
            />
          </div>
        </div>

        {error && (
          <p className="mt-2 text-[10px] text-destructive font-mono">{error}</p>
        )}

        <div className="mt-5 flex justify-end gap-2">
          <button onClick={onClose} className="rounded-md px-3 py-1.5 text-[10px] text-muted-foreground hover:bg-white/5">
            cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="rounded-md bg-primary px-3 py-1.5 text-[10px] font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {saving ? "saving..." : "save"}
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}
