"use client";

import { useMemo, useState } from "react";
import { PlanCard } from "@/components/chat/PlanCard";
import { MessageBubble } from "@/components/chat/MessageBubble";
import { QuickActions } from "@/components/chat/QuickActions";
import { Loader } from "@/components/ui/Loader";
import { MessageList } from "@/components/ui/MessageList";
import { PromptInput } from "@/components/ui/PromptInput";
import { useChat } from "@/hooks/useChat";

interface ChatPanelProps {
  envId: string | null;
  onEnvCreated: (envId: string) => void;
}

export function ChatPanel({ envId, onEnvCreated }: ChatPanelProps) {
  const [draft, setDraft] = useState("");
  const { messages, loading, error, sendPrompt } = useChat({ onEnvCreated });

  const subtitle = useMemo(
    () => (envId ? `Active environment: ${envId.slice(0, 8)}` : "No environment selected"),
    [envId]
  );

  async function submit(next?: string) {
    const value = (next ?? draft).trim();
    if (!value) {
      return;
    }
    setDraft("");
    await sendPrompt(value, envId);
  }

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-[var(--border)] px-4 py-3">
        <h1 className="text-xl text-[var(--ink)]">ZeroRL Studio</h1>
        <p className="text-xs text-[var(--soft-ink)]">{subtitle}</p>
      </header>

      <div className="min-h-0 flex-1">
        <MessageList>
          <div className="py-3">
            <PlanCard />
            <QuickActions disabled={loading} onSelect={(prompt) => void submit(prompt)} />
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {loading ? <Loader label="Agents working" /> : null}
            {error ? <p className="mt-2 text-xs text-red-700">{error}</p> : null}
          </div>
        </MessageList>
      </div>

      <footer className="border-t border-[var(--border)] p-3">
        <PromptInput
          value={draft}
          loading={loading}
          placeholder="Describe a new RL environment or refinement..."
          onChange={setDraft}
          onSubmit={() => {
            void submit();
          }}
        />
      </footer>
    </div>
  );
}
