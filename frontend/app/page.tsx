"use client";

import { useState } from "react";
import { AgentStatusPanel } from "@/components/agents/AgentStatusPanel";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { TabPanel } from "@/components/TabPanel";
import { useSessionStore } from "@/lib/sessionStore";

export default function Home() {
  const [activeEnvId, setActiveEnvId] = useState<string | null>(null);
  const [chatResetToken, setChatResetToken] = useState(0);
  const clearEnvSession = useSessionStore((state) => state.clearEnvSession);

  function resetWorkspace() {
    if (activeEnvId) {
      clearEnvSession(activeEnvId);
    }
    setActiveEnvId(null);
    setChatResetToken((prev) => prev + 1);
  }

  return (
    <main className="min-h-screen p-3 lg:p-4">
      <div className="mx-auto grid h-[calc(100vh-1.5rem)] max-w-[1800px] grid-cols-1 gap-3 lg:h-[calc(100vh-2rem)] lg:grid-cols-5 lg:gap-4">
        <section className="glass-panel flex min-h-[45vh] flex-col overflow-hidden lg:col-span-2 lg:min-h-full">
          <div className="min-h-0 flex-1">
            <ChatPanel envId={activeEnvId} onEnvCreated={setActiveEnvId} resetToken={chatResetToken} />
          </div>
          <div className="border-t border-[var(--border)] bg-[var(--surface-muted)]/35 p-3">
            <AgentStatusPanel />
          </div>
        </section>

        <section className="glass-panel min-h-[50vh] overflow-hidden lg:col-span-3 lg:min-h-full">
          <TabPanel envId={activeEnvId} onEnvSelected={setActiveEnvId} onResetWorkspace={resetWorkspace} />
        </section>
      </div>
    </main>
  );
}
