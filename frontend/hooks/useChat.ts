"use client";

import { useCallback, useState } from "react";
import { createEnvironment, getEnvironment } from "@/lib/api";
import type { ChatMessage, EnvironmentMeta } from "@/lib/types";

interface UseChatOptions {
  onEnvCreated?: (envId: string) => void;
}

export function useChat(options: UseChatOptions = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastEnv, setLastEnv] = useState<EnvironmentMeta | null>(null);

  const createRunId = () =>
    `run-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

  const sendPrompt = useCallback(
    async (content: string, envId?: string | null) => {
      setLoading(true);
      setError(null);
      const runId = createRunId();
      if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("zerorl:run-start", { detail: { runId } }));
      }
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content,
        createdAt: Date.now()
      };
      setMessages((prev) => [...prev, userMessage]);

      try {
        const created = await createEnvironment(content, envId, runId);
        const env = await getEnvironment(created.env_id);

        setLastEnv(env);
        options.onEnvCreated?.(created.env_id);

        const assistantMessage: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: `${created.summary}\n\nValidation: ${created.validation.stage}\nSaved: ${created.saved ? "yes" : "no"}`,
          createdAt: Date.now()
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to generate environment";
        setError(message);
        setMessages((prev) => [
          ...prev,
          {
            id: `assistant-error-${Date.now()}`,
            role: "assistant",
            content: `Generation failed:\n${message}`,
            createdAt: Date.now()
          }
        ]);
      } finally {
        setLoading(false);
      }
    },
    [options]
  );

  const clearConversation = useCallback(() => {
    setMessages([]);
    setError(null);
    setLastEnv(null);
  }, []);

  return {
    messages,
    loading,
    error,
    lastEnv,
    sendPrompt,
    clearConversation
  };
}
