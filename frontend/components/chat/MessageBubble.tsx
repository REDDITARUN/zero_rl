import { Markdown } from "@/components/ui/Markdown";
import type { ChatMessage } from "@/lib/types";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <article
      className={`mb-3 rounded-2xl border px-3 py-2 ${
        isUser
          ? "ml-8 border-[var(--accent)]/30 bg-[var(--accent-soft)] text-[var(--ink)]"
          : "mr-8 border-[var(--border)] bg-[var(--surface)]"
      }`}
    >
      <p className="mb-1 text-[10px] uppercase tracking-[0.16em] text-[var(--soft-ink)]">
        {isUser ? "You" : "ZeroRL"}
      </p>
      <Markdown content={message.content} />
    </article>
  );
}
