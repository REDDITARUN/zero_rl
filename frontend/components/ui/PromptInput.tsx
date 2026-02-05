"use client";

import { FormEvent } from "react";

interface PromptInputProps {
  value: string;
  loading?: boolean;
  placeholder?: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

export function PromptInput({
  value,
  loading = false,
  placeholder,
  onChange,
  onSubmit
}: PromptInputProps) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-2">
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder ?? "Describe your RL environment..."}
        className="h-24 w-full resize-none rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3 text-sm text-[var(--ink)] outline-none transition focus:border-[var(--accent)]"
      />
      <button
        type="submit"
        disabled={loading || value.trim().length < 3}
        className="w-full rounded-xl bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white transition disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? "Generating..." : "Generate Environment"}
      </button>
    </form>
  );
}
