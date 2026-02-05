interface CodeBlockProps {
  code: string;
}

export function CodeBlock({ code }: CodeBlockProps) {
  return (
    <pre className="scrollbar-thin overflow-x-auto rounded-xl border border-[var(--border)] bg-[#fbf6ed] p-3 text-xs leading-5 text-[var(--ink)]">
      <code>{code}</code>
    </pre>
  );
}
