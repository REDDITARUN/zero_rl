interface MarkdownProps {
  content: string;
}

export function Markdown({ content }: MarkdownProps) {
  return (
    <div className="space-y-2 text-sm leading-relaxed text-[var(--ink)]/90">
      {content.split("\n").map((line, idx) => (
        <p key={`${line}-${idx}`} className="whitespace-pre-wrap">
          {line}
        </p>
      ))}
    </div>
  );
}
