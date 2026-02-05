interface FileTreeProps {
  files: string[];
  selected: string | null;
  onSelect: (name: string) => void;
}

export function FileTree({ files, selected, onSelect }: FileTreeProps) {
  return (
    <aside className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-2">
      <p className="mb-2 px-2 text-xs uppercase tracking-[0.18em] text-[var(--soft-ink)]">Files</p>
      <ul className="space-y-1">
        {files.map((file) => (
          <li key={file}>
            <button
              onClick={() => onSelect(file)}
              type="button"
              className={`w-full rounded-md px-2 py-1.5 text-left text-sm ${
                selected === file ? "bg-[var(--accent-soft)] text-[var(--ink)]" : "text-[var(--soft-ink)] hover:bg-[#f3ead9]"
              }`}
            >
              {file}
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}
