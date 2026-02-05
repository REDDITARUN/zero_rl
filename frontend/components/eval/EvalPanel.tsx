"use client";

import { useEval } from "@/hooks/useEval";

export function EvalPanel({ envId }: { envId: string | null }) {
  const { events, latest, busy, episodes, setEpisodes, maxSteps, setMaxSteps, start } = useEval(envId);
  const frameSrc = latest?.frame ? `data:image/png;base64,${latest.frame}` : null;

  return (
    <div className="grid h-full gap-3 p-4 md:grid-cols-[1.4fr_1fr]">
      <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg text-[var(--ink)]">Evaluation</h3>
          <button
            disabled={!envId || busy}
            onClick={() => void start()}
            className="rounded-lg bg-[var(--accent)] px-4 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-50"
            type="button"
          >
            {busy ? "Running..." : "Start Eval"}
          </button>
        </div>

        <div className="mb-3 grid gap-2 md:grid-cols-2">
          <label className="text-xs text-[var(--soft-ink)]">
            Episodes
            <input
              type="number"
              min={1}
              max={20}
              value={episodes}
              onChange={(event) => setEpisodes(Number(event.target.value))}
              className="mt-1 w-full rounded-lg border border-[var(--border)] bg-[#fffaf0] px-3 py-2 text-sm"
            />
          </label>
          <label className="text-xs text-[var(--soft-ink)]">
            Max Steps
            <input
              type="number"
              min={10}
              max={2000}
              value={maxSteps}
              onChange={(event) => setMaxSteps(Number(event.target.value))}
              className="mt-1 w-full rounded-lg border border-[var(--border)] bg-[#fffaf0] px-3 py-2 text-sm"
            />
          </label>
        </div>

        <div className="flex h-[340px] items-center justify-center rounded-lg border border-dashed border-[var(--border)] bg-[#fdf8ef]">
          {frameSrc ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={frameSrc} alt="Eval frame" className="max-h-full rounded-md" />
          ) : (
            <p className="text-sm text-[var(--soft-ink)]">Run eval to see policy rollout.</p>
          )}
        </div>
      </div>

      <div className="space-y-3 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3">
        <h4 className="text-sm text-[var(--ink)]">Action Trace</h4>
        <div className="text-xs text-[var(--soft-ink)]">
          <p>Status: {latest?.status ?? "idle"}</p>
          <p>Episode: {latest?.episode ?? 0}</p>
          <p>Step: {latest?.step ?? 0}</p>
          <p>Action: {latest?.action ?? "-"}</p>
          <p>Reward: {(latest?.reward ?? 0).toFixed(3)}</p>
          <p>Cumulative: {(latest?.cumulative_reward ?? 0).toFixed(3)}</p>
          <p>{latest?.message ?? ""}</p>
        </div>

        <div className="max-h-[320px] overflow-auto rounded-lg border border-[var(--border)] bg-[#fbf6ed] p-2 text-xs text-[var(--soft-ink)]">
          {events.length === 0 ? (
            <p>No eval events yet.</p>
          ) : (
            <ul className="space-y-1">
              {events
                .filter((event) => typeof event.step === "number")
                .slice(-30)
                .map((event, idx) => (
                  <li key={`${event.episode}-${event.step}-${idx}`}>
                    ep{event.episode} s{event.step} a{event.action} r{(event.reward ?? 0).toFixed(2)}
                  </li>
                ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
