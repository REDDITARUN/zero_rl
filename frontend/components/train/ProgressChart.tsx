"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type { TrainingPoint } from "@/lib/types";

export function ProgressChart({ events }: { events: TrainingPoint[] }) {
  const rows = events.map((event) => ({
    episode: event.episode,
    reward: Number(event.reward.toFixed(3)),
    avg: Number(event.avg_reward_100.toFixed(3))
  }));

  return (
    <div className="h-72 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#d7c8b0" />
          <XAxis dataKey="episode" stroke="#7c715f" fontSize={12} />
          <YAxis stroke="#7c715f" fontSize={12} />
          <Tooltip />
          <Line type="monotone" dataKey="reward" stroke="#3f5a4b" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="avg" stroke="#8f6b3f" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
