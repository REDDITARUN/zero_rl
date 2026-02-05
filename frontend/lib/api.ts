import type {
  ChatResponse,
  EnvironmentMeta,
  EnvironmentSummary,
  RuntimeState,
  TrainingConfig
} from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function createEnvironment(prompt: string, envId?: string | null): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, env_id: envId ?? null })
  });
  return parseResponse<ChatResponse>(response);
}

export async function getEnvironment(envId: string): Promise<EnvironmentMeta> {
  const response = await fetch(`${API_BASE}/envs/${envId}`, { cache: "no-store" });
  return parseResponse<EnvironmentMeta>(response);
}

export async function listEnvironments(): Promise<EnvironmentSummary[]> {
  const response = await fetch(`${API_BASE}/envs`, { cache: "no-store" });
  return parseResponse<EnvironmentSummary[]>(response);
}

export async function saveEnvironment(envId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/envs/${envId}/save`, { method: "POST" });
  await parseResponse<Record<string, unknown>>(response);
}

export async function getEnvironmentFile(envId: string, filename: string): Promise<string> {
  const response = await fetch(`${API_BASE}/envs/${envId}/files/${filename}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load ${filename}`);
  }
  return response.text();
}

export async function resetEnvironment(envId: string): Promise<RuntimeState> {
  const response = await fetch(`${API_BASE}/envs/${envId}/reset`, {
    method: "POST"
  });
  return parseResponse<RuntimeState>(response);
}

export async function stepEnvironment(envId: string, action: number | string): Promise<RuntimeState> {
  const response = await fetch(`${API_BASE}/envs/${envId}/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action })
  });
  return parseResponse<RuntimeState>(response);
}

export async function getEnvironmentState(envId: string): Promise<RuntimeState> {
  const response = await fetch(`${API_BASE}/envs/${envId}/state`, { cache: "no-store" });
  return parseResponse<RuntimeState>(response);
}

export async function startTraining(envId: string, config: TrainingConfig): Promise<void> {
  const response = await fetch(`${API_BASE}/train/${envId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config)
  });
  await parseResponse<Record<string, unknown>>(response);
}

export async function startEvaluation(envId: string, episodes: number, maxSteps: number): Promise<void> {
  const response = await fetch(`${API_BASE}/eval/${envId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ episodes, max_steps: maxSteps })
  });
  await parseResponse<Record<string, unknown>>(response);
}

export async function getRenderFrame(envId: string): Promise<string> {
  const response = await fetch(`${API_BASE}/render/${envId}`, { cache: "no-store" });
  const payload = await parseResponse<{ frame: string }>(response);
  return `data:image/png;base64,${payload.frame}`;
}

export async function downloadEnvZip(envId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/envs/${envId}/download`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Download failed");
  }
  return response.blob();
}
