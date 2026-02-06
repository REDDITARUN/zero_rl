export type AgentRunStatus = "idle" | "working" | "complete" | "error";

export interface AgentState {
  id: string;
  name: string;
  status: AgentRunStatus;
  message: string;
  logs?: string[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
}

export interface ChatResponse {
  env_id: string;
  name: string;
  success: boolean;
  summary: string;
  files: Record<string, string>;
  validation: {
    success: boolean;
    stage: string;
    errors: string[];
    warnings: string[];
  };
  saved: boolean;
}

export interface EnvironmentSummary {
  env_id: string;
  name: string;
  prompt: string;
  created_at: string;
  saved: boolean;
}

export interface EnvironmentMeta {
  env_id: string;
  name: string;
  prompt: string;
  created_at: string;
  updated_at: string;
  saved: boolean;
  action_space: {
    type: string;
    n: number;
    actions: string[];
  };
  observation_space: {
    type: string;
    shape: number[];
    dtype: string;
    description?: string[];
  };
  reward: string;
  files: Record<string, string>;
  last_training?: {
    algorithm?: "PPO" | "DQN" | "A2C";
    model_path?: string;
    timesteps?: number;
  };
}

export interface RuntimeState {
  env_id: string;
  step: number;
  last_action: string | null;
  last_reward: number;
  cumulative_reward: number;
  terminated: boolean;
  truncated: boolean;
  observation: unknown;
  info: Record<string, unknown>;
  frame: string;
  available_actions: string[];
  history: Array<{
    step: number;
    action: string;
    reward: number;
    cumulative_reward: number;
    terminated: boolean;
    truncated: boolean;
  }>;
}

export interface TrainingConfig {
  algorithm: "PPO" | "DQN" | "A2C";
  timesteps: number;
  learning_rate: number;
  gamma: number;
  batch_size: number;
  n_steps: number;
  epsilon: number;
}

export interface TrainingPoint {
  env_id: string;
  status: "queued" | "running" | "complete" | "error";
  timesteps: number;
  reward: number;
  episode: number;
  avg_reward_100: number;
  message: string;
  payload?: Record<string, unknown>;
}

export interface EvalPoint {
  env_id: string;
  status: "running" | "complete" | "error";
  message?: string;
  episode?: number;
  step?: number;
  action?: number;
  reward?: number;
  cumulative_reward?: number;
  terminated?: boolean;
  truncated?: boolean;
  observation?: unknown;
  info?: Record<string, unknown>;
  frame?: string;
}
