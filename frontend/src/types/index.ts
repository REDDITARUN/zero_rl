export type EnvironmentMeta = {
  id: string;
  name: string;
  description: string;
  type: "gym" | "genesis";
  created_at: string;
  thumbnail?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  images?: string[];
  error?: string;
  timestamp: number;
};

export type EnvFile = {
  name: string;
  content: string;
};

export type EnvDetails = {
  id: string;
  config: Record<string, unknown>;
  files: Record<string, string>;
};
