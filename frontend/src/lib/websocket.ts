/**
 * WebSocket connection manager with auto-reconnect.
 * Connects to the FastAPI backend at /ws/chat.
 */

export type WSMessage =
  | { type: "agent_event"; event: { type: string; data: Record<string, unknown> } }
  | { type: "status"; status: string }
  | { type: "response"; content: string; env_id?: string }
  | { type: "error"; message: string; traceback?: string; recoverable?: boolean }
  | { type: "session_cleared"; session_id: string }
  | { type: "pong" };

export type WSHandler = (msg: WSMessage) => void;

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/chat";

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private handlers = new Set<WSHandler>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 16000;
  private _connected = false;
  private pingInterval: ReturnType<typeof setInterval> | null = null;

  get connected(): boolean {
    return this._connected;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) return;

    try {
      this.ws = new WebSocket(WS_URL);
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this._connected = true;
      this.reconnectDelay = 1000;
      this.notify({ type: "status", status: "connected" });
      this.pingInterval = setInterval(() => {
        this.send({ type: "ping" });
      }, 25000);
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WSMessage;
        this.notify(msg);
      } catch { /* ignore malformed */ }
    };

    this.ws.onclose = () => {
      this._connected = false;
      if (this.pingInterval) clearInterval(this.pingInterval);
      this.notify({ type: "status", status: "disconnected" });
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  disconnect(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.pingInterval) clearInterval(this.pingInterval);
    this.ws?.close();
    this.ws = null;
    this._connected = false;
  }

  send(data: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  subscribe(handler: WSHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  private notify(msg: WSMessage): void {
    this.handlers.forEach((h) => h(msg));
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, this.reconnectDelay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
  }
}

export const wsManager = new WebSocketManager();
