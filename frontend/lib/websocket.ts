const WS_BASE = process.env.NEXT_PUBLIC_WS_BASE ?? "ws://localhost:8000";

export function connectSocket(path: string, onMessage: (payload: unknown) => void): () => void {
  const socket = new WebSocket(`${WS_BASE}${path}`);

  socket.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data));
    } catch {
      // no-op
    }
  };

  const ping = setInterval(() => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send("ping");
    }
  }, 10000);

  return () => {
    clearInterval(ping);
    socket.close();
  };
}
