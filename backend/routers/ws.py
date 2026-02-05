"""WebSocket hub for agent and training status streams."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["ws"])


class WebSocketHub:
    """Tracks active websocket clients by channel."""

    def __init__(self) -> None:
        self.agent_clients: set[WebSocket] = set()
        self.train_clients: set[WebSocket] = set()
        self.eval_clients: set[WebSocket] = set()

    async def register(self, channel: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if channel == "agents":
            self.agent_clients.add(websocket)
        elif channel == "train":
            self.train_clients.add(websocket)
        else:
            self.eval_clients.add(websocket)

    def unregister(self, channel: str, websocket: WebSocket) -> None:
        if channel == "agents":
            self.agent_clients.discard(websocket)
        elif channel == "train":
            self.train_clients.discard(websocket)
        else:
            self.eval_clients.discard(websocket)

    async def broadcast_agent(self, event: dict) -> None:
        await self._broadcast(self.agent_clients, event)

    async def broadcast_train(self, event: dict) -> None:
        await self._broadcast(self.train_clients, event)

    async def broadcast_eval(self, event: dict) -> None:
        await self._broadcast(self.eval_clients, event)

    async def _broadcast(self, sockets: set[WebSocket], payload: dict) -> None:
        stale: list[WebSocket] = []
        for socket in sockets:
            try:
                await socket.send_json(payload)
            except Exception:  # noqa: BLE001
                stale.append(socket)
        for socket in stale:
            sockets.discard(socket)


ws_hub = WebSocketHub()


@router.websocket("/ws/agents")
async def agents_ws(websocket: WebSocket) -> None:
    """WebSocket stream for agent status updates."""

    await ws_hub.register("agents", websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_hub.unregister("agents", websocket)


@router.websocket("/ws/train")
async def train_ws(websocket: WebSocket) -> None:
    """WebSocket stream for training updates."""

    await ws_hub.register("train", websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_hub.unregister("train", websocket)


@router.websocket("/ws/eval")
async def eval_ws(websocket: WebSocket) -> None:
    """WebSocket stream for evaluation rollout updates."""

    await ws_hub.register("eval", websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_hub.unregister("eval", websocket)
