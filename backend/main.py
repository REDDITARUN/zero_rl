"""FastAPI application — REST + WebSocket entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.routes.assets import router as assets_router  # noqa: E402
from backend.routes.chat import router as chat_router  # noqa: E402
from backend.routes.envs import router as envs_router  # noqa: E402
from backend.routes.settings import router as settings_router  # noqa: E402
from backend.services.runner import stream_env_frames  # noqa: E402

app = FastAPI(title="ZeroRL", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:1420",  # Tauri dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(assets_router, prefix="/api/assets", tags=["assets"])
app.include_router(envs_router, prefix="/api/envs", tags=["envs"])
app.include_router(settings_router, prefix="/api/settings", tags=["settings"])


@app.get("/api/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.websocket("/ws/env/{env_id}/frames")
async def env_frames_ws(websocket: WebSocket, env_id: str) -> None:
    """Interactive env viewer — streams frames, accepts actions from client.

    Client sends: {"action": [0.1, -0.2, ...]} or {"action": 2} or {"cmd": "reset"}
    Server sends: {"step", "frame", "reward", "done", "actions", "info", "action_space"}
    If no action received, uses random action (auto-play mode).
    """
    await websocket.accept()
    try:
        async for frame_data in stream_env_frames(env_id, websocket=websocket):
            await websocket.send_text(json.dumps(frame_data))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
