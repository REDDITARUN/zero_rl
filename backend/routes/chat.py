"""WebSocket handler for chat — streams orchestrator events to the frontend."""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.orchestrator import AgentEvent, Orchestrator

router = APIRouter()

_orchestrators: dict[str, Orchestrator] = {}


def _make_event_sender(
    websocket: WebSocket, loop: asyncio.AbstractEventLoop
) -> Any:
    """Create a send_fn closure that can be updated on reconnect."""

    def on_event(event: AgentEvent) -> None:
        asyncio.run_coroutine_threadsafe(
            websocket.send_text(
                json.dumps(
                    {
                        "type": "agent_event",
                        "event": {"type": event.type, "data": event.data},
                    }
                )
            ),
            loop,
        )

    return on_event


def _get_orchestrator(
    session_id: str, websocket: WebSocket, loop: asyncio.AbstractEventLoop
) -> Orchestrator:
    """Get or create an orchestrator, always updating the event callback."""
    on_event = _make_event_sender(websocket, loop)

    if session_id not in _orchestrators:
        _orchestrators[session_id] = Orchestrator(on_event=on_event)
    else:
        _orchestrators[session_id].on_event = on_event
    return _orchestrators[session_id]


def _extract_env_id(orchestrator: Orchestrator) -> str:
    """Extract env_id from orchestrator state or message history."""
    import re

    for msg in reversed(orchestrator.messages[-20:]):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and "envs/" in content:
            m = re.search(r"envs/([^/\s]+)/", content)
            if m:
                return m.group(1)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if "envs/" in text:
                        m = re.search(r"envs/([^/\s]+)/", text)
                        if m:
                            return m.group(1)
    return ""


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    """Bidirectional WebSocket for the agent chat loop.

    Client sends:
        {"type": "message", "content": "...", "images": [...], "session_id": "..."}
        {"type": "clear_session", "session_id": "..."}
        {"type": "ping"}

    Server streams back:
        {"type": "agent_event", "event": {"type": "...", "data": {...}}}
        {"type": "response",   "content": "...", "env_id": "..."}
        {"type": "error",      "message": "..."}
        {"type": "pong"}
    """
    await websocket.accept()
    session_id = "default"
    loop = asyncio.get_running_loop()

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            if data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if data.get("type") == "clear_session":
                sid = data.get("session_id", session_id)
                if sid in _orchestrators:
                    _orchestrators[sid].clear_history()
                await websocket.send_text(
                    json.dumps({"type": "session_cleared", "session_id": sid})
                )
                continue

            if data.get("type") != "message":
                continue

            content = data.get("content", "")
            images = data.get("images", [])
            session_id = data.get("session_id", session_id)

            orchestrator = _get_orchestrator(session_id, websocket, loop)

            await websocket.send_text(
                json.dumps({"type": "status", "status": "processing"})
            )

            try:
                response = await asyncio.to_thread(
                    orchestrator.run, content, images or None
                )
                env_id = _extract_env_id(orchestrator)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "response",
                            "content": response,
                            "env_id": env_id,
                        }
                    )
                )
            except Exception as e:
                err_str = str(e)
                is_tool_corruption = (
                    "tool_use" in err_str and "tool_result" in err_str
                )
                if is_tool_corruption:
                    orchestrator._sanitize_messages()

                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": err_str,
                            "traceback": traceback.format_exc(),
                            "recoverable": is_tool_corruption,
                        }
                    )
                )

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
