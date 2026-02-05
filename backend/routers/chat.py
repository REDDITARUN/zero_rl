"""Chat orchestration router."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException

from codex_client import CodexSDKError
from models.schemas import ChatRequest, ChatResponse, ValidationResult
from orchestrator import Orchestrator
from routers.envs import get_env_record, put_env_record
from routers.ws import ws_hub

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def create_or_update_environment(request: ChatRequest) -> ChatResponse:
    """Generate or modify environment files from natural language prompt."""

    base_record = None
    if request.env_id:
        try:
            base_record = get_env_record(request.env_id)
        except HTTPException as exc:
            raise HTTPException(status_code=404, detail=f"Base environment not found: {request.env_id}") from exc

    async def status_callback(event: dict) -> None:
        await ws_hub.broadcast_agent(event)

    orchestrator = Orchestrator(status_callback=status_callback)
    try:
        result = await orchestrator.generate_environment(
            request.prompt,
            env_id=request.env_id,
            base_record=base_record,
        )
    except CodexSDKError as exc:
        raise HTTPException(status_code=503, detail=f"Codex generation failed: {exc}") from exc

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result)

    now = datetime.utcnow().isoformat()
    created_at = base_record["created_at"] if base_record else now
    record = {
        "env_id": result["env_id"],
        "name": result["name"],
        "prompt": request.prompt,
        "created_at": created_at,
        "updated_at": now,
        "saved": False,
        "action_space": result["metadata"].get("action_space", {}),
        "observation_space": result["metadata"].get("observation_space", {}),
        "reward": result["metadata"].get("reward", ""),
        "files": result["files"],
        "codex_threads": result.get("codex_threads", {}),
        "last_training": base_record.get("last_training", {}) if base_record else {},
    }

    put_env_record(record)

    return ChatResponse(
        env_id=result["env_id"],
        name=result["name"],
        success=True,
        summary=result["summary"],
        files={name: f"in-memory://{result['env_id']}/{name}" for name in result["files"].keys()},
        validation=ValidationResult(**result["validation"]),
        saved=False,
    )
