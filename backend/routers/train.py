"""Training routes with configurable algorithms and hyperparameters."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from models.schemas import TrainRequest
from routers.envs import get_env_record, put_env_record
from routers.ws import ws_hub
from runtime.trainer import run_training

router = APIRouter(tags=["train"])

TRAIN_TASKS: dict[str, asyncio.Task] = {}


@router.post("/train/{env_id}")
async def start_training(env_id: str, request: TrainRequest) -> dict:
    """Start background training and stream progress over websocket."""

    record = get_env_record(env_id)
    if env_id in TRAIN_TASKS and not TRAIN_TASKS[env_id].done():
        raise HTTPException(status_code=409, detail="Training already in progress")

    env_code = record.get("files", {}).get("env.py")
    if not env_code:
        raise HTTPException(status_code=400, detail="env.py missing for environment")

    params = request.model_dump()

    async def publish(event: dict) -> None:
        await ws_hub.broadcast_train(event)
        if event.get("status") == "complete":
            payload = event.get("payload", {})
            record["last_training"] = {
                "algorithm": payload.get("algorithm", params["algorithm"]),
                "model_path": payload.get("model_path", ""),
                "timesteps": params["timesteps"],
            }
            put_env_record(record)

    task = asyncio.create_task(run_training(env_id, env_code, params, publish))
    TRAIN_TASKS[env_id] = task
    return {"status": "queued", "env_id": env_id, **params}
