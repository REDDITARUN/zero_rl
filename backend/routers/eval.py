"""Evaluation routes for trained policy rollouts."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from models.schemas import EvalRequest
from routers.envs import get_env_record
from routers.ws import ws_hub
from runtime.evaluator import run_evaluation

router = APIRouter(tags=["eval"])

EVAL_TASKS: dict[str, asyncio.Task] = {}


@router.post("/eval/{env_id}")
async def start_eval(env_id: str, request: EvalRequest) -> dict:
    """Start evaluation stream for the latest trained model."""

    record = get_env_record(env_id)
    training = record.get("last_training", {})
    model_path = training.get("model_path")
    algorithm = training.get("algorithm")
    if not model_path or not algorithm:
        raise HTTPException(status_code=400, detail="No trained model found. Run training first.")

    env_code = record.get("files", {}).get("env.py")
    if not env_code:
        raise HTTPException(status_code=400, detail="env.py missing for environment")

    if env_id in EVAL_TASKS and not EVAL_TASKS[env_id].done():
        raise HTTPException(status_code=409, detail="Evaluation already running")

    async def publish(event: dict) -> None:
        await ws_hub.broadcast_eval(event)

    task = asyncio.create_task(
        run_evaluation(
            env_id=env_id,
            env_code=env_code,
            model_path=model_path,
            algorithm=algorithm,
            episodes=request.episodes,
            max_steps=request.max_steps,
            callback=publish,
        )
    )
    EVAL_TASKS[env_id] = task

    return {
        "status": "queued",
        "env_id": env_id,
        "algorithm": algorithm,
        "episodes": request.episodes,
        "max_steps": request.max_steps,
    }
