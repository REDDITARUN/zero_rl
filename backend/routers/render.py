"""Rendering route for RGB frame previews."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from routers.envs import get_env_record
from runtime.renderer import render_frame_base64_from_code

router = APIRouter(tags=["render"])


@router.get("/render/{env_id}")
async def get_frame(env_id: str) -> dict:
    """Return a static render frame from current env code."""

    record = get_env_record(env_id)
    env_code = record.get("files", {}).get("env.py")
    if not env_code:
        raise HTTPException(status_code=400, detail="env.py missing for environment")

    try:
        frame = render_frame_base64_from_code(env_id, env_code)
        return {"env_id": env_id, "frame": frame, "mime": "image/png"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Render failed: {type(exc).__name__}: {exc}") from exc
