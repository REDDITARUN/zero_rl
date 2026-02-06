"""Environment metadata, runtime control, and gallery persistence routes."""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse

from config import ENVS_DIR
from models.schemas import EnvActionRequest, EnvSummary, RuntimeState, SaveResponse
from runtime.env_session import env_runtime_manager
from utils.file_ops import ensure_env_dir

router = APIRouter(tags=["envs"])

ENV_CACHE: dict[str, dict[str, Any]] = {}
ENV_CLASS_PATTERN = re.compile(r"class\s+\w+Env\s*\(")


def _has_env_class(env_code: str) -> bool:
    return bool(ENV_CLASS_PATTERN.search(env_code))


def _reward_text(config: dict[str, Any]) -> str:
    direct = config.get("reward")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    components = config.get("reward_components")
    if isinstance(components, list):
        parts: list[str] = []
        for comp in components:
            if not isinstance(comp, dict):
                continue
            name = str(comp.get("name", "")).strip()
            value = comp.get("value")
            if name:
                parts.append(f"{name}: {value}")
        if parts:
            return "; ".join(parts)
    return ""


def _read_saved_record(env_id: str) -> dict[str, Any] | None:
    env_dir = ENVS_DIR / env_id
    if not env_dir.exists():
        return None

    manifest_path = env_dir / "manifest.json"
    config_path = env_dir / "config.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    files: dict[str, str] = {}
    for path in sorted(env_dir.glob("*")):
        if path.is_file() and path.name in {"env.py", "train.py", "README.md", "config.json"}:
            files[path.name] = path.read_text(encoding="utf-8")
    env_code = files.get("env.py", "")
    if not env_code or not _has_env_class(env_code):
        return None

    return {
        "env_id": env_id,
        "name": manifest.get("name", env_id),
        "prompt": manifest.get("prompt", ""),
        "created_at": manifest.get("created_at", datetime.utcnow().isoformat()),
        "updated_at": manifest.get("updated_at", manifest.get("created_at", datetime.utcnow().isoformat())),
        "saved": True,
        "action_space": config.get("action_space", {"type": "Discrete", "n": 4, "actions": ["up", "right", "down", "left"]}),
        "observation_space": config.get("observation_space", {"type": "Box", "shape": [6], "dtype": "float32", "description": []}),
        "reward": _reward_text(config),
        "files": files,
        "codex_threads": manifest.get("codex_threads", {}),
        "last_training": manifest.get("last_training", {}),
    }


def get_env_record(env_id: str) -> dict[str, Any]:
    """Load working record from memory cache or disk."""

    cached = ENV_CACHE.get(env_id)
    if cached is not None:
        env_code = str(cached.get("files", {}).get("env.py", ""))
        if env_code and _has_env_class(env_code):
            return cached
        ENV_CACHE.pop(env_id, None)

    saved = _read_saved_record(env_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="Environment not found")

    ENV_CACHE[env_id] = saved
    return saved


def put_env_record(record: dict[str, Any]) -> None:
    """Store environment working state in memory."""

    ENV_CACHE[record["env_id"]] = record


def save_env_record(env_id: str) -> dict[str, Any]:
    """Persist cached env record to disk and mark as saved."""

    record = get_env_record(env_id)
    env_dir = ensure_env_dir(env_id)

    files: dict[str, str] = record.get("files", {})
    for filename in ["env.py", "train.py", "README.md", "config.json"]:
        content = files.get(filename)
        if content is None:
            continue
        (env_dir / filename).write_text(content, encoding="utf-8")

    now = datetime.utcnow().isoformat()
    if not record.get("created_at"):
        record["created_at"] = now
    record["updated_at"] = now
    record["saved"] = True

    manifest = {
        "env_id": env_id,
        "name": record.get("name", env_id),
        "prompt": record.get("prompt", ""),
        "created_at": record.get("created_at", now),
        "updated_at": now,
        "codex_threads": record.get("codex_threads", {}),
        "last_training": record.get("last_training", {}),
    }
    (env_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    put_env_record(record)
    return record


def _action_labels(record: dict[str, Any]) -> list[str]:
    actions = record.get("action_space", {}).get("actions", [])
    return [str(a) for a in actions]


@router.get("/envs", response_model=list[EnvSummary])
async def get_envs() -> list[EnvSummary]:
    """List only saved environments for gallery tab."""

    rows: list[EnvSummary] = []
    seen: set[str] = set()

    for env_id, record in ENV_CACHE.items():
        if not record.get("saved"):
            continue
        created_at = datetime.fromisoformat(record["created_at"])
        rows.append(EnvSummary(env_id=env_id, name=record["name"], prompt=record.get("prompt", ""), created_at=created_at, saved=True))
        seen.add(env_id)

    for path in sorted(ENVS_DIR.glob("*")):
        if not path.is_dir() or path.name in seen:
            continue
        saved = _read_saved_record(path.name)
        if saved is None:
            continue
        created_at = datetime.fromisoformat(saved["created_at"])
        rows.append(EnvSummary(env_id=path.name, name=saved["name"], prompt=saved.get("prompt", ""), created_at=created_at, saved=True))

    return sorted(rows, key=lambda r: r.created_at, reverse=True)


@router.get("/envs/{env_id}")
async def get_env_metadata(env_id: str) -> dict[str, Any]:
    """Return working env metadata used across tabs."""

    record = get_env_record(env_id)
    return {
        "env_id": record["env_id"],
        "name": record["name"],
        "prompt": record.get("prompt", ""),
        "created_at": record["created_at"],
        "updated_at": record.get("updated_at", record["created_at"]),
        "saved": bool(record.get("saved", False)),
        "action_space": record.get("action_space", {}),
        "observation_space": record.get("observation_space", {}),
        "reward": record.get("reward", ""),
        "files": {name: f"in-memory://{env_id}/{name}" for name in record.get("files", {}).keys()},
        "last_training": record.get("last_training", {}),
    }


@router.get("/envs/{env_id}/files/{filename}", response_class=PlainTextResponse)
async def get_env_file(env_id: str, filename: str) -> str:
    """Read generated source files from working cache/disk."""

    record = get_env_record(env_id)
    files = record.get("files", {})
    if filename not in files:
        raise HTTPException(status_code=404, detail=f"Missing file: {filename}")
    return files[filename]


@router.post("/envs/{env_id}/save", response_model=SaveResponse)
async def save_env(env_id: str) -> SaveResponse:
    """Persist in-memory environment to disk/gallery."""

    record = save_env_record(env_id)
    return SaveResponse(env_id=env_id, saved=True, path=str(ENVS_DIR / env_id))


@router.post("/envs/{env_id}/reset", response_model=RuntimeState)
async def reset_env(env_id: str) -> RuntimeState:
    """Reset interactive runtime for selected environment."""

    record = get_env_record(env_id)
    env_code = record.get("files", {}).get("env.py")
    if not env_code:
        raise HTTPException(status_code=400, detail="env.py missing for environment")

    try:
        payload = env_runtime_manager.reset(env_id, env_code, _action_labels(record))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Runtime reset failed: {type(exc).__name__}: {exc}") from exc
    return RuntimeState(**payload)


@router.post("/envs/{env_id}/step", response_model=RuntimeState)
async def step_env(env_id: str, request: EnvActionRequest) -> RuntimeState:
    """Apply one action to interactive runtime and return updated state."""

    record = get_env_record(env_id)
    env_code = record.get("files", {}).get("env.py")
    if not env_code:
        raise HTTPException(status_code=400, detail="env.py missing for environment")

    labels = _action_labels(record)
    labels = env_runtime_manager.get_action_labels(env_id, env_code, labels)
    action_value = request.action
    if isinstance(action_value, str):
        if action_value not in labels:
            raise HTTPException(status_code=422, detail=f"Unknown action '{action_value}'")
        action_index = labels.index(action_value)
    else:
        action_index = int(action_value)

    try:
        payload = env_runtime_manager.step(env_id, env_code, labels, action_index)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Runtime step failed: {type(exc).__name__}: {exc}") from exc
    return RuntimeState(**payload)


@router.get("/envs/{env_id}/state", response_model=RuntimeState)
async def get_runtime_state(env_id: str) -> RuntimeState:
    """Return current interactive runtime state; reset if absent."""

    record = get_env_record(env_id)
    env_code = record.get("files", {}).get("env.py")
    if not env_code:
        raise HTTPException(status_code=400, detail="env.py missing for environment")

    try:
        payload = env_runtime_manager.get_state(env_id, env_code, _action_labels(record))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Runtime state failed: {type(exc).__name__}: {exc}") from exc
    return RuntimeState(**payload)


@router.get("/envs/{env_id}/download")
async def download_env(env_id: str) -> StreamingResponse:
    """Download current env working files as zip."""

    record = get_env_record(env_id)
    files = record.get("files", {})
    if not files:
        raise HTTPException(status_code=404, detail="No files available for download")

    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, content in files.items():
            archive.writestr(f"{env_id}/{filename}", content)

    blob.seek(0)
    return StreamingResponse(
        blob,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{env_id}.zip"'},
    )
