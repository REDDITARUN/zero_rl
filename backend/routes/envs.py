"""REST endpoints for environment CRUD."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.config import ENVS_DIR

router = APIRouter()


def _safe_env_dir(env_id: str) -> Path:
    """Resolve env_dir and verify it's under ENVS_DIR (path traversal guard)."""
    env_dir = (ENVS_DIR / env_id).resolve()
    if not str(env_dir).startswith(str(ENVS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid env_id")
    return env_dir


def _read_config(env_dir: Path) -> dict:
    """Read config from JSON or extract top-level assignments from config.py."""
    config_json = env_dir / "config.json"
    if config_json.exists():
        try:
            return json.loads(config_json.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    config_py = env_dir / "config.py"
    if config_py.exists():
        try:
            source = config_py.read_text()
            tree = ast.parse(source)
            cfg: dict = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and isinstance(
                            node.value, ast.Constant
                        ):
                            cfg[target.id] = node.value.value
            if cfg:
                return cfg
        except Exception:
            pass

    return {}


@router.get("")
async def list_envs() -> list[dict]:
    """List all generated environments."""
    envs = []
    if not ENVS_DIR.exists():
        return envs

    for d in sorted(ENVS_DIR.iterdir()):
        if not d.is_dir():
            continue
        cfg = _read_config(d)
        envs.append(
            {
                "id": d.name,
                "name": cfg.get("env_name", d.name),
                "description": cfg.get("description", ""),
                "type": cfg.get("env_type", "gym"),
                "created_at": str(d.stat().st_mtime),
                "has_env": (d / "env.py").exists(),
            }
        )
    return envs


@router.get("/{env_id}")
async def get_env(env_id: str) -> dict:
    """Get details + source code of a specific environment."""
    env_dir = _safe_env_dir(env_id)
    if not env_dir.exists():
        raise HTTPException(status_code=404, detail="Environment not found")

    cfg = _read_config(env_dir)
    files: dict[str, str] = {}
    for f in env_dir.iterdir():
        if f.is_file() and f.suffix == ".py":
            try:
                files[f.name] = f.read_text()
            except OSError:
                files[f.name] = "# Error reading file"

    return {"id": env_id, "config": cfg, "files": files}


@router.delete("/{env_id}")
async def delete_env(env_id: str) -> dict:
    """Delete a generated environment."""
    env_dir = _safe_env_dir(env_id)
    if not env_dir.exists():
        raise HTTPException(status_code=404, detail="Environment not found")
    shutil.rmtree(env_dir)
    return {"deleted": env_id}
