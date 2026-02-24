"""REST endpoints for browsing and managing generated assets."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from core.config import ASSETS_DIR

router = APIRouter()


def _safe_asset_dir(asset_id: str) -> Path:
    """Resolve asset_dir and verify it's under ASSETS_DIR."""
    asset_dir = (ASSETS_DIR / asset_id).resolve()
    if not str(asset_dir).startswith(str(ASSETS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid asset_id")
    return asset_dir


@router.get("")
async def list_assets() -> list[dict]:
    """List all generated assets with descriptions."""
    assets: list[dict] = []
    if not ASSETS_DIR.exists():
        return assets

    for d in sorted(ASSETS_DIR.iterdir()):
        if not d.is_dir():
            continue

        desc = ""
        desc_file = d / "description.txt"
        if desc_file.exists():
            try:
                desc = desc_file.read_text().strip()[:200]
            except OSError:
                pass

        files: list[str] = []
        file_types: list[str] = []
        for f in d.iterdir():
            if f.is_file() and f.suffix in (".stl", ".urdf", ".scad"):
                files.append(f.name)
                ext = f.suffix.lstrip(".")
                if ext not in file_types:
                    file_types.append(ext)

        params_file = d / "parameters.json"
        has_params = params_file.exists()

        assets.append(
            {
                "id": d.name,
                "name": d.name.replace("_", " ").replace("-", " "),
                "description": desc,
                "file_types": file_types,
                "files": files,
                "has_params": has_params,
            }
        )
    return assets


@router.get("/{asset_id}")
async def get_asset(asset_id: str) -> dict:
    """Get full details of an asset including file contents."""
    asset_dir = _safe_asset_dir(asset_id)
    if not asset_dir.exists():
        raise HTTPException(status_code=404, detail="Asset not found")

    desc = ""
    desc_file = asset_dir / "description.txt"
    if desc_file.exists():
        try:
            desc = desc_file.read_text().strip()
        except OSError:
            pass

    files: dict[str, str] = {}
    binary_files: list[str] = []
    for f in asset_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix in (".stl", ".urdf"):
            binary_files.append(f.name)
        if f.suffix in (".scad", ".urdf", ".xml", ".txt", ".json", ".py"):
            try:
                files[f.name] = f.read_text()
            except OSError:
                files[f.name] = "# Error reading file"

    params: list[dict] = []
    params_file = asset_dir / "parameters.json"
    if params_file.exists():
        try:
            params = json.loads(params_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "id": asset_id,
        "description": desc,
        "files": files,
        "binary_files": binary_files,
        "parameters": params,
    }


@router.get("/{asset_id}/file/{filename}")
async def get_asset_file(asset_id: str, filename: str) -> FileResponse:
    """Serve a binary asset file (e.g. STL)."""
    asset_dir = _safe_asset_dir(asset_id)
    filepath = (asset_dir / filename).resolve()
    if not str(filepath).startswith(str(asset_dir)):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    media = "application/octet-stream"
    if filepath.suffix == ".stl":
        media = "model/stl"
    elif filepath.suffix == ".urdf":
        media = "application/xml"

    return FileResponse(filepath, media_type=media, filename=filename)
