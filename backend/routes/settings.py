"""Settings endpoints — LLM provider, model, API key management."""

from __future__ import annotations

import os

from fastapi import APIRouter
from pydantic import BaseModel

from core.config import DEFAULT_MODEL, DEFAULT_PROVIDER, _KEY_ENV_VARS

router = APIRouter()


class SettingsPayload(BaseModel):
    """Settings update request body."""

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None


PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "o1-mini",
    ],
    "google-genai": [
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.0-flash",
    ],
}


@router.get("")
async def get_settings() -> dict:
    """Return current LLM settings (redact API key)."""
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    key_var = _KEY_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
    has_key = bool(os.getenv(key_var, ""))

    return {
        "provider": provider,
        "model": model,
        "has_api_key": has_key,
        "providers": PROVIDER_MODELS,
    }


@router.put("")
async def update_settings(payload: SettingsPayload) -> dict:
    """Update LLM settings at runtime (env vars, not persisted to disk)."""
    if payload.provider:
        os.environ["LLM_PROVIDER"] = payload.provider
    if payload.model:
        os.environ["LLM_MODEL"] = payload.model
    if payload.api_key:
        provider_for_key = payload.provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
        key_var = _KEY_ENV_VARS.get(
            provider_for_key, f"{provider_for_key.upper()}_API_KEY"
        )
        os.environ[key_var] = payload.api_key

    return await get_settings()
