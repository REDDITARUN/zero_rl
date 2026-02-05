"""FastAPI entrypoint for ZeroRL backend."""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import FRONTEND_ORIGIN, USE_CODEX_SDK
from routers import chat, envs, eval, render, train, ws

app = FastAPI(title="ZeroRL API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(envs.router, prefix="/api")
app.include_router(train.router, prefix="/api")
app.include_router(eval.router, prefix="/api")
app.include_router(render.router, prefix="/api")
app.include_router(ws.router)


@app.on_event("startup")
async def startup_checks() -> None:
    """Log common local setup issues early."""

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"Python {sys.version_info.major}.{sys.version_info.minor} detected. "
            "ZeroRL backend requires Python 3.11."
        )

    if USE_CODEX_SDK and not (os.getenv("OPENAI_API_KEY") or os.getenv("CODEX_API_KEY")):
        raise RuntimeError("Missing OPENAI_API_KEY/CODEX_API_KEY with USE_CODEX_SDK=true")


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Simple liveness endpoint."""

    return {"status": "ok", "service": "zerorl-backend"}
