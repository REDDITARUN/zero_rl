"""Pydantic request/response schemas for ZeroRL."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for environment generation chat endpoint."""

    prompt: str = Field(min_length=3)
    env_id: Optional[str] = None
    run_id: Optional[str] = None


class AgentStatusEvent(BaseModel):
    """Represents status update for a single agent."""

    agent_id: str
    status: Literal["idle", "working", "complete", "error"]
    message: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ValidationResult(BaseModel):
    """Validation result payload."""

    success: bool
    stage: str
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response body for chat environment generation."""

    env_id: str
    name: str
    success: bool
    summary: str
    files: Dict[str, str]
    validation: ValidationResult
    saved: bool = False
    run_id: Optional[str] = None


class EnvSummary(BaseModel):
    """A lightweight representation for gallery/listing."""

    env_id: str
    name: str
    prompt: str
    created_at: datetime
    saved: bool = True


class TrainRequest(BaseModel):
    """Training trigger request."""

    algorithm: Literal["PPO", "DQN", "A2C"] = "PPO"
    timesteps: int = Field(default=5000, ge=100, le=500000)
    learning_rate: float = Field(default=3e-4, gt=0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    batch_size: int = Field(default=64, ge=8, le=4096)
    n_steps: int = Field(default=512, ge=16, le=8192)
    epsilon: float = Field(default=0.05, ge=0.0, le=1.0)


class EvalRequest(BaseModel):
    """Evaluation trigger request."""

    episodes: int = Field(default=1, ge=1, le=20)
    max_steps: int = Field(default=250, ge=10, le=2000)


class EnvActionRequest(BaseModel):
    """Agent action request payload."""

    action: int | str


class RuntimeState(BaseModel):
    """Serialized runtime state for env control panel."""

    env_id: str
    step: int
    last_action: Optional[str] = None
    last_reward: float = 0.0
    cumulative_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    observation: Any = None
    info: Dict[str, Any] = Field(default_factory=dict)
    frame: str
    available_actions: List[str] = Field(default_factory=list)


class SaveResponse(BaseModel):
    """Response after persisting an environment to gallery."""

    env_id: str
    saved: bool
    path: str


class TrainProgressEvent(BaseModel):
    """Training progress event sent to websocket clients."""

    env_id: str
    status: Literal["queued", "running", "complete", "error"]
    timesteps: int = 0
    reward: float = 0.0
    episode: int = 0
    avg_reward_100: float = 0.0
    message: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
