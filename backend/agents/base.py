"""Base agent abstractions for ZeroRL generation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentResult:
    """Result payload returned by each agent."""

    name: str
    payload: Any


class BaseAgent:
    """Common interface for skill-inspired generation agents."""

    name: str = "base"

    async def run(self, prompt: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Execute agent task and return structured result."""

        raise NotImplementedError
