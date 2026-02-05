"""Code validation agent with fix-loop support."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from agents.base import AgentResult, BaseAgent


class ValidatorAgent(BaseAgent):
    """Validates generated env code; attempts lightweight runtime checks."""

    name = "validator"

    async def run(self, prompt: str, context: dict | None = None) -> AgentResult:
        context = context or {}
        env_code = context.get("env_code", "")

        errors: list[str] = []
        warnings: list[str] = []
        stage = "syntax"

        try:
            ast.parse(env_code)
        except SyntaxError as exc:
            errors.append(f"Syntax error line {exc.lineno}: {exc.msg}")
            return AgentResult(self.name, {"success": False, "stage": stage, "errors": errors, "warnings": warnings})

        stage = "import"
        tmp_path = None
        module = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tmp:
                tmp.write(env_code)
                tmp_path = Path(tmp.name)

            spec = importlib.util.spec_from_file_location("generated_env", tmp_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to build import spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules["generated_env"] = module
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Import error: {type(exc).__name__}: {exc}")
            return AgentResult(self.name, {"success": False, "stage": stage, "errors": errors, "warnings": warnings})
        finally:
            if tmp_path and tmp_path.exists():
                os.unlink(tmp_path)

        stage = "class_discovery"
        env_class = None
        for name, obj in vars(module).items():
            if isinstance(obj, type) and name.endswith("Env") and name != "Env":
                env_class = obj
                break

        if env_class is None:
            errors.append("No environment class ending with 'Env' found.")
            return AgentResult(self.name, {"success": False, "stage": stage, "errors": errors, "warnings": warnings})

        stage = "runtime"
        try:
            env = env_class(render_mode="rgb_array")
            obs, _ = env.reset()
            if hasattr(env, "action_space"):
                action = env.action_space.sample()
                env.step(action)
            frame = env.render()
            if frame is None:
                warnings.append("render() returned None for rgb_array mode")
            env.close()
            if obs is None:
                warnings.append("reset() returned None observation")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Runtime smoke test failed: {type(exc).__name__}: {exc}")
            return AgentResult(self.name, {"success": False, "stage": stage, "errors": errors, "warnings": warnings})

        return AgentResult(self.name, {"success": True, "stage": "complete", "errors": errors, "warnings": warnings})
