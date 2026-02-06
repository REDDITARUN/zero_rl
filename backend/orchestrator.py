"""ZeroRL orchestration pipeline for parallel Codex agent execution."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from textwrap import dedent
from typing import Any, Awaitable, Callable, Dict
from uuid import uuid4

from agents.validator import ValidatorAgent
from codex_client import CodexSDKClient, CodexSDKError
from config import (
    CODEX_MAX_RETRIES,
    CODEX_MODEL,
    CODEX_MODEL_ARCHITECT,
    CODEX_MODEL_DOCS,
    CODEX_MODEL_REWARDS,
    CODEX_MODEL_SPACES,
    CODEX_MODEL_TRAINER,
    CODEX_MODEL_VALIDATOR,
    MAX_FIX_ATTEMPTS,
)


StatusCallback = Callable[[dict[str, Any]], Awaitable[None]]
MAX_UI_ACTIONS = 32
DEFAULT_MODEL_FALLBACK = "gpt-5-codex"


class AgentStatus(str, Enum):
    """State machine for each agent in the dashboard."""

    IDLE = "idle"
    WORKING = "working"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentTask:
    """Runtime metadata for an orchestrated agent."""

    agent_id: str
    name: str
    skill: str
    status: AgentStatus = AgentStatus.IDLE
    message: str = ""


class Orchestrator:
    """Coordinates generation, validation, and documentation in phases."""

    def __init__(self, status_callback: StatusCallback | None = None) -> None:
        self.status_callback = status_callback
        self.agents: Dict[str, AgentTask] = {
            "architect": AgentTask("architect", "Env Architect", "env-architect"),
            "rewards": AgentTask("rewards", "Reward Engineer", "reward-engineer"),
            "spaces": AgentTask("spaces", "Space Designer", "space-designer"),
            "validator": AgentTask("validator", "Code Validator", "code-validator"),
            "docs": AgentTask("docs", "Docs Generator", "docs-generator"),
            "trainer": AgentTask("trainer", "Trainer Config", "trainer-config"),
        }
        self.validator = ValidatorAgent()
        self.codex_client = CodexSDKClient()

    async def generate_environment(
        self,
        user_prompt: str,
        env_id: str | None = None,
        base_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run complete workflow and return generated artifact payload."""

        env_id = env_id or str(uuid4())
        env_name = self._derive_env_name(user_prompt)
        thread_map: dict[str, str] = dict((base_record or {}).get("codex_threads", {}))

        context: dict[str, Any] = {
            "env_id": env_id,
            "env_name": base_record.get("name", env_name) if base_record else env_name,
            "base_env_code": (base_record or {}).get("files", {}).get("env.py", ""),
            "base_train_code": (base_record or {}).get("files", {}).get("train.py", ""),
            "base_readme": (base_record or {}).get("files", {}).get("README.md", ""),
            "base_config": (base_record or {}).get("files", {}).get("config.json", ""),
        }

        # Phase 1: architect first (largest token budget), then rewards/spaces in parallel.
        # This is more reliable than launching three heavy generations simultaneously.
        await self._update_status("architect", AgentStatus.WORKING, "Generating environment architecture")
        arch_result = await self._run_architect(user_prompt, context, thread_map.get("architect"))
        context.update(arch_result[0])
        thread_map["architect"] = arch_result[1] or thread_map.get("architect", "")
        await self._update_status("architect", AgentStatus.COMPLETE, "Architecture complete")

        await self._update_status("rewards", AgentStatus.WORKING, "Designing reward shaping")
        await self._update_status("spaces", AgentStatus.WORKING, "Designing action/observation spaces")
        reward_result, space_result = await asyncio.gather(
            self._run_rewards(user_prompt, context, thread_map.get("rewards")),
            self._run_spaces(user_prompt, context, thread_map.get("spaces")),
        )

        context.update(reward_result[0])
        context.update(space_result[0])
        self._normalize_space_metadata(context)
        thread_map["rewards"] = reward_result[1] or thread_map.get("rewards", "")
        thread_map["spaces"] = space_result[1] or thread_map.get("spaces", "")
        await self._update_status("rewards", AgentStatus.COMPLETE, "Rewards complete")
        await self._update_status("spaces", AgentStatus.COMPLETE, "Spaces complete")

        context["env_code"] = self._merge_outputs(context)

        # Phase 2: validation + codex fix loop
        validation_payload, validator_thread = await self._validate_with_retries(
            user_prompt,
            context,
            thread_map.get("validator"),
        )
        if validator_thread:
            thread_map["validator"] = validator_thread

        if not validation_payload["success"]:
            return {
                "env_id": env_id,
                "name": context.get("env_name", env_name),
                "success": False,
                "summary": "Validation failed",
                "files": {},
                "validation": validation_payload,
                "metadata": {
                    "prompt": user_prompt,
                    "created_at": datetime.utcnow().isoformat(),
                    "action_space": context.get("action_space", {}),
                    "observation_space": context.get("observation_space", {}),
                    "reward": context.get("description", ""),
                },
                "codex_threads": thread_map,
            }

        # Phase 3: docs + training script
        await self._update_status("docs", AgentStatus.WORKING, "Generating docs")
        await self._update_status("trainer", AgentStatus.WORKING, "Generating training script")

        docs_result, trainer_result = await asyncio.gather(
            self._run_docs(user_prompt, context, thread_map.get("docs")),
            self._run_trainer(user_prompt, context, thread_map.get("trainer")),
        )

        context.update({"readme": docs_result[0]["readme"], "config": docs_result[0]["config"]})
        context.update({"train_code": trainer_result[0]["train_code"]})
        thread_map["docs"] = docs_result[1] or thread_map.get("docs", "")
        thread_map["trainer"] = trainer_result[1] or thread_map.get("trainer", "")

        await self._update_status("docs", AgentStatus.COMPLETE, "Docs complete")
        await self._update_status("trainer", AgentStatus.COMPLETE, "Training script complete")

        files = {
            "env.py": context["env_code"],
            "train.py": context["train_code"],
            "README.md": context["readme"],
            "config.json": context["config"],
        }

        return {
            "env_id": env_id,
            "name": context.get("env_name", env_name),
            "success": True,
            "summary": "Environment generated with Codex and validated",
            "files": files,
            "validation": validation_payload,
            "metadata": {
                "prompt": user_prompt,
                "created_at": datetime.utcnow().isoformat(),
                "action_space": context.get("action_space", {}),
                "observation_space": context.get("observation_space", {}),
                "reward": context.get("description", ""),
            },
            "codex_threads": {k: v for k, v in thread_map.items() if v},
        }

    async def _validate_with_retries(
        self,
        user_prompt: str,
        context: dict[str, Any],
        thread_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        await self._update_status("validator", AgentStatus.WORKING, "Running validation pipeline")

        payload: dict[str, Any] = {"success": False, "stage": "unknown", "errors": ["Validation did not run"], "warnings": []}
        latest_thread = thread_id

        for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
            validation = await self.validator.run(user_prompt, context)
            payload = validation.payload
            if payload["success"]:
                await self._update_status("validator", AgentStatus.COMPLETE, f"Validation passed on attempt {attempt}")
                return payload, latest_thread

            fix_payload, latest_thread = await self._run_validator_fix(
                user_prompt,
                context,
                payload,
                latest_thread,
            )
            context["env_code"] = fix_payload["env_code"]
            await self._update_status(
                "validator",
                AgentStatus.WORKING,
                f"Validation failed at {payload['stage']}; Codex fix attempt {attempt}/{MAX_FIX_ATTEMPTS}",
            )

        await self._update_status("validator", AgentStatus.ERROR, "Validation failed after retries")
        return payload, latest_thread

    async def _run_architect(
        self,
        user_prompt: str,
        context: dict[str, Any],
        thread_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        schema = {
            "type": "object",
            "properties": {
                "env_name": {"type": "string", "pattern": "^[A-Z][A-Za-z0-9]+$"},
                "env_code": {"type": "string"},
            },
            "required": ["env_name", "env_code"],
            "additionalProperties": False,
        }
        prompt = dedent(
            f"""
            You are the env-architect skill for ZeroRL.
            Build a complete Gymnasium + Pygame environment from the user request.

            User request:
            {user_prompt}

            Existing env.py (if editing):
            ```python
            {context.get('base_env_code','')}
            ```

            Constraints:
            - class name must end with Env
            - environment must directly reflect the user task (entities, goal, transitions)
            - do not return generic template code; include concrete mechanics for this prompt
            - include metadata with human/rgb_array
            - implement __init__, reset, step, render, close
            - return deterministic, testable behavior
            - ensure observation fits observation_space
            - use earthy render palette: background (245,240,230), agent (65,92,87), goal (103,138,99)
            - pygame rendering must remain stable: prefer basic shapes + optional text labels
            - emojis are optional via pygame font text rendering; always fallback to plain shapes if font missing
            - include all named entities from prompt in both logic and render (e.g., squirrel/acorns/hunter/pitfalls)
            - expose self.action_labels list and ensure action_space.n == len(self.action_labels)
            - step(action) must map actions using self.action_labels order
            - keep env.py concise (< 260 lines)
            - return only one environment class and required helper methods

            Return JSON only.
            """
        ).strip()
        return await self._run_codex_agent(
            agent_id="architect",
            prompt=prompt,
            output_schema=schema,
            thread_id=thread_id,
        )

    async def _run_rewards(
        self,
        user_prompt: str,
        context: dict[str, Any],
        thread_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        schema = {
            "type": "object",
            "properties": {
                "reward_mode": {"type": "string", "enum": ["dense", "sparse"]},
                "description": {"type": "string"},
            },
            "required": ["reward_mode", "description"],
            "additionalProperties": False,
        }
        prompt = dedent(
            f"""
            You are the reward-engineer skill for ZeroRL.
            Decide and describe the reward strategy.

            User request:
            {user_prompt}

            Existing reward description:
            {context.get('reward', '')}

            Return JSON only.
            """
        ).strip()
        return await self._run_codex_agent(
            agent_id="rewards",
            prompt=prompt,
            output_schema=schema,
            thread_id=thread_id,
        )

    async def _run_spaces(
        self,
        user_prompt: str,
        context: dict[str, Any],
        thread_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        schema = {
            "type": "object",
            "properties": {
                "action_space": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "n": {"type": "integer"},
                        "actions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["type", "n", "actions"],
                    "additionalProperties": False,
                },
                "observation_space": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "shape": {"type": "array", "items": {"type": "integer"}},
                        "dtype": {"type": "string"},
                        "description": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["type", "shape", "dtype", "description"],
                    "additionalProperties": False,
                },
            },
            "required": ["action_space", "observation_space"],
            "additionalProperties": False,
        }
        prompt = dedent(
            f"""
            You are the space-designer skill for ZeroRL.
            Define action and observation spaces for the requested task.

            User request:
            {user_prompt}

            Environment code to align with:
            ```python
            {context.get('env_code','')}
            ```

            Constraints:
            - keep action space compact and human-operable for UI controls
            - avoid combinatorial explosion for multi-agent tasks
            - unless explicitly requested otherwise, keep actions <= 12
            - action labels must be unique, concise, and deterministic
            - action_space.n must equal len(action_space.actions)

            Return JSON only.
            """
        ).strip()
        return await self._run_codex_agent(
            agent_id="spaces",
            prompt=prompt,
            output_schema=schema,
            thread_id=thread_id,
        )

    async def _run_docs(
        self,
        user_prompt: str,
        context: dict[str, Any],
        thread_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        schema = {
            "type": "object",
            "properties": {
                "readme": {"type": "string"},
                "config_json": {"type": "string"},
            },
            "required": ["readme", "config_json"],
            "additionalProperties": False,
        }
        prompt = dedent(
            f"""
            You are the docs-generator skill for ZeroRL.
            Create README.md and config.json text for the generated env.

            User request:
            {user_prompt}

            Environment name: {context.get('env_name')}
            Reward description: {context.get('description')}
            Action space: {json.dumps(context.get('action_space', {}))}
            Observation space: {json.dumps(context.get('observation_space', {}))}

            Return JSON only, with:
            - readme: full README markdown
            - config_json: JSON string for config.json (object serialized as a string)
            - keep README concise (< 120 lines)
            """
        ).strip()
        payload, next_thread = await self._run_codex_agent(
            agent_id="docs",
            prompt=prompt,
            output_schema=schema,
            thread_id=thread_id,
        )

        raw_config = str(payload.get("config_json", "{}")).strip()
        try:
            config_obj = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            snippet = _extract_first_json_object(raw_config)
            if snippet is None:
                raise CodexSDKError("Docs agent returned invalid config_json") from exc
            try:
                config_obj = json.loads(snippet)
            except json.JSONDecodeError as second_exc:
                raise CodexSDKError("Docs agent returned malformed config_json") from second_exc
        if not isinstance(config_obj, dict):
            raise CodexSDKError("Docs agent returned non-object config_json")
        payload["config"] = json.dumps(config_obj, indent=2)
        return payload, next_thread

    async def _run_trainer(
        self,
        user_prompt: str,
        context: dict[str, Any],
        thread_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        schema = {
            "type": "object",
            "properties": {
                "train_code": {"type": "string"},
            },
            "required": ["train_code"],
            "additionalProperties": False,
        }
        prompt = dedent(
            f"""
            You are the trainer-config skill for ZeroRL.
            Generate robust train.py for PPO/DQN/A2C support with JSON logging.

            User request:
            {user_prompt}

            Environment name: {context.get('env_name')}
            Existing train.py:
            ```python
            {context.get('base_train_code','')}
            ```

            Requirements:
            - argparse for algorithm, timesteps, learning_rate, gamma, epsilon, n_steps, batch_size
            - JSON progress lines with episode/reward/timesteps
            - save model file based on algorithm
            - keep script concise (< 220 lines)

            Return JSON only.
            """
        ).strip()
        return await self._run_codex_agent(
            agent_id="trainer",
            prompt=prompt,
            output_schema=schema,
            thread_id=thread_id,
        )

    async def _run_validator_fix(
        self,
        user_prompt: str,
        context: dict[str, Any],
        validation_payload: dict[str, Any],
        thread_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        schema = {
            "type": "object",
            "properties": {
                "env_code": {"type": "string"},
            },
            "required": ["env_code"],
            "additionalProperties": False,
        }
        prompt = dedent(
            f"""
            You are code-validator fix loop for ZeroRL.
            Fix env.py based strictly on validation payload.

            User request:
            {user_prompt}

            Validation payload:
            {json.dumps(validation_payload, indent=2)}

            Current env.py:
            ```python
            {context['env_code']}
            ```

            Return JSON only.
            """
        ).strip()
        return await self._run_codex_agent(
            agent_id="validator",
            prompt=prompt,
            output_schema=schema,
            thread_id=thread_id,
        )

    async def _update_status(self, agent_id: str, status: AgentStatus, message: str) -> None:
        self.agents[agent_id].status = status
        self.agents[agent_id].message = message
        if self.status_callback:
            await self.status_callback(
                {
                    "agent_id": agent_id,
                    "status": status.value,
                    "message": message,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )

    async def _on_codex_progress(self, agent_id: str, event: dict[str, Any]) -> None:
        message = str(event.get("message", "")).strip()
        if not message:
            return
        if self.agents[agent_id].status == AgentStatus.WORKING and self.agents[agent_id].message == message:
            return
        await self._update_status(agent_id, AgentStatus.WORKING, message[:220])

    async def _run_codex_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        output_schema: dict[str, Any],
        thread_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        last_error: CodexSDKError | None = None
        bridge_attempt = 0
        current_thread = thread_id
        model = self._model_for_agent(agent_id)

        while True:
            try:
                return await self.codex_client.run_json(
                    agent_id=agent_id,
                    prompt=prompt,
                    output_schema=output_schema,
                    thread_id=current_thread,
                    progress_callback=lambda event: self._on_codex_progress(agent_id, event),
                    model=model,
                )
            except CodexSDKError as exc:
                last_error = exc
                error_text = str(exc).lower()
                incompatible_thread = (
                    "recorded with model" in error_text and "resuming with" in error_text
                ) or ("session was recorded with model" in error_text)
                if incompatible_thread and current_thread:
                    current_thread = None
                    bridge_attempt = 0
                    await self._update_status(
                        agent_id,
                        AgentStatus.WORKING,
                        "Thread model mismatch; starting a fresh thread",
                    )
                    continue

                model_unavailable = (
                    "model_not_found" in error_text
                    or "does not exist" in error_text
                    or "invalid model" in error_text
                )
                if model_unavailable:
                    fallback = CODEX_MODEL if model != CODEX_MODEL else DEFAULT_MODEL_FALLBACK
                    if fallback and fallback != model:
                        previous = model
                        model = fallback
                        current_thread = None
                        bridge_attempt = 0
                        await self._update_status(
                            agent_id,
                            AgentStatus.WORKING,
                            f"Model '{previous}' unavailable; retrying with '{fallback}'",
                        )
                        continue

                retryable = (
                    "timed out" in error_text
                    or "invalid bridge json envelope" in error_text
                    or "codex bridge failed" in error_text
                    or "stream interrupted" in error_text
                )
                bridge_attempt += 1
                if retryable and bridge_attempt < max(1, CODEX_MAX_RETRIES):
                    await self._update_status(
                        agent_id,
                        AgentStatus.WORKING,
                        f"Bridge issue on attempt {bridge_attempt}/{CODEX_MAX_RETRIES}; retrying",
                    )
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise CodexSDKError(f"Unknown Codex error for agent '{agent_id}'")

    def _model_for_agent(self, agent_id: str) -> str:
        mapping = {
            "architect": CODEX_MODEL_ARCHITECT,
            "rewards": CODEX_MODEL_REWARDS,
            "spaces": CODEX_MODEL_SPACES,
            "validator": CODEX_MODEL_VALIDATOR,
            "docs": CODEX_MODEL_DOCS,
            "trainer": CODEX_MODEL_TRAINER,
        }
        return mapping.get(agent_id, CODEX_MODEL) or CODEX_MODEL

    def _derive_env_name(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "maze" in lowered:
            return "ZeroMaze"
        if "collect" in lowered:
            return "ZeroCollector"
        if "chase" in lowered or "pursuit" in lowered:
            return "ZeroChase"
        return "ZeroGrid"

    def _normalize_space_metadata(self, context: dict[str, Any]) -> None:
        raw_space = context.get("action_space")
        actions: list[str] = []
        if isinstance(raw_space, dict):
            seen: set[str] = set()
            for value in raw_space.get("actions", []):
                label = str(value).strip()
                if not label or label in seen:
                    continue
                seen.add(label)
                actions.append(label)

        if not actions:
            actions = ["up", "right", "down", "left"]
        if len(actions) > MAX_UI_ACTIONS:
            actions = actions[:MAX_UI_ACTIONS]

        space_type = "Discrete"
        if isinstance(raw_space, dict):
            raw_type = str(raw_space.get("type", "")).strip()
            if raw_type:
                space_type = raw_type

        context["action_space"] = {
            "type": space_type,
            "n": len(actions),
            "actions": actions,
        }

        raw_obs = context.get("observation_space")
        if not isinstance(raw_obs, dict):
            return
        shape = raw_obs.get("shape", [])
        normalized_shape = [int(v) for v in shape if isinstance(v, int) and v > 0]
        if not normalized_shape:
            normalized_shape = [6]
        description_raw = raw_obs.get("description", [])
        if not isinstance(description_raw, list):
            description_raw = []
        description = [str(item).strip() for item in description_raw if str(item).strip()][:16]
        context["observation_space"] = {
            "type": str(raw_obs.get("type", "Box")),
            "shape": normalized_shape,
            "dtype": str(raw_obs.get("dtype", "float32")),
            "description": description,
        }

    def _merge_outputs(self, context: dict[str, Any]) -> str:
        """Inject reward mode adjustment into generated env code when needed."""

        code = context["env_code"]
        if context.get("reward_mode") == "sparse":
            dense_block = """        reward = -0.01\n        reward += (self.prev_distance - current_distance) * 0.05\n        if reached_goal:\n            reward += 10.0\n        self.prev_distance = current_distance\n        return reward\n"""
            sparse_block = """        reward = 10.0 if reached_goal else -0.01\n        self.prev_distance = current_distance\n        return reward\n"""
            code = code.replace(dense_block, sparse_block)
        return code


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None
