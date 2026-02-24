"""Orchestrator — ReAct agent with deterministic guardrails.

The core loop: LLM reasons → calls tools → observes → repeats.
Guardrails enforce:
  - Auto-validation after file_write to env.py
  - Fix loop counter (max 3 attempts)
  - Escalation to eval_agent on persistent failures
"""

from __future__ import annotations

import base64
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from core.agents.eval_agent import run_eval_agent
from core.config import ENVS_DIR, MAX_FIX_ATTEMPTS, get_chat_model
from core.prompts.orchestrator import build_system_prompt
from core.tools.cad_generate import cad_generate
from core.tools.coding import (
    ALL_CODING_TOOLS,
    code_search,
    dir_list,
    file_edit,
    file_read,
    file_write,
    shell,
)
from core.tools.doc_tool import doc_lookup
from core.tools.eval_tool import eval_env
from core.tools.urdf_generate import urdf_generate

ALL_TOOLS = [
    *ALL_CODING_TOOLS,
    eval_env,
    cad_generate,
    urdf_generate,
    doc_lookup,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


@dataclass
class AgentEvent:
    """Event emitted by the orchestrator for the UI."""

    type: str  # "tool_start", "tool_end", "message", "error", "validation"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorState:
    """Mutable state for a single orchestrator run."""

    env_id: str = ""
    fix_attempts: int = 0
    validation_passed: bool = False
    events: list[AgentEvent] = field(default_factory=list)


class Orchestrator:
    """ReAct orchestrator with guardrail wrapper."""

    def __init__(
        self,
        on_event: Callable[[AgentEvent], None] | None = None,
        max_iterations: int = 30,
    ) -> None:
        self.max_iterations = max_iterations
        self.on_event = on_event or (lambda e: None)
        self.messages: list[BaseMessage] = []

    def _emit(self, event: AgentEvent) -> None:
        self.on_event(event)

    def _sanitize_messages(self) -> None:
        """Ensure every tool_use in an AIMessage has a matching ToolMessage.

        Anthropic requires that every tool_use block is immediately followed
        by a ToolMessage with the same tool_call_id. If we crashed mid-loop,
        orphaned tool_use blocks will cause 400 errors on subsequent calls.
        """
        if not self.messages:
            return

        tool_result_ids: set[str] = set()
        for msg in self.messages:
            if isinstance(msg, ToolMessage):
                tool_result_ids.add(msg.tool_call_id)

        cleaned: list[BaseMessage] = []
        for msg in self.messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                orphaned = [
                    tc for tc in msg.tool_calls
                    if tc["id"] not in tool_result_ids
                ]
                if orphaned:
                    text_content = ""
                    if isinstance(msg.content, str):
                        text_content = msg.content
                    elif isinstance(msg.content, list):
                        for part in msg.content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_content += part.get("text", "")

                    matched = [
                        tc for tc in msg.tool_calls
                        if tc["id"] in tool_result_ids
                    ]

                    if matched:
                        new_msg = AIMessage(
                            content=text_content or "",
                            tool_calls=matched,
                        )
                        cleaned.append(new_msg)
                    elif text_content:
                        cleaned.append(AIMessage(content=text_content))
                    continue
            cleaned.append(msg)

        self.messages = cleaned

    def clear_history(self) -> None:
        """Reset conversation history, keeping only the system prompt."""
        system_msgs = [m for m in self.messages if isinstance(m, SystemMessage)]
        self.messages = system_msgs

    def run(
        self,
        user_message: str,
        image_paths: list[str] | None = None,
    ) -> str:
        """Run the orchestrator on a user request.

        Args:
            user_message: The user's natural language prompt.
            image_paths: Optional paths to attached images.

        Returns:
            The agent's final text response.
        """
        system_prompt = build_system_prompt()
        if not self.messages:
            self.messages.append(SystemMessage(content=system_prompt))

        human_content = self._build_human_content(user_message, image_paths)
        self.messages.append(HumanMessage(content=human_content))

        llm = get_chat_model().bind_tools(ALL_TOOLS)
        state = OrchestratorState()

        for iteration in range(self.max_iterations):
            self._sanitize_messages()

            try:
                response = llm.invoke(self.messages)
            except Exception as e:
                err_str = str(e)
                if "tool_use" in err_str and "tool_result" in err_str:
                    self._sanitize_messages()
                    try:
                        response = llm.invoke(self.messages)
                    except Exception:
                        raise
                else:
                    raise

            self.messages.append(response)

            if not response.tool_calls:
                return response.content

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                self._emit(
                    AgentEvent(
                        type="tool_start",
                        data={"tool": tool_name, "args": tool_args},
                    )
                )

                result = self._execute_tool(tc, state)

                self._emit(
                    AgentEvent(
                        type="tool_end",
                        data={"tool": tool_name, "result": result[:500]},
                    )
                )

                self.messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

                guardrail_msgs = self._apply_guardrails(tc, result, state)
                self.messages.extend(guardrail_msgs)

        return "Orchestrator reached maximum iterations."

    def _build_human_content(
        self, text: str, images: list[str] | None
    ) -> list[dict] | str:
        """Build multimodal message content.

        Each entry in *images* can be:
          - A data-URL  (``data:image/...;base64,...``)  — saved to temp file.
          - A local file path — read and converted to a data-URL.

        Saved paths are stored in ``self._user_image_paths`` so the LLM
        can forward them to tools like ``cad_generate`` or ``urdf_generate``.
        """
        if not images:
            return text

        saved_paths: list[str] = []
        content: list[dict] = []

        for img in images:
            url, path = self._resolve_image(img)
            if url:
                content.append(
                    {"type": "image_url", "image_url": {"url": url}}
                )
            if path:
                saved_paths.append(path)

        self._user_image_paths = saved_paths

        hint = text
        if saved_paths:
            paths_str = ", ".join(saved_paths)
            hint += (
                f"\n\n[Reference images saved at: {paths_str}. "
                "Pass a path to the image_path parameter of cad_generate "
                "or urdf_generate if the user's request involves generating "
                "a 3D model or robot based on this image.]"
            )

        content.insert(0, {"type": "text", "text": hint})
        return content if len(content) > 1 else text

    @staticmethod
    def _resolve_image(img: str) -> tuple[str | None, str | None]:
        """Turn a data-URL or file path into (data_url, local_path).

        Data-URLs are also persisted to a temp file so tools can read them.
        """
        if img.startswith("data:"):
            m = re.match(r"data:image/(\w+);base64,(.*)", img, re.DOTALL)
            ext = m.group(1) if m else "png"
            if ext == "jpeg":
                ext = "jpg"
            tmp = tempfile.NamedTemporaryFile(
                suffix=f".{ext}", prefix="ref_img_", delete=False
            )
            raw = base64.b64decode(m.group(2)) if m else b""
            tmp.write(raw)
            tmp.close()
            return img, tmp.name

        p = Path(img)
        if p.exists():
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime = mime_map.get(p.suffix.lower(), "image/jpeg")
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f"data:{mime};base64,{b64}", str(p)

        return None, None

    def _execute_tool(self, tool_call: dict, state: OrchestratorState) -> str:
        """Execute a single tool call."""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        tool_fn = TOOL_MAP.get(tool_name)
        if tool_fn is None:
            return f"Unknown tool: {tool_name}"

        try:
            result = tool_fn.invoke(tool_args)

            if tool_name == "file_write":
                self._detect_env_id(tool_args.get("path", ""), state)

            return str(result)
        except Exception as e:
            return f"Tool error: {e}"

    def _detect_env_id(self, path: str, state: OrchestratorState) -> None:
        """Extract env_id when writing to envs/{id}/."""
        m = re.search(r"envs/([^/]+)/", path)
        if m:
            state.env_id = m.group(1)

    def _sibling_files_ready(self, state: OrchestratorState) -> bool:
        """Check that at least config.py exists alongside env.py."""
        if not state.env_id:
            return False
        env_dir = ENVS_DIR / state.env_id
        return (env_dir / "config.py").exists()

    def _apply_guardrails(
        self,
        tool_call: dict,
        result: str,
        state: OrchestratorState,
    ) -> list[BaseMessage]:
        """Apply deterministic guardrails after tool execution."""
        extra: list[BaseMessage] = []
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "file_write" and "env.py" in tool_args.get("path", ""):
            if not state.env_id:
                return extra

            if state.fix_attempts > MAX_FIX_ATTEMPTS:
                extra.append(
                    HumanMessage(
                        content=(
                            "[GUARDRAIL] Maximum fix attempts exceeded. "
                            "Stop trying to write env.py and explain to the "
                            "user what went wrong."
                        )
                    )
                )
                return extra

            if not self._sibling_files_ready(state):
                extra.append(
                    HumanMessage(
                        content=(
                            "[GUARDRAIL] env.py written but config.py not found yet. "
                            "Write all sibling files (config.py, renderer.py) before "
                            "validation will run automatically."
                        )
                    )
                )
                return extra

            self._emit(
                AgentEvent(
                    type="validation",
                    data={"env_id": state.env_id, "action": "auto_validate"},
                )
            )
            val_result = eval_env.invoke({"env_id": state.env_id})

            if "FAIL" in val_result:
                state.fix_attempts += 1
                self._emit(
                    AgentEvent(
                        type="error",
                        data={
                            "message": f"Validation failed (attempt {state.fix_attempts}/{MAX_FIX_ATTEMPTS})",
                            "details": val_result,
                        },
                    )
                )

                if state.fix_attempts >= MAX_FIX_ATTEMPTS:
                    diagnosis = self._escalate_to_eval_agent(
                        state.env_id, val_result
                    )
                    extra.append(
                        HumanMessage(
                            content=(
                                f"[GUARDRAIL] Validation failed {MAX_FIX_ATTEMPTS} times. "
                                f"Eval agent diagnosis:\n\n{diagnosis}\n\n"
                                "This is the FINAL attempt. Apply the fix or explain "
                                "to the user what went wrong. Do NOT write env.py again "
                                "unless you are confident the fix is correct."
                            )
                        )
                    )
                else:
                    extra.append(
                        HumanMessage(
                            content=(
                                f"[GUARDRAIL] Auto-validation result "
                                f"(attempt {state.fix_attempts}/{MAX_FIX_ATTEMPTS}):\n\n"
                                f"{val_result}\n\n"
                                "Fix the error and write the corrected env.py."
                            )
                        )
                    )
            else:
                state.validation_passed = True
                state.fix_attempts = 0
                self._emit(
                    AgentEvent(
                        type="validation_passed",
                        data={"env_id": state.env_id},
                    )
                )
                extra.append(
                    HumanMessage(
                        content=f"[GUARDRAIL] Validation passed!\n\n{val_result}"
                    )
                )

        return extra

    def _escalate_to_eval_agent(self, env_id: str, error_output: str) -> str:
        """Call the eval agent for deeper diagnosis."""
        self._emit(
            AgentEvent(
                type="tool_start",
                data={"tool": "eval_agent", "args": {"env_id": env_id}},
            )
        )

        recent_context = ""
        for msg in self.messages[-10:]:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                recent_context += msg.content[:500] + "\n---\n"

        diagnosis = run_eval_agent(
            env_id=env_id,
            error_stage="validation",
            error_message=error_output,
            conversation_context=recent_context,
        )

        self._emit(
            AgentEvent(
                type="tool_end",
                data={"tool": "eval_agent", "result": diagnosis[:500]},
            )
        )

        return diagnosis


def create_orchestrator(
    on_event: Callable[[AgentEvent], None] | None = None,
) -> Orchestrator:
    """Factory to create a new orchestrator instance."""
    return Orchestrator(on_event=on_event)
