"""Eval agent — investigates validation failures using read-only file access.

Called when deterministic eval_tool fails and the orchestrator can't fix it
after initial attempts. This agent reads the generated code + error output
and diagnoses the root cause with LLM reasoning.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from core.config import ENVS_DIR, get_chat_model
from core.prompts.eval_agent import EVAL_AGENT_SYSTEM
from core.tools.coding import code_search, dir_list, file_read

_EVAL_TOOLS = [file_read, dir_list, code_search]


def run_eval_agent(
    env_id: str,
    error_stage: str,
    error_message: str,
    conversation_context: str = "",
    max_iterations: int = 8,
) -> str:
    """Run the eval agent to diagnose a validation failure.

    Args:
        env_id: The environment directory name.
        error_stage: Which validation stage failed.
        error_message: The error traceback or message.
        conversation_context: Optional prior conversation for context.
        max_iterations: Max reasoning steps.

    Returns:
        Diagnosis with specific fix instructions.
    """
    llm = get_chat_model().bind_tools(_EVAL_TOOLS)
    tool_map = {t.name: t for t in _EVAL_TOOLS}

    prompt = (
        f"Environment: envs/{env_id}/\n"
        f"Validation failed at stage: {error_stage}\n\n"
        f"Error:\n{error_message}\n\n"
    )
    if conversation_context:
        prompt += f"Context from conversation:\n{conversation_context}\n\n"
    prompt += (
        "Please investigate the generated code, find the root cause, "
        "and provide specific fix instructions."
    )

    messages: list = [
        SystemMessage(content=EVAL_AGENT_SYSTEM),
        HumanMessage(content=prompt),
    ]

    for _ in range(max_iterations):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if tool_fn is None:
                result = f"Unknown tool: {tc['name']}"
            else:
                result = tool_fn.invoke(tc["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    return "Eval agent reached max iterations without a final diagnosis."
