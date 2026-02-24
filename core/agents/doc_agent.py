"""Doc agent — explores cloned documentation repos to answer API questions.

Cloned repos live under docs/:
  - docs/gymnasium-repo/docs/   (Gymnasium markdown + RST)
  - docs/genesis-doc/source/    (Genesis RST docs)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from core.config import DOCS_DIR, ROOT_DIR, get_chat_model
from core.prompts.doc_agent import DOC_AGENT_SYSTEM


ALLOWED_PREFIXES = ("docs/", "core/prompts/skills/")

_EXPECTED_DIRS = [
    DOCS_DIR / "gymnasium-repo" / "docs",
    DOCS_DIR / "genesis-doc" / "source",
]


def _is_allowed_path(path: str) -> bool:
    """Check that the path is within allowed doc directories."""
    return any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _docs_available() -> tuple[bool, str]:
    """Check whether the cloned doc repos exist."""
    missing = [str(d.relative_to(ROOT_DIR)) for d in _EXPECTED_DIRS if not d.is_dir()]
    if missing:
        return False, (
            f"Documentation directories not found: {', '.join(missing)}. "
            "Clone them first:\n"
            "  git clone --depth 1 https://github.com/Farama-Foundation/Gymnasium.git docs/gymnasium-repo\n"
            "  git clone --depth 1 https://github.com/Genesis-Embodied-AI/genesis-doc.git docs/genesis-doc"
        )
    return True, ""


@tool
def doc_search(query: str, directory: str = "docs") -> str:
    """Search documentation files for a keyword or pattern.

    Args:
        query: The search term or regex pattern.
        directory: Subdirectory to search (must be under docs/ or core/prompts/skills/).
    """
    if not _is_allowed_path(directory if directory.endswith("/") else directory + "/"):
        return f"Access denied: search is restricted to {ALLOWED_PREFIXES}"
    search_path = ROOT_DIR / directory
    if not search_path.is_dir():
        return f"Directory not found: {directory}"

    try:
        result = subprocess.run(
            ["rg", "--line-number", "--no-heading", "--max-count=30",
             "--type=rst", "--type=md", "--type=py", "-i",
             query, str(search_path)],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        return "ERROR: ripgrep (rg) not found."
    except subprocess.TimeoutExpired:
        return "ERROR: Search timed out."

    output = result.stdout.strip()
    if not output:
        return f"No matches for '{query}' in {directory}/"
    lines = output.splitlines()[:60]
    return "\n".join(l.replace(str(ROOT_DIR) + "/", "") for l in lines)


@tool
def doc_read(path: str, offset: int = 0, limit: int = 100) -> str:
    """Read a documentation file (scoped to docs/ and skills/).

    Args:
        path: File path relative to project root (must be under docs/ or core/prompts/skills/).
        offset: Starting line (0-based).
        limit: Number of lines to read.
    """
    if not _is_allowed_path(path):
        return f"Access denied: read is restricted to {ALLOWED_PREFIXES}"
    p = ROOT_DIR / path
    if not p.exists():
        return f"File not found: {path}"
    if not p.is_file():
        return f"Not a file: {path}"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(0, offset)
    end = min(start + limit, len(lines))
    numbered = [f"{i+1:>6}|{line}" for i, line in enumerate(lines[start:end], start=start)]
    return "\n".join(numbered) if numbered else "(empty file)"


@tool
def doc_list(path: str = "docs") -> str:
    """List contents of a documentation directory (scoped to docs/ and skills/).

    Args:
        path: Directory path relative to project root (must be under docs/ or core/prompts/skills/).
    """
    if not _is_allowed_path(path if path.endswith("/") else path + "/"):
        return f"Access denied: listing is restricted to {ALLOWED_PREFIXES}"
    p = ROOT_DIR / path
    if not p.is_dir():
        return f"Not a directory: {path}"
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
    lines = [f"  {e.name}{'/' if e.is_dir() else ''}" for e in entries[:50]]
    return f"{path}/\n" + "\n".join(lines)


_DOC_TOOLS = [doc_search, doc_read, doc_list]


def run_doc_agent(question: str, max_iterations: int = 8) -> str:
    """Run the doc agent to answer a technical documentation question.

    Uses a simple ReAct loop: LLM reasons, calls tools, observes, repeats.
    Returns early if docs are missing or all tool results are errors.
    """
    ok, err_msg = _docs_available()
    if not ok:
        return err_msg

    llm = get_chat_model().bind_tools(_DOC_TOOLS)
    tool_map = {t.name: t for t in _DOC_TOOLS}

    messages: list = [
        SystemMessage(content=DOC_AGENT_SYSTEM),
        HumanMessage(content=question),
    ]

    consecutive_empty = 0

    for iteration in range(max_iterations):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content if isinstance(response.content, str) else str(response.content)

        all_empty = True
        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if tool_fn is None:
                result = f"Unknown tool: {tc['name']}"
            else:
                result = tool_fn.invoke(tc["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )
            if "No matches" not in result and "not found" not in result.lower():
                all_empty = False

        if all_empty:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

        if consecutive_empty >= 3:
            messages.append(
                HumanMessage(
                    content=(
                        "[SYSTEM] Multiple searches returned no results. "
                        "Stop searching and provide your best answer based "
                        "on what you already know. If you truly cannot find "
                        "the answer, say so."
                    )
                )
            )

    # Final call without tools to force a text response
    messages.append(
        HumanMessage(
            content=(
                "[SYSTEM] You have used all available iterations. "
                "Summarize your findings now. Do NOT call any more tools."
            )
        )
    )
    bare_llm = get_chat_model()
    final = bare_llm.invoke(messages)
    if final.content:
        return final.content if isinstance(final.content, str) else str(final.content)
    return "Could not find an answer in the documentation. Try rephrasing the question."
