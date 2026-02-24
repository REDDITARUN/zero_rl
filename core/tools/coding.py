"""Coding tools for the orchestrator agent.

Provides LangChain-compatible tools for:
  - file_read, file_write, file_edit
  - dir_list
  - code_search (ripgrep)
  - shell (subprocess)
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from core.config import ROOT_DIR, ENVS_DIR


def _resolve(path: str) -> Path:
    """Resolve a path relative to ROOT_DIR and guard against traversal."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT_DIR / p
    p = p.resolve()
    if not str(p).startswith(str(ROOT_DIR)):
        raise PermissionError(f"Access denied outside project root: {p}")
    return p


# ── File operations ──────────────────────────────────────────────────────────


@tool
def file_read(path: str, offset: int = 0, limit: int = -1) -> str:
    """Read a file and return its contents with line numbers.

    Args:
        path: File path (absolute or relative to project root).
        offset: Starting line (0-based). Defaults to 0.
        limit: Number of lines to read. -1 means all.
    """
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: File not found: {p}"
    if not p.is_file():
        return f"ERROR: Not a file: {p}"

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(0, offset)
    end = len(lines) if limit < 0 else min(start + limit, len(lines))
    numbered = [f"{i + 1:>6}|{line}" for i, line in enumerate(lines[start:end], start=start)]
    return "\n".join(numbered) if numbered else "(empty file)"


@tool
def file_write(path: str, contents: str) -> str:
    """Write contents to a file, creating parent directories as needed.

    Args:
        path: Destination file path.
        contents: Full file contents to write.
    """
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(contents, encoding="utf-8")
    return f"Wrote {len(contents)} bytes to {p.relative_to(ROOT_DIR)}"


@tool
def file_edit(path: str, old_string: str, new_string: str) -> str:
    """Replace an exact substring in a file.

    Args:
        path: File to edit.
        old_string: The exact text to find (must appear exactly once).
        new_string: The replacement text.
    """
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: File not found: {p}"

    text = p.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        return "ERROR: old_string not found in file."
    if count > 1:
        return f"ERROR: old_string found {count} times — must be unique. Add more context."

    text = text.replace(old_string, new_string, 1)
    p.write_text(text, encoding="utf-8")
    return f"Edited {p.relative_to(ROOT_DIR)}"


# ── Directory ────────────────────────────────────────────────────────────────


@tool
def dir_list(path: str = ".", max_depth: int = 2) -> str:
    """List directory contents as a tree.

    Args:
        path: Directory path. Defaults to project root.
        max_depth: How many levels deep to recurse.
    """
    p = _resolve(path)
    if not p.is_dir():
        return f"ERROR: Not a directory: {p}"

    lines: list[str] = []

    def _walk(d: Path, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda e: (e.is_file(), e.name))
        except PermissionError:
            return
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                _walk(entry, depth + 1, prefix + extension)

    lines.append(f"{p.relative_to(ROOT_DIR)}/")
    _walk(p, 1, "")
    return "\n".join(lines)


# ── Search ───────────────────────────────────────────────────────────────────


@tool
def code_search(pattern: str, path: str = ".", glob_filter: str = "") -> str:
    """Search for a regex pattern in files using ripgrep.

    Args:
        pattern: Regex pattern to search for.
        path: Directory to search in.
        glob_filter: Optional glob filter (e.g. '*.py').
    """
    p = _resolve(path)
    cmd = ["rg", "--line-number", "--no-heading", "--max-count=50", pattern, str(p)]
    if glob_filter:
        cmd.insert(1, f"--glob={glob_filter}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return "ERROR: ripgrep (rg) not found. Install it: brew install ripgrep"
    except subprocess.TimeoutExpired:
        return "ERROR: Search timed out after 15 seconds."

    output = result.stdout.strip()
    if not output:
        return "No matches found."

    lines = output.splitlines()
    rel_lines = []
    for line in lines[:100]:
        try:
            rel_lines.append(line.replace(str(ROOT_DIR) + "/", ""))
        except Exception:
            rel_lines.append(line)
    return "\n".join(rel_lines)


# ── Shell ────────────────────────────────────────────────────────────────────


@tool
def shell(command: str, working_directory: Optional[str] = None) -> str:
    """Execute a shell command and return stdout + stderr.

    Args:
        command: The shell command to run.
        working_directory: Optional working directory. Defaults to project root.
    """
    cwd = _resolve(working_directory) if working_directory else ROOT_DIR

    blocked_patterns = [
        "rm -rf /", "rm -rf ~", "rm -rf $HOME",
        ":(){ :|:& };:", "mkfs", "dd if=",
        "> /dev/sd", "chmod -R 777 /",
        "curl | sh", "curl | bash", "wget | sh",
    ]
    cmd_lower = command.lower().replace("\\", "")
    if any(b in cmd_lower for b in blocked_patterns):
        return "ERROR: Potentially destructive command blocked."

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(cwd),
            env={**os.environ, "PYTHONPATH": str(ROOT_DIR)},
        )
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 60 seconds."

    output_parts: list[str] = []
    if result.stdout.strip():
        output_parts.append(result.stdout.strip())
    if result.stderr.strip():
        output_parts.append(f"STDERR:\n{result.stderr.strip()}")
    output_parts.append(f"(exit code: {result.returncode})")

    combined = "\n".join(output_parts)
    if len(combined) > 10000:
        combined = combined[:5000] + "\n... (truncated) ...\n" + combined[-5000:]
    return combined


# ── All tools list ───────────────────────────────────────────────────────────

ALL_CODING_TOOLS = [file_read, file_write, file_edit, dir_list, code_search, shell]
