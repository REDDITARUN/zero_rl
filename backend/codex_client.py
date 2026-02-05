"""Async bridge client for invoking the official Codex SDK."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Awaitable, Callable

from config import CODEX_MODEL, CODEX_TIMEOUT_SEC, ROOT_DIR, USE_CODEX_SDK


class CodexSDKError(RuntimeError):
    """Raised when the Codex bridge process fails."""


class CodexSDKClient:
    """Runs agent prompts through the Node-based official Codex SDK."""

    def __init__(self) -> None:
        self.enabled = USE_CODEX_SDK
        self.model = CODEX_MODEL
        self.timeout_sec = CODEX_TIMEOUT_SEC
        self.bridge_dir = ROOT_DIR / "backend" / "codex_bridge"
        self.bridge_script = self.bridge_dir / "run-agent.mjs"
        self.api_key = os.getenv("CODEX_API_KEY") or os.getenv("OPENAI_API_KEY")

    async def run_json(
        self,
        *,
        agent_id: str,
        prompt: str,
        output_schema: dict[str, Any],
        thread_id: str | None = None,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """Execute a Codex turn and parse structured JSON response + thread id."""

        if not self.enabled:
            raise CodexSDKError("Codex SDK is disabled by USE_CODEX_SDK=false")
        if not self.api_key:
            raise CodexSDKError("Missing OPENAI_API_KEY/CODEX_API_KEY")
        if not self.bridge_script.exists():
            raise CodexSDKError(f"Missing Codex bridge script: {self.bridge_script}")

        payload = {
            "agent_id": agent_id,
            "prompt": prompt,
            "model": self.model,
            "output_schema": output_schema,
            "working_directory": str(ROOT_DIR),
            "sandbox_mode": "read-only",
            "approval_policy": "never",
            "network_access_enabled": True,
            "api_key": self.api_key,
            "thread_id": thread_id,
        }

        process = await asyncio.create_subprocess_exec(
            "node",
            str(self.bridge_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.bridge_dir),
        )

        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps(payload).encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
        except asyncio.TimeoutError as exc:
            process.kill()
            raise CodexSDKError(f"Codex bridge timed out after {self.timeout_sec}s") from exc
        except Exception as exc:  # noqa: BLE001
            process.kill()
            raise CodexSDKError(f"Codex bridge payload write failed: {type(exc).__name__}: {exc}") from exc

        if process.stdout is None or process.stderr is None:
            process.kill()
            raise CodexSDKError("Codex bridge failed to initialize stdio")

        stderr_task = asyncio.create_task(process.stderr.read())
        raw_lines: list[str] = []
        envelope: dict[str, Any] | None = None
        deadline = asyncio.get_running_loop().time() + self.timeout_sec

        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                process.kill()
                raise CodexSDKError(f"Codex bridge timed out after {self.timeout_sec}s")

            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                process.kill()
                raise CodexSDKError(f"Codex bridge timed out after {self.timeout_sec}s") from exc

            if not line:
                break

            text = line.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            raw_lines.append(text)

            parsed = _try_parse_json_line(text)
            if parsed is None:
                continue

            kind = str(parsed.get("type", ""))
            if kind == "progress":
                if progress_callback is not None:
                    await progress_callback(parsed)
                continue

            if kind == "result" or ("ok" in parsed and "final_response" in parsed):
                envelope = parsed

        try:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                process.kill()
                raise CodexSDKError(f"Codex bridge timed out after {self.timeout_sec}s")
            await asyncio.wait_for(process.wait(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            process.kill()
            raise CodexSDKError(f"Codex bridge timed out after {self.timeout_sec}s") from exc

        stderr = await stderr_task
        raw_stderr = stderr.decode("utf-8", errors="ignore").strip()

        if envelope is None:
            envelope = _find_result_envelope(raw_lines)
        if envelope is None:
            preview = "\n".join(raw_lines[-5:])
            if process.returncode and raw_stderr:
                raise CodexSDKError(raw_stderr)
            raise CodexSDKError(
                f"Invalid bridge JSON envelope (exit={process.returncode}): {preview[:500]}"
            )

        if process.returncode != 0 or not envelope.get("ok"):
            raise CodexSDKError(str(envelope.get("error") or raw_stderr or "Codex bridge failed"))

        final_response = envelope.get("final_response", "")
        if not isinstance(final_response, str) or not final_response.strip():
            raise CodexSDKError("Codex returned empty final_response")

        parsed = _parse_json_response(final_response)
        return parsed, envelope.get("thread_id")


def _parse_json_response(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise CodexSDKError("Model did not return valid JSON object")

    snippet = stripped[start : end + 1]
    parsed = json.loads(snippet)
    if not isinstance(parsed, dict):
        raise CodexSDKError("Model output JSON is not an object")
    return parsed


def _try_parse_json_line(text: str) -> dict[str, Any] | None:
    cleaned = _strip_ansi(text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = cleaned[start : end + 1]
        try:
            obj = json.loads(snippet)
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    return obj


def _find_result_envelope(lines: list[str]) -> dict[str, Any] | None:
    for text in reversed(lines):
        obj = _try_parse_json_line(text)
        if obj is None:
            continue
        if obj.get("type") == "result" or ("ok" in obj and ("final_response" in obj or "error" in obj)):
            return obj
    return None


def _strip_ansi(value: str) -> str:
    out = []
    i = 0
    n = len(value)
    while i < n:
        ch = value[i]
        if ch == "\x1b" and i + 1 < n and value[i + 1] == "[":
            i += 2
            while i < n and value[i] not in "ABCDEFGHJKSTfmnsu":
                i += 1
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)
