"""Tool definitions, JSON schemas, and dispatch for Anthropic tool_use.

Every tool:
  * validates its inputs before doing any work,
  * returns a structured dict {"success", "result", "error"},
  * never raises out to the caller (all exceptions are caught), and
  * is logged via loguru with latency.

Tool execution is wrapped with a hard timeout (config.tool_timeout_s). Each tool
runs in a separate, short-lived process; if it overruns the timeout the process
is terminated (SIGTERM, then SIGKILL), so a runaway tool's CPU/memory work is
actually reclaimed rather than leaked into the background. A timed-out tool
returns a structured timeout error.
"""

from __future__ import annotations

import multiprocessing
import queue as queue_mod
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from config import get_settings
from utils.logging import log_tool_call
from utils.safety import (
    safe_eval_math,
    safe_resolve_path,
)

# --------------------------------------------------------------------------- #
# Tool result helpers
# --------------------------------------------------------------------------- #


def _ok(result: Any) -> dict:
    return {"success": True, "result": result, "error": None}


def _err(message: str) -> dict:
    return {"success": False, "result": None, "error": message}


# --------------------------------------------------------------------------- #
# Individual tools
# --------------------------------------------------------------------------- #


def web_search(query: str) -> dict:
    """Search the web via Tavily and return the top 3 results."""
    if not isinstance(query, str) or not query.strip():
        return _err("`query` must be a non-empty string.")

    cfg = get_settings()
    if not cfg.tavily_api_key:
        return _err("TAVILY_API_KEY is not configured; web_search is unavailable.")

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=cfg.tavily_api_key)
        response = client.search(query=query.strip(), max_results=3)
        results = []
        for item in (response or {}).get("results", [])[:3]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": (item.get("content", "") or "")[:500],
                    "url": item.get("url", ""),
                }
            )
        if not results:
            return _ok({"query": query, "results": [], "note": "No results found."})
        return _ok({"query": query, "results": results})
    except Exception as exc:  # noqa: BLE001 - tools must never raise
        return _err(f"Web search failed: {exc}")


def calculator(expression: str) -> dict:
    """Evaluate a math expression via the safe AST evaluator (no arbitrary code)."""
    ok, value, error = safe_eval_math(expression)
    if not ok:
        return _err(error or "Invalid expression.")
    return _ok({"expression": (expression or "").strip(), "value": value})


def get_datetime() -> dict:
    """Return the current UTC datetime and timezone info."""
    try:
        now = datetime.now(timezone.utc)
        return _ok(
            {
                "iso": now.isoformat(),
                "utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": "UTC",
                "unix": int(now.timestamp()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"Could not get datetime: {exc}")


def read_file(path: str) -> dict:
    """Read a UTF-8 text file restricted to the current working dir and subdirs."""
    guard = safe_resolve_path(path)
    if not guard.ok:
        return _err(guard.error or "Invalid path.")

    target = Path(guard.value)
    if not target.exists():
        return _err(f"File not found: {path}")
    if not target.is_file():
        return _err(f"Not a regular file: {path}")

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        if len(content) > 100_000:
            content = content[:100_000] + "\n... (truncated)"
        return _ok({"path": path, "content": content})
    except Exception as exc:  # noqa: BLE001
        return _err(f"Could not read file: {exc}")


def write_file(path: str, content: str, confirm_overwrite: bool = False) -> dict:
    """Write text to a file in cwd/subdirs; never overwrite without confirmation."""
    if not isinstance(content, str):
        return _err("`content` must be a string.")

    guard = safe_resolve_path(path)
    if not guard.ok:
        return _err(guard.error or "Invalid path.")

    target = Path(guard.value)
    if target.exists() and not confirm_overwrite:
        return _err(
            f"File '{path}' already exists. Re-call with confirm_overwrite=true to replace it."
        )

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return _ok({"path": path, "bytes_written": len(content.encode("utf-8"))})
    except Exception as exc:  # noqa: BLE001
        return _err(f"Could not write file: {exc}")


def summarize_text(text: str) -> dict:
    """Summarize long text by calling the LLM recursively (one extra call)."""
    if not isinstance(text, str) or not text.strip():
        return _err("`text` must be a non-empty string.")

    cfg = get_settings()
    if not cfg.anthropic_api_key:
        return _err("ANTHROPIC_API_KEY is not configured; summarize_text is unavailable.")

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=cfg.anthropic_api_key, timeout=cfg.api_timeout_s)
        prompt = (
            "Summarize the following text concisely, preserving key facts and "
            "numbers. Use at most 5 sentences.\n\n" + text[:50_000]
        )
        resp = client.messages.create(
            model=cfg.anthropic_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        return _ok({"summary": summary.strip()})
    except Exception as exc:  # noqa: BLE001
        return _err(f"Summarization failed: {exc}")


# --------------------------------------------------------------------------- #
# Registry, schemas, and dispatch
# --------------------------------------------------------------------------- #

_TOOL_FUNCS: dict[str, Callable[..., dict]] = {
    "web_search": web_search,
    "calculator": calculator,
    "get_datetime": get_datetime,
    "read_file": read_file,
    "write_file": write_file,
    "summarize_text": summarize_text,
}


def get_tool_schemas() -> list[dict]:
    """Return Anthropic tool schemas for all available tools."""
    return [
        {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use when the user asks "
                "about recent events, facts you are unsure of, or anything that "
                "needs up-to-date external data. Returns the top 3 results."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
        {
            "name": "calculator",
            "description": (
                "Evaluate an arithmetic or basic math expression. Supports + - * / "
                "** % and the functions sqrt, sin, cos, tan, log, pow, abs plus "
                "constants pi and e. Use for any non-trivial arithmetic."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '2 * (3 + 4) ** 2'.",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "get_datetime",
            "description": "Get the current UTC date and time. Use when the user asks about the current time/date.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file. Restricted to the current working "
                "directory and its subdirectories. Use to inspect local files."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the working directory.",
                    }
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": (
                "Write text to a file within the working directory. Will NOT "
                "overwrite an existing file unless confirm_overwrite is true."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the working directory.",
                    },
                    "content": {"type": "string", "description": "Text to write."},
                    "confirm_overwrite": {
                        "type": "boolean",
                        "description": "Set true to overwrite an existing file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "summarize_text",
            "description": (
                "Summarize a long block of text into a few sentences. Use when "
                "you need to condense large content before reasoning about it."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to summarize."}
                },
                "required": ["text"],
            },
        },
    ]


def _summarize_result(result: dict) -> str:
    if result.get("success"):
        return f"ok: {str(result.get('result'))[:200]}"
    return f"error: {result.get('error')}"


# Process start method for tool isolation. "spawn" is the safe default: it does
# not inherit the parent's threads/locks (important under Streamlit), at the cost
# of re-importing this module in the child. Overridable for tests.
MP_START_METHOD = "spawn"

# How long to wait for a terminated worker to die before escalating to SIGKILL.
_REAP_GRACE_S = 1.0


def _tool_worker(name: str, tool_input: dict, out_queue: Any) -> None:
    """Run a tool in a child process and put its structured result on a queue.

    Looks the tool up by name in the registry (rather than receiving the callable
    by reference) so it works under both "spawn" (fresh re-import) and "fork"
    (inherited registry) start methods.
    """
    try:
        func = _TOOL_FUNCS.get(name)
        if func is None:
            out_queue.put(_err(f"Unknown tool: {name}"))
            return
        try:
            out_queue.put(func(**tool_input))
        except TypeError as exc:
            out_queue.put(_err(f"Invalid arguments for '{name}': {exc}"))
        except Exception as exc:  # noqa: BLE001 - tools must never raise out
            out_queue.put(_err(f"Tool '{name}' raised unexpectedly: {exc}"))
    except Exception:  # noqa: BLE001 - last-resort guard; never hang the parent
        try:
            out_queue.put(_err(f"Tool '{name}' failed in its worker process."))
        except Exception:  # noqa: BLE001
            pass


def _reap(process: Any) -> None:
    """Terminate a still-running worker, escalating SIGTERM -> SIGKILL."""
    if process.is_alive():
        process.terminate()
        process.join(_REAP_GRACE_S)
    if process.is_alive():
        process.kill()
        process.join(_REAP_GRACE_S)


def dispatch_tool(name: str, tool_input: dict, timeout_s: float | None = None) -> dict:
    """Execute a tool by name with a hard timeout, logging the call.

    The tool runs in an isolated child process so a runaway tool can be truly
    killed on timeout (reclaiming its CPU/memory) instead of leaking. Always
    returns a structured {"success", "result", "error"} dict; never raises.
    """
    timeout_s = timeout_s if timeout_s is not None else get_settings().tool_timeout_s

    # Fast-path validation in the parent to avoid spawning a process needlessly.
    if _TOOL_FUNCS.get(name) is None:
        result = _err(f"Unknown tool: {name}")
        log_tool_call(name, tool_input, _summarize_result(result), 0.0, False)
        return result

    if not isinstance(tool_input, dict):
        result = _err("Tool input must be an object.")
        log_tool_call(name, tool_input, _summarize_result(result), 0.0, False)
        return result

    start = time.perf_counter()
    try:
        ctx = multiprocessing.get_context(MP_START_METHOD)
        out_queue = ctx.Queue()
        process = ctx.Process(
            target=_tool_worker, args=(name, tool_input, out_queue), daemon=True
        )
        process.start()
    except Exception as exc:  # noqa: BLE001 - process machinery unavailable
        result = _err(f"Tool '{name}' could not be launched: {exc}")
        log_tool_call(name, tool_input, _summarize_result(result), 0.0, False)
        return result

    try:
        # Drain the result first (avoids the join-before-drain queue deadlock).
        result = out_queue.get(timeout=timeout_s)
    except queue_mod.Empty:
        if process.is_alive():
            result = _err(f"Tool '{name}' timed out after {timeout_s:.0f}s.")
        else:
            result = _err(f"Tool '{name}' exited without returning a result.")
    finally:
        _reap(process)
        out_queue.close()

    latency_ms = (time.perf_counter() - start) * 1000.0
    if not isinstance(result, dict) or "success" not in result:
        result = _err(f"Tool '{name}' returned a malformed result.")

    log_tool_call(name, tool_input, _summarize_result(result), latency_ms, result["success"])
    return result
