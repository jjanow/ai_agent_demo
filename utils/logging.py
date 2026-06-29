"""Loguru-based structured logging configuration.

Console sink at INFO+ (configurable via LOG_LEVEL) and a rotating file sink at
DEBUG+ written to logs/agent_{date}.log. Helper functions emit consistent,
structured records for LLM calls, tool calls, and loop iterations.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

_CONFIGURED = False


def setup_logging(log_level: str = "INFO", log_dir: str | Path = "logs") -> "logger.__class__":
    """Configure loguru sinks once. Safe to call multiple times.

    Args:
        log_level: Minimum level for the console sink.
        log_dir: Directory for the dated DEBUG log file.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return logger

    logger.remove()

    logger.add(
        sys.stderr,
        level=log_level.upper(),
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        ),
    )

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.add(
        log_path / f"agent_{date_str}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
    )

    _CONFIGURED = True
    logger.debug("Logging configured (console level={})", log_level.upper())
    return logger


def log_llm_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    stop_reason: str | None = None,
) -> None:
    """Record a single LLM API call with token usage and latency."""
    logger.info(
        "LLM call | model={} input_tokens={} output_tokens={} latency_ms={:.1f} stop_reason={}",
        model,
        input_tokens,
        output_tokens,
        latency_ms,
        stop_reason,
    )


def log_tool_call(
    tool_name: str,
    tool_input: Any,
    result_summary: str,
    latency_ms: float,
    success: bool,
) -> None:
    """Record a single tool invocation with inputs and a result summary."""
    level = "INFO" if success else "WARNING"
    logger.log(
        level,
        "Tool call | name={} success={} latency_ms={:.1f} input={} result={}",
        tool_name,
        success,
        latency_ms,
        _truncate(repr(tool_input), 300),
        _truncate(result_summary, 300),
    )


def log_iteration(iteration: int, stop_reason: str | None) -> None:
    """Record a loop iteration number and the LLM stop_reason."""
    logger.info("Loop iteration | n={} stop_reason={}", iteration, stop_reason)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... (+{len(text) - limit} chars)"
