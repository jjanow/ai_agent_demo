"""Lightweight input/output and tool safety guards.

These are defense-in-depth checks, not a substitute for proper sandboxing. They
cover the common failure modes: oversized/garbage input, leaked API keys in
output, path traversal in file tools, and calculator expression injection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_INPUT_CHARS = 10_000

# Anthropic/OpenAI-style secret keys, e.g. sk-ant-... or sk-...
_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b")

# Whitelist of characters/tokens allowed in calculator expressions.
# Allowed: digits, whitespace, + - * / . ( ) ** and the named functions below.
_CALC_ALLOWED_NAMES = {"sqrt", "sin", "cos", "log", "pi", "e", "tan", "pow", "abs"}
_CALC_TOKEN_RE = re.compile(r"[A-Za-z_]+")
_CALC_CHAR_RE = re.compile(r"^[0-9\s+\-*/.()%,a-zA-Z_]+$")


@dataclass
class GuardResult:
    ok: bool
    value: str = ""
    error: str | None = None


def sanitize_input(text: str, max_chars: int = MAX_INPUT_CHARS) -> GuardResult:
    """Strip null bytes, enforce a length cap, and reject mostly-garbage input.

    Rejects input that is more than 50% non-printable characters.
    """
    if text is None:
        return GuardResult(ok=False, error="Input was None.")

    cleaned = text.replace("\x00", "")

    if len(cleaned) > max_chars:
        return GuardResult(
            ok=False, error=f"Input too long ({len(cleaned)} > {max_chars} chars)."
        )

    if cleaned:
        non_printable = sum(1 for c in cleaned if not (c.isprintable() or c in "\n\r\t"))
        if non_printable / len(cleaned) > 0.5:
            return GuardResult(
                ok=False, error="Input rejected: more than 50% non-printable characters."
            )

    return GuardResult(ok=True, value=cleaned)


def redact_secrets(text: str) -> str:
    """Redact anything that looks like a leaked API key from output text."""
    if not text:
        return text
    return _API_KEY_RE.sub("[REDACTED_API_KEY]", text)


def safe_resolve_path(path: str, base_dir: str | Path | None = None) -> GuardResult:
    """Resolve a user-supplied path, rejecting traversal and out-of-cwd access.

    Rules:
      - Reject any path containing ".." segments.
      - Reject absolute paths.
      - Final resolved path must remain within base_dir (defaults to cwd).
    """
    if path is None or path == "":
        return GuardResult(ok=False, error="Empty path.")

    if ".." in Path(path).parts:
        return GuardResult(ok=False, error="Path traversal ('..') is not allowed.")

    if Path(path).is_absolute():
        return GuardResult(ok=False, error="Absolute paths are not allowed.")

    base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    resolved = (base / path).resolve()

    try:
        resolved.relative_to(base)
    except ValueError:
        return GuardResult(ok=False, error="Path escapes the allowed base directory.")

    return GuardResult(ok=True, value=str(resolved))


def validate_calculator_expression(expression: str) -> GuardResult:
    """Validate that a calculator expression only uses whitelisted tokens.

    Allowed: digits, whitespace, + - * / . ( ) % , and the named math helpers
    sqrt/sin/cos/log/tan/pow/abs plus constants pi/e.
    """
    if expression is None or not expression.strip():
        return GuardResult(ok=False, error="Empty expression.")

    expr = expression.strip()

    if not _CALC_CHAR_RE.match(expr):
        return GuardResult(ok=False, error="Expression contains disallowed characters.")

    for name in _CALC_TOKEN_RE.findall(expr):
        if name not in _CALC_ALLOWED_NAMES:
            return GuardResult(
                ok=False, error=f"Disallowed name in expression: '{name}'."
            )

    # Block dunder / attribute access just in case.
    if "__" in expr:
        return GuardResult(ok=False, error="Disallowed token '__' in expression.")

    return GuardResult(ok=True, value=expr)
