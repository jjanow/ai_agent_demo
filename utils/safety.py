"""Lightweight input/output and tool safety guards.

These are defense-in-depth checks, not a substitute for proper sandboxing. They
cover the common failure modes: oversized/garbage input, leaked API keys in
output, path traversal in file tools, and calculator expression injection.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path

MAX_INPUT_CHARS = 10_000

# Cap exponentiation to avoid trivial CPU/memory exhaustion via huge powers.
MAX_EXPONENT = 1000

# Anthropic/OpenAI-style secret keys, e.g. sk-ant-... or sk-...
_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b")

# Tavily-style search API keys, e.g. tvly-...
_TAVILY_KEY_RE = re.compile(r"\btvly-[A-Za-z0-9_\-]{8,}\b")

# File suffixes that typically hold private keys/certificates.
_SENSITIVE_SUFFIXES = {".pem", ".key", ".crt", ".p12", ".pfx"}

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
    redacted = _API_KEY_RE.sub("[REDACTED_API_KEY]", text)
    redacted = _TAVILY_KEY_RE.sub("[REDACTED_API_KEY]", redacted)
    return redacted


def safe_resolve_path(path: str, base_dir: str | Path | None = None) -> GuardResult:
    """Resolve a user-supplied path, rejecting traversal and out-of-cwd access.

    Rules:
      - Reject any path containing ".." segments.
      - Reject absolute paths.
      - Reject dotfiles/dotdirs (e.g. .env, .git, .ssh) and private-key/cert files.
      - Final resolved path must remain within base_dir (defaults to cwd).
    """
    if path is None or path == "":
        return GuardResult(ok=False, error="Empty path.")

    if ".." in Path(path).parts:
        return GuardResult(ok=False, error="Path traversal ('..') is not allowed.")

    if Path(path).is_absolute():
        return GuardResult(ok=False, error="Absolute paths are not allowed.")

    parts = Path(path).parts
    if any(part.startswith(".") for part in parts) or (
        Path(path).suffix.lower() in _SENSITIVE_SUFFIXES
    ):
        return GuardResult(ok=False, error="Access to sensitive file is not allowed.")

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


# Whitelisted callables and constants for AST-based math evaluation.
_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "pow": math.pow,
    "abs": abs,
}
_MATH_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class _ExponentTooLarge(Exception):
    """Internal signal that an exponent exceeded MAX_EXPONENT."""


def safe_eval_math(expression: str) -> tuple[bool, "int | float | None", "str | None"]:
    """Safely evaluate a math expression using an AST whitelist.

    Returns (ok, value, error). On success error is None; on failure value is
    None. Never calls eval()/exec(); only a fixed set of arithmetic operators,
    math functions, and constants are permitted.
    """
    if expression is None or not expression.strip():
        return (False, None, "Empty expression.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return (False, None, f"Could not evaluate expression: {exc}")

    def _eval(node: ast.AST) -> "int | float":
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Disallowed expression.")
            return node.value

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            left = _eval(node.left)
            right = _eval(node.right)
            if op_type is ast.Pow:
                if abs(right) > MAX_EXPONENT:
                    raise _ExponentTooLarge()
                return left ** right
            handler = _ALLOWED_BINOPS.get(op_type)
            if handler is None:
                raise ValueError("Disallowed expression.")
            return handler(left, right)

        if isinstance(node, ast.UnaryOp):
            handler = _ALLOWED_UNARYOPS.get(type(node.op))
            if handler is None:
                raise ValueError("Disallowed expression.")
            return handler(_eval(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ValueError("Disallowed expression.")
            func_name = node.func.id
            func = _MATH_FUNCS.get(func_name)
            if func is None:
                raise ValueError(f"Disallowed name: '{func_name}'.")
            args = [_eval(arg) for arg in node.args]
            if func_name == "pow" and len(args) >= 2 and abs(args[1]) > MAX_EXPONENT:
                raise _ExponentTooLarge()
            return func(*args)

        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                raise ValueError("Disallowed expression.")
            if node.id in _MATH_CONSTS:
                return _MATH_CONSTS[node.id]
            raise ValueError(f"Disallowed name: '{node.id}'.")

        raise ValueError("Disallowed expression.")

    try:
        value = _eval(tree)
    except _ExponentTooLarge:
        return (False, None, "Exponent too large.")
    except ZeroDivisionError:
        return (False, None, "Division by zero.")
    except ValueError as exc:
        return (False, None, str(exc))
    except Exception as exc:  # noqa: BLE001 - surface evaluation errors safely
        return (False, None, f"Could not evaluate expression: {exc}")

    return (True, value, None)
