"""Tests for the safety guards in utils.safety."""

from __future__ import annotations

import pytest

from utils.safety import redact_secrets, safe_eval_math, safe_resolve_path


def test_redact_secrets_redacts_sk_key() -> None:
    text = "here is my key sk-ant-abcdef1234567890XYZ and more"
    result = redact_secrets(text)
    assert "sk-ant-abcdef1234567890XYZ" not in result
    assert "[REDACTED_API_KEY]" in result


def test_redact_secrets_redacts_tavily_key() -> None:
    text = "search key tvly-abc12345defXYZ done"
    result = redact_secrets(text)
    assert "tvly-abc12345defXYZ" not in result
    assert "[REDACTED_API_KEY]" in result


def test_redact_secrets_leaves_normal_text() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    assert redact_secrets(text) == text


def test_safe_resolve_path_rejects_traversal() -> None:
    assert not safe_resolve_path("../x").ok


def test_safe_resolve_path_rejects_absolute() -> None:
    assert not safe_resolve_path("/etc/passwd").ok


def test_safe_resolve_path_rejects_dotfile() -> None:
    assert not safe_resolve_path(".env").ok


def test_safe_resolve_path_rejects_dotdir() -> None:
    assert not safe_resolve_path(".git/config").ok


def test_safe_resolve_path_rejects_sensitive_suffix() -> None:
    assert not safe_resolve_path("secret.pem").ok


def test_safe_resolve_path_accepts_normal_relative(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = safe_resolve_path("notes.txt")
    assert result.ok
    assert result.value.endswith("notes.txt")


def test_safe_eval_math_arithmetic() -> None:
    assert safe_eval_math("2 * (3 + 4) ** 2") == (True, 98, None)


def test_safe_eval_math_functions_and_constants() -> None:
    ok, value, error = safe_eval_math("sqrt(16) + log(e)")
    assert ok is True
    assert error is None
    assert value == pytest.approx(5.0)


def test_safe_eval_math_division_by_zero() -> None:
    ok, _value, error = safe_eval_math("1 / 0")
    assert ok is False
    assert "zero" in error.lower()


def test_safe_eval_math_blocks_import() -> None:
    ok, _value, _error = safe_eval_math("__import__('os').system('ls')")
    assert ok is False


def test_safe_eval_math_empty() -> None:
    ok, _value, error = safe_eval_math("")
    assert ok is False
    assert "empty" in error.lower()


def test_safe_eval_math_huge_exponent() -> None:
    ok, _value, _error = safe_eval_math("9 ** 99999")
    assert ok is False


def test_safe_eval_math_disallowed_name() -> None:
    ok, _value, _error = safe_eval_math("foo(2)")
    assert ok is False
