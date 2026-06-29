"""Central configuration loaded from environment via python-dotenv.

No values are hardcoded here; every setting is read from the environment
(optionally populated by a local .env file). Import `settings` to access them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env from the project directory (and current working dir) if present.
load_dotenv()


def _get_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration."""

    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    max_iterations: int = field(default_factory=lambda: _get_int("MAX_ITERATIONS", "10"))
    token_budget: int = field(default_factory=lambda: _get_int("TOKEN_BUDGET", "80000"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    # Operational constants (also overridable via env, never hardcoded inline).
    api_timeout_s: float = field(default_factory=lambda: float(os.getenv("API_TIMEOUT_S", "30")))
    tool_timeout_s: float = field(default_factory=lambda: float(os.getenv("TOOL_TIMEOUT_S", "10")))
    max_input_chars: int = field(default_factory=lambda: _get_int("MAX_INPUT_CHARS", "10000"))
    max_output_tokens: int = field(default_factory=lambda: _get_int("MAX_OUTPUT_TOKENS", "4096"))

    def require_anthropic(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.anthropic_api_key


def get_settings() -> Settings:
    """Build a fresh Settings snapshot from the current environment."""
    return Settings()


# Module-level convenience instance.
settings = get_settings()
