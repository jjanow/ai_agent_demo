"""Conversation history management with a token budget and JSON persistence.

Messages are stored in Anthropic Messages API format (a list of
{"role", "content"} dicts). The system prompt is stored separately because the
Anthropic API takes `system` as a top-level parameter, not as a message; it is
therefore never trimmed.

Token usage is estimated cheaply as len(chars)/4. When the estimate exceeds the
configured budget, the oldest non-system messages are dropped until under
budget, taking care not to leave an orphaned tool_result as the first message.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

_CHARS_PER_TOKEN = 4


class ConversationMemory:
    def __init__(
        self,
        system_prompt: str = "",
        token_budget: int = 80_000,
        session_id: str | None = None,
        persist_dir: str | Path = "sessions",
    ) -> None:
        self.system_prompt = system_prompt
        self.token_budget = token_budget
        self.session_id = session_id or str(uuid.uuid4())
        self.persist_dir = Path(persist_dir)
        self.messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Mutators
    # ------------------------------------------------------------------ #

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._enforce_budget()

    def add_assistant(self, content: Any) -> None:
        """Add an assistant message. `content` may be a string or list of blocks."""
        self.messages.append({"role": "assistant", "content": content})
        self._enforce_budget()

    def add_tool_result(
        self, tool_use_id: str, content: Any, is_error: bool = False
    ) -> None:
        """Add a tool_result block, batching consecutive results into one user msg.

        Anthropic expects tool_result blocks to live in a user message that
        immediately follows the assistant tool_use turn. Multiple results may
        share a single user message, so we append to the trailing tool-result
        user message when one exists.
        """
        block = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content if isinstance(content, str) else json.dumps(content),
        }
        if is_error:
            block["is_error"] = True

        if self.messages and self._is_tool_result_message(self.messages[-1]):
            self.messages[-1]["content"].append(block)
        else:
            self.messages.append({"role": "user", "content": [block]})
        self._enforce_budget()

    def clear(self) -> None:
        self.messages = []

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the message list (Anthropic format)."""
        return list(self.messages)

    def estimate_tokens(self) -> int:
        total_chars = len(self.system_prompt)
        for msg in self.messages:
            total_chars += self._content_chars(msg.get("content"))
        return total_chars // _CHARS_PER_TOKEN

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def export_json(self) -> str:
        return json.dumps(
            {
                "session_id": self.session_id,
                "system_prompt": self.system_prompt,
                "token_budget": self.token_budget,
                "messages": self.messages,
            },
            indent=2,
        )

    def import_json(self, data: str) -> None:
        payload = json.loads(data)
        self.session_id = payload.get("session_id", self.session_id)
        self.system_prompt = payload.get("system_prompt", self.system_prompt)
        self.token_budget = payload.get("token_budget", self.token_budget)
        self.messages = payload.get("messages", [])

    def save(self) -> Path:
        """Persist the session to persist_dir/<session_id>.json."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        path = self.persist_dir / f"{self.session_id}.json"
        path.write_text(self.export_json(), encoding="utf-8")
        return path

    def load(self, session_id: str) -> bool:
        path = self.persist_dir / f"{session_id}.json"
        if not path.exists():
            return False
        self.import_json(path.read_text(encoding="utf-8"))
        return True

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_tool_result_message(msg: dict[str, Any]) -> bool:
        content = msg.get("content")
        return (
            msg.get("role") == "user"
            and isinstance(content, list)
            and all(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            )
        )

    @classmethod
    def _content_chars(cls, content: Any) -> int:
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, str):
                    total += len(block)
                elif isinstance(block, dict):
                    total += len(json.dumps(block, default=str))
                else:
                    total += len(str(block))
            return total
        return len(str(content))

    def _enforce_budget(self) -> None:
        """Drop oldest non-system messages until under the token budget."""
        if self.estimate_tokens() <= self.token_budget:
            return

        trimmed = 0
        while self.messages and self.estimate_tokens() > self.token_budget:
            self.messages.pop(0)
            trimmed += 1

        # Avoid leaving a dangling tool_result as the first message.
        while self.messages and self._is_tool_result_message(self.messages[0]):
            self.messages.pop(0)
            trimmed += 1

        if trimmed:
            logger.debug(
                "Trimmed {} oldest message(s) to respect token budget ({}).",
                trimmed,
                self.token_budget,
            )
