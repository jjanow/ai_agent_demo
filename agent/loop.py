"""Core agent loop built with LangGraph.

Graph topology:

    input_node -> reasoning_node
    reasoning_node -(tool_use)-> tool_dispatch_node -> tool_result_node -> reasoning_node
    reasoning_node -(text / limit reached)-> output_node -> END

The loop is resilient to the documented edge cases: a hard cap on LLM calls per
turn, empty/malformed tool results, API timeouts and rate limits (with backoff),
max_tokens truncation, per-tool timeouts, and multiple simultaneous tool_use
blocks in a single response.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, TypedDict

from loguru import logger

from config import Settings, get_settings
from utils.logging import log_iteration, log_llm_call, setup_logging
from utils.safety import redact_secrets, sanitize_input

from .memory import ConversationMemory
from .prompts import get_system_prompt
from .tools import dispatch_tool, get_tool_schemas

# LangGraph
from langgraph.graph import StateGraph, START, END

# Anthropic error types (imported lazily-safe at module load).
try:
    from anthropic import (
        Anthropic,
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
    )
except Exception:  # pragma: no cover - anthropic should be installed
    Anthropic = None  # type: ignore
    APITimeoutError = APIConnectionError = RateLimitError = Exception  # type: ignore


PARTIAL_PREFIX = "I've exceeded my reasoning limit. Here's what I have so far: "


class AgentState(TypedDict, total=False):
    user_input: str
    iteration: int
    exceeded: bool
    pending_tool_calls: list[dict]
    stop_reason: Optional[str]
    last_text: str
    final_text: Optional[str]
    partial: bool
    input_tokens_total: int
    output_tokens_total: int
    error: Optional[str]
    _tool_results: list[dict]


class Agent:
    """Mini-Claude agent. Owns the LLM client, memory, and the compiled graph."""

    def __init__(
        self,
        memory: ConversationMemory | None = None,
        settings: Settings | None = None,
        on_text: Callable[[str], None] | None = None,
        on_tool: Callable[[str, dict, dict], None] | None = None,
        on_iteration: Callable[[int], None] | None = None,
        client: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        setup_logging(self.settings.log_level)

        if client is not None:
            self.client = client
        elif Anthropic is not None:
            self.client = Anthropic(
                api_key=self.settings.anthropic_api_key or "missing",
                timeout=self.settings.api_timeout_s,
            )
        else:  # pragma: no cover
            self.client = None

        self.memory = memory or ConversationMemory(
            system_prompt=get_system_prompt(),
            token_budget=self.settings.token_budget,
        )
        self.on_text = on_text
        self.on_tool = on_tool
        self.on_iteration = on_iteration
        self.tools = get_tool_schemas()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, user_input: str) -> dict:
        """Run one full user turn and return a result summary dict."""
        initial: AgentState = {
            "user_input": user_input,
            "iteration": 0,
            "exceeded": False,
            "pending_tool_calls": [],
            "stop_reason": None,
            "last_text": "",
            "final_text": None,
            "partial": False,
            "input_tokens_total": 0,
            "output_tokens_total": 0,
            "error": None,
        }
        # Allow enough graph supersteps for the max LLM-call loop.
        final_state = self.graph.invoke(
            initial, config={"recursion_limit": self.settings.max_iterations * 4 + 10}
        )
        try:
            self.memory.save()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist session: {}", exc)

        return {
            "text": final_state.get("final_text") or "",
            "iterations": final_state.get("iteration", 0),
            "partial": final_state.get("partial", False),
            "input_tokens": final_state.get("input_tokens_total", 0),
            "output_tokens": final_state.get("output_tokens_total", 0),
            "error": final_state.get("error"),
        }

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("input_node", self.input_node)
        graph.add_node("reasoning_node", self.reasoning_node)
        graph.add_node("tool_dispatch_node", self.tool_dispatch_node)
        graph.add_node("tool_result_node", self.tool_result_node)
        graph.add_node("output_node", self.output_node)

        graph.add_edge(START, "input_node")
        graph.add_edge("input_node", "reasoning_node")
        graph.add_conditional_edges(
            "reasoning_node",
            self._route_after_reasoning,
            {
                "tools": "tool_dispatch_node",
                "continue": "reasoning_node",
                "output": "output_node",
            },
        )
        graph.add_edge("tool_dispatch_node", "tool_result_node")
        graph.add_edge("tool_result_node", "reasoning_node")
        graph.add_edge("output_node", END)
        return graph.compile()

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def input_node(self, state: AgentState) -> dict:
        """Validate and sanitize the user message, then add it to memory."""
        guard = sanitize_input(
            state.get("user_input", ""), max_chars=self.settings.max_input_chars
        )
        if not guard.ok:
            logger.warning("Input rejected: {}", guard.error)
            return {
                "exceeded": False,
                "final_text": f"Your message was rejected: {guard.error}",
                "error": guard.error,
                "stop_reason": "input_rejected",
            }
        self.memory.add_user(guard.value)
        return {}

    def reasoning_node(self, state: AgentState) -> dict:
        """Call the Anthropic API with the full history and tool schemas."""
        # Short-circuit if a previous node rejected the input.
        if state.get("error") and state.get("stop_reason") == "input_rejected":
            return {"exceeded": True}

        prev_stop = state.get("stop_reason")
        iteration = state.get("iteration", 0)
        if iteration >= self.settings.max_iterations:
            logger.warning("Max iterations ({}) reached.", self.settings.max_iterations)
            return {"exceeded": True, "partial": True}

        iteration += 1
        if self.on_iteration:
            try:
                self.on_iteration(iteration)
            except Exception:  # noqa: BLE001
                pass
        try:
            message = self._call_llm()
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM call failed irrecoverably: {}", exc)
            return {
                "iteration": iteration,
                "final_text": (
                    "I hit a problem talking to the model and couldn't complete "
                    f"this request ({exc})."
                ),
                "error": str(exc),
                "stop_reason": "api_error",
                "exceeded": True,
            }

        stop_reason = getattr(message, "stop_reason", None)
        blocks = self._blocks_to_dicts(getattr(message, "content", []) or [])
        log_iteration(iteration, stop_reason)

        # Drop malformed tool_use blocks (missing a truthy id or name) BEFORE
        # persisting. Otherwise they land in history with no matching
        # tool_result, violating the Anthropic API contract (every tool_use must
        # be answered by a tool_result on the next turn) and 400-ing the next
        # real call. Filtered blocks simply vanish; routing then lets the model
        # recover (see _route_after_reasoning).
        kept_blocks: list[dict] = []
        for b in blocks:
            if b.get("type") == "tool_use" and not (b.get("id") and b.get("name")):
                logger.warning("Filtering malformed tool_use block: {}", b)
                continue
            kept_blocks.append(b)
        blocks = kept_blocks

        # Persist the assistant turn (text and/or well-formed tool_use) verbatim.
        self.memory.add_assistant(blocks)

        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        tool_calls = [b for b in blocks if b.get("type") == "tool_use"]

        usage = getattr(message, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0

        # Accumulate text across consecutive max_tokens continuations so the
        # earlier (truncated) segments survive into the final answer rather than
        # being overwritten by the last segment alone.
        if prev_stop == "max_tokens" and text:
            new_last = (state.get("last_text", "") or "") + text
        else:
            new_last = text or state.get("last_text", "")

        return {
            "iteration": iteration,
            "stop_reason": stop_reason,
            "pending_tool_calls": tool_calls,
            "last_text": new_last,
            "input_tokens_total": state.get("input_tokens_total", 0) + in_tok,
            "output_tokens_total": state.get("output_tokens_total", 0) + out_tok,
        }

    def tool_dispatch_node(self, state: AgentState) -> dict:
        """Execute every pending tool_use block, collecting structured results."""
        tool_calls = state.get("pending_tool_calls", [])
        results: list[dict] = []

        for call in tool_calls:
            name = call.get("name")
            tool_id = call.get("id")
            tool_input = call.get("input")

            # Malformed tool_use: missing name or id -> log and skip.
            if not name or not tool_id:
                logger.warning("Skipping malformed tool_use block: {}", call)
                continue
            if not isinstance(tool_input, dict):
                tool_input = {}

            result = dispatch_tool(name, tool_input, timeout_s=self.settings.tool_timeout_s)
            if self.on_tool:
                try:
                    self.on_tool(name, tool_input, result)
                except Exception:  # noqa: BLE001
                    pass
            results.append({"tool_use_id": tool_id, "name": name, "result": result})

        return {"_tool_results": results}

    def tool_result_node(self, state: AgentState) -> dict:
        """Inject collected tool results back into the conversation history."""
        results = state.get("_tool_results", [])  # type: ignore[arg-type]
        for item in results:
            result = item["result"]
            is_error = not result.get("success", False)
            payload = result.get("result") if result.get("success") else result.get("error")

            # Empty tool result: inject an explicit error, never skip.
            if payload is None or payload == "":
                payload = "Tool returned an empty result."
                is_error = True

            self.memory.add_tool_result(
                tool_use_id=item["tool_use_id"], content=payload, is_error=is_error
            )
        return {"pending_tool_calls": []}

    def output_node(self, state: AgentState) -> dict:
        """Produce the final text answer and redact any leaked secrets."""
        if state.get("final_text"):
            return {"final_text": redact_secrets(state["final_text"])}

        last_text = state.get("last_text", "") or ""
        if state.get("partial"):
            body = last_text or "(no partial answer was produced)"
            final = PARTIAL_PREFIX + body
        else:
            final = last_text or "I don't have a response for that."

        return {"final_text": redact_secrets(final)}

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def _route_after_reasoning(self, state: AgentState) -> str:
        if state.get("exceeded") or state.get("final_text"):
            return "output"
        if state.get("pending_tool_calls"):
            return "tools"
        if state.get("stop_reason") == "max_tokens":
            # Truncated text with no tool call: continue generating.
            logger.info("stop_reason=max_tokens; continuing to let the model finish.")
            return "continue"
        if state.get("stop_reason") == "tool_use":
            # The model asked for tools but every tool_use block was malformed
            # and filtered out in reasoning_node. Rather than emit an empty
            # answer, re-call the LLM so it can recover. The max_iterations cap
            # bounds this loop.
            logger.info(
                "stop_reason=tool_use with no pending tool calls; "
                "continuing to let the model recover."
            )
            return "continue"
        return "output"

    # ------------------------------------------------------------------ #
    # LLM call with retry/backoff
    # ------------------------------------------------------------------ #

    def _call_llm(self):
        """Call Anthropic with retries. Streams text when on_text is provided."""
        history = self.memory.get_history()
        system = self.memory.system_prompt
        model = self.settings.anthropic_model
        max_tokens = self.settings.max_output_tokens

        rate_limit_attempts = 0
        timeout_attempts = 0
        delay = 1.0

        while True:
            start = time.perf_counter()
            try:
                if self.on_text is not None and hasattr(self.client, "messages") and hasattr(
                    self.client.messages, "stream"
                ):
                    message = self._stream_call(system, history, model, max_tokens)
                else:
                    message = self.client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=history,
                        tools=self.tools,
                    )
                latency_ms = (time.perf_counter() - start) * 1000.0
                usage = getattr(message, "usage", None)
                log_llm_call(
                    model=model,
                    input_tokens=getattr(usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(usage, "output_tokens", 0) or 0,
                    latency_ms=latency_ms,
                    stop_reason=getattr(message, "stop_reason", None),
                )
                return message

            except RateLimitError as exc:
                rate_limit_attempts += 1
                if rate_limit_attempts > 3:
                    logger.error("Rate limit retries exhausted.")
                    raise
                logger.warning(
                    "Rate limited (attempt {}/3); backing off {:.1f}s.",
                    rate_limit_attempts,
                    delay,
                )
                time.sleep(delay)
                delay *= 2

            except (APITimeoutError, APIConnectionError) as exc:
                timeout_attempts += 1
                if timeout_attempts > 1:  # retry exactly once
                    logger.error("API timeout after one retry; failing gracefully.")
                    raise
                logger.warning("API timeout; retrying once after {:.1f}s.", delay)
                time.sleep(delay)
                delay *= 2

    def _stream_call(self, system, history, model, max_tokens):
        """Stream a response, emitting text deltas via on_text, return final msg."""
        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=history,
            tools=self.tools,
        ) as stream:
            for text in stream.text_stream:
                if self.on_text:
                    # Best-effort per-chunk redaction of leaked secrets.
                    self.on_text(redact_secrets(text))
            return stream.get_final_message()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _blocks_to_dicts(content: Any) -> list[dict]:
        """Normalize SDK content blocks (or dicts) into Anthropic-format dicts."""
        out: list[dict] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    out.append({"type": "text", "text": block.get("text", "")})
                elif btype == "tool_use":
                    out.append(
                        {
                            "type": "tool_use",
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "input": block.get("input", {}),
                        }
                    )
                continue

            btype = getattr(block, "type", None)
            if btype == "text":
                out.append({"type": "text", "text": getattr(block, "text", "")})
            elif btype == "tool_use":
                out.append(
                    {
                        "type": "tool_use",
                        "id": getattr(block, "id", None),
                        "name": getattr(block, "name", None),
                        "input": getattr(block, "input", {}) or {},
                    }
                )
        return out
