"""Streamlit chat UI for the Mini-Claude (Aria) agent.

Run with:  streamlit run main.py

Features:
  * Dark-themed chat with user/assistant bubbles.
  * Sidebar: session id, running token usage, last-turn iteration count,
    clear & export buttons.
  * Tool calls rendered as collapsible expanders under the assistant turn.
  * Streaming output via the Anthropic streaming API.
  * "Thinking... (iteration N)" status while the agent loops.
"""

from __future__ import annotations

import streamlit as st

from config import get_settings
from agent.loop import Agent
from agent.memory import ConversationMemory
from agent.prompts import get_system_prompt
from utils.logging import setup_logging
from utils.safety import redact_secrets

st.set_page_config(page_title="Aria — Mini Claude", page_icon="*", layout="wide")

# ----------------------------- styling -------------------------------------- #
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    section[data-testid="stSidebar"] { background-color: #161a23; }
    .stChatMessage { border-radius: 12px; }
    code { color: #9ecbff; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(
            system_prompt=get_system_prompt(), token_budget=settings.token_budget
        )
    if "turns" not in st.session_state:
        # Each turn: {"role": "user"|"assistant", "text": str, "tools": [..]}
        st.session_state.turns = []
    if "total_input_tokens" not in st.session_state:
        st.session_state.total_input_tokens = 0
    if "total_output_tokens" not in st.session_state:
        st.session_state.total_output_tokens = 0
    if "last_iterations" not in st.session_state:
        st.session_state.last_iterations = 0


def _redact_obj(obj):
    """Recursively redact secrets from every string in a dict/list/str structure.

    Returns a new structure; the original is left untouched. Tool inputs and
    results can contain file contents or leaked keys, so redact before display.
    """
    if isinstance(obj, str):
        return redact_secrets(obj)
    if isinstance(obj, dict):
        return {k: _redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_obj(v) for v in obj]
    return obj


def _render_tool_expanders(tools: list[dict]) -> None:
    for call in tools:
        name = call.get("name", "tool")
        ok = call.get("result", {}).get("success", False)
        label = f"{'✓' if ok else '✗'} tool: {name}"
        with st.expander(label, expanded=False):
            st.markdown("**Input**")
            st.json(_redact_obj(call.get("input", {})))
            st.markdown("**Result**")
            st.json(_redact_obj(call.get("result", {})))


def _render_history() -> None:
    for turn in st.session_state.turns:
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            if turn.get("tools"):
                _render_tool_expanders(turn["tools"])


def _sidebar() -> None:
    settings = get_settings()
    mem: ConversationMemory = st.session_state.memory
    with st.sidebar:
        st.title("Aria")
        st.caption("A mini Claude with tool use")
        st.divider()

        st.markdown("**Session**")
        st.code(mem.session_id, language=None)
        st.markdown(f"**Model:** `{settings.anthropic_model}`")

        st.divider()
        st.markdown("**Token usage (running)**")
        col1, col2 = st.columns(2)
        col1.metric("Input", st.session_state.total_input_tokens)
        col2.metric("Output", st.session_state.total_output_tokens)
        st.metric("Last-turn iterations", st.session_state.last_iterations)
        st.caption(f"History estimate: ~{mem.estimate_tokens()} tokens")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            mem.clear()
            st.session_state.turns = []
            st.session_state.total_input_tokens = 0
            st.session_state.total_output_tokens = 0
            st.session_state.last_iterations = 0
            st.rerun()

        st.download_button(
            "Export conversation",
            data=mem.export_json(),
            file_name=f"aria_session_{mem.session_id}.json",
            mime="application/json",
            use_container_width=True,
        )

        if not settings.anthropic_api_key:
            st.warning("ANTHROPIC_API_KEY is not set. Add it to .env to chat.")


def main() -> None:
    _init_state()
    _sidebar()

    st.title("Chat with Aria")
    _render_history()

    prompt = st.chat_input("Ask Aria something...")
    if not prompt:
        return

    # Render the user's message immediately.
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.turns.append({"role": "user", "text": prompt, "tools": []})

    settings = get_settings()
    if not settings.anthropic_api_key:
        with st.chat_message("assistant"):
            st.error("Cannot run: ANTHROPIC_API_KEY is not configured.")
        st.session_state.turns.append(
            {"role": "assistant", "text": "(no API key configured)", "tools": []}
        )
        return

    with st.chat_message("assistant"):
        status = st.status("Thinking... (iteration 1)", expanded=False)
        text_placeholder = st.empty()
        collected_tools: list[dict] = []
        streamed: dict[str, str] = {"text": ""}

        def on_text(delta: str) -> None:
            streamed["text"] += delta
            text_placeholder.markdown(streamed["text"])

        def on_tool(name: str, tool_input: dict, result: dict) -> None:
            collected_tools.append(
                {"name": name, "input": tool_input, "result": result}
            )

        def on_iteration(n: int) -> None:
            status.update(label=f"Thinking... (iteration {n})")

        agent = Agent(
            memory=st.session_state.memory,
            settings=settings,
            on_text=on_text,
            on_tool=on_tool,
            on_iteration=on_iteration,
        )

        result = agent.run(prompt)
        status.update(label=f"Done in {result['iterations']} iteration(s)", state="complete")

        final_text = result["text"] or "(no response)"
        text_placeholder.markdown(final_text)
        if collected_tools:
            _render_tool_expanders(collected_tools)

    st.session_state.turns.append(
        {"role": "assistant", "text": final_text, "tools": collected_tools}
    )
    st.session_state.total_input_tokens += result["input_tokens"]
    st.session_state.total_output_tokens += result["output_tokens"]
    st.session_state.last_iterations = result["iterations"]


if __name__ == "__main__":
    main()
