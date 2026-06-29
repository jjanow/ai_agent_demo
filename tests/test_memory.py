"""Tests for agent.memory.ConversationMemory."""

from agent.memory import ConversationMemory


def test_add_and_retrieve():
    mem = ConversationMemory(system_prompt="sys", token_budget=80_000)
    mem.add_user("hello")
    mem.add_assistant("hi there")
    history = mem.get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "assistant", "content": "hi there"}


def test_tool_results_batch_into_one_user_message():
    mem = ConversationMemory(system_prompt="sys")
    mem.add_user("do stuff")
    mem.add_assistant(
        [
            {"type": "tool_use", "id": "a", "name": "calculator", "input": {}},
            {"type": "tool_use", "id": "b", "name": "get_datetime", "input": {}},
        ]
    )
    mem.add_tool_result("a", {"value": 1})
    mem.add_tool_result("b", {"value": 2})
    history = mem.get_history()
    # The two tool results should share a single user message.
    assert history[-1]["role"] == "user"
    assert len(history[-1]["content"]) == 2
    assert history[-1]["content"][0]["type"] == "tool_result"


def test_token_trimming_triggers():
    mem = ConversationMemory(system_prompt="system", token_budget=100)
    # Each message ~250 chars -> ~62 tokens; a few exceed the 100-token budget.
    for i in range(10):
        mem.add_user("x" * 250)
        mem.add_assistant("y" * 250)
    assert mem.estimate_tokens() <= 100
    # System prompt is preserved regardless of trimming.
    assert mem.system_prompt == "system"
    # Some messages were dropped.
    assert len(mem.get_history()) < 20


def test_system_message_never_trimmed():
    big_system = "S" * 1000
    mem = ConversationMemory(system_prompt=big_system, token_budget=10)
    mem.add_user("hello")
    # Budget is tiny but system prompt stays intact.
    assert mem.system_prompt == big_system


def test_export_import_round_trip():
    mem = ConversationMemory(system_prompt="sys", token_budget=80_000)
    mem.add_user("hello")
    mem.add_assistant("world")
    exported = mem.export_json()

    restored = ConversationMemory()
    restored.import_json(exported)
    assert restored.session_id == mem.session_id
    assert restored.system_prompt == "sys"
    assert restored.get_history() == mem.get_history()


def test_save_and_load(tmp_path):
    mem = ConversationMemory(system_prompt="sys", persist_dir=tmp_path)
    mem.add_user("hello")
    path = mem.save()
    assert path.exists()

    other = ConversationMemory(persist_dir=tmp_path)
    assert other.load(mem.session_id) is True
    assert other.get_history() == mem.get_history()


def test_clear():
    mem = ConversationMemory(system_prompt="sys")
    mem.add_user("hello")
    mem.clear()
    assert mem.get_history() == []


def test_trimming_keeps_valid_conversation_start():
    mem = ConversationMemory(system_prompt="system", token_budget=260)
    mem.add_user("x" * 250)
    mem.add_assistant(
        [{"type": "tool_use", "id": "a", "name": "calculator", "input": {}}]
    )
    mem.add_tool_result("a", {"value": 1})
    # More turns force trimming; ending on a user message means the surviving
    # window starts with a leading assistant message that must be repaired away.
    for _ in range(5):
        mem.add_assistant("z" * 250)
        mem.add_user("y" * 250)

    history = mem.get_history()
    assert history
    first = history[0]
    assert first.get("role") == "user"
    assert not mem._is_tool_result_message(first)


def test_load_returns_false_on_malformed_json(tmp_path):
    mem = ConversationMemory(persist_dir=tmp_path)
    (tmp_path / f"{mem.session_id}.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    assert mem.load(mem.session_id) is False
