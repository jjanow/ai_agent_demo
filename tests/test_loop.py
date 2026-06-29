"""Tests for the LangGraph agent loop with a mocked Anthropic client."""

from types import SimpleNamespace

import pytest

from agent.loop import Agent, PARTIAL_PREFIX
from agent.memory import ConversationMemory
from agent.prompts import get_system_prompt


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(block_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def make_message(content, stop_reason, input_tokens=10, output_tokens=5):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return make_message([text_block("done")], "end_turn")


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


class AlwaysToolMessages:
    """A client whose every response asks for another calculator tool call."""

    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return make_message(
            [tool_block(f"t{self.calls}", "calculator", {"expression": "1+1"})],
            "tool_use",
        )


class AlwaysToolClient:
    def __init__(self):
        self.messages = AlwaysToolMessages()


def fake_settings(max_iterations=10, token_budget=80_000):
    return SimpleNamespace(
        anthropic_api_key="test-key",
        tavily_api_key="",
        anthropic_model="claude-test",
        max_iterations=max_iterations,
        token_budget=token_budget,
        log_level="INFO",
        api_timeout_s=30,
        tool_timeout_s=10,
        max_input_chars=10_000,
        max_output_tokens=1024,
    )


@pytest.fixture
def memory(tmp_path):
    return ConversationMemory(
        system_prompt=get_system_prompt(), token_budget=80_000, persist_dir=tmp_path
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_single_turn_no_tools(memory):
    client = FakeClient([make_message([text_block("Hello, I am Aria.")], "end_turn")])
    agent = Agent(memory=memory, settings=fake_settings(), client=client)
    result = agent.run("hi")

    assert result["text"] == "Hello, I am Aria."
    assert result["iterations"] == 1
    assert result["partial"] is False
    assert client.messages.calls == 1
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5


def test_single_tool_call_and_result(memory):
    client = FakeClient(
        [
            make_message(
                [tool_block("t1", "calculator", {"expression": "2+2"})], "tool_use"
            ),
            make_message([text_block("The answer is 4.")], "end_turn"),
        ]
    )
    agent = Agent(memory=memory, settings=fake_settings(), client=client)
    result = agent.run("what is 2+2?")

    assert result["text"] == "The answer is 4."
    assert result["iterations"] == 2
    assert client.messages.calls == 2

    # A tool_result was injected into history.
    history = memory.get_history()
    tool_results = [
        b
        for msg in history
        if isinstance(msg["content"], list)
        for b in msg["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert "4" in tool_results[0]["content"]


def test_max_iteration_cap(memory):
    client = AlwaysToolClient()
    agent = Agent(memory=memory, settings=fake_settings(max_iterations=3), client=client)
    result = agent.run("loop forever")

    assert result["partial"] is True
    assert result["iterations"] == 3
    assert client.messages.calls == 3
    assert result["text"].startswith(PARTIAL_PREFIX)


def test_multiple_tool_use_blocks_in_one_response(memory):
    client = FakeClient(
        [
            make_message(
                [
                    tool_block("t1", "calculator", {"expression": "2+2"}),
                    tool_block("t2", "get_datetime", {}),
                ],
                "tool_use",
            ),
            make_message([text_block("Both done.")], "end_turn"),
        ]
    )
    agent = Agent(memory=memory, settings=fake_settings(), client=client)
    result = agent.run("calc and time")

    assert result["text"] == "Both done."
    assert result["iterations"] == 2

    # Both tool results should be present, batched into one user message.
    history = memory.get_history()
    tool_result_msgs = [
        msg
        for msg in history
        if isinstance(msg["content"], list)
        and all(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in msg["content"]
        )
    ]
    assert len(tool_result_msgs) == 1
    assert len(tool_result_msgs[0]["content"]) == 2


def test_malformed_tool_use_is_filtered_and_recovers(memory):
    # First response's ONLY block is a malformed tool_use (name=None). With the
    # filtering fix it is dropped before persistence, routing returns
    # "continue", and the agent re-calls the client to recover.
    bad_block = SimpleNamespace(type="tool_use", id="t1", name=None, input={})
    client = FakeClient(
        [
            make_message([bad_block], "tool_use"),
            make_message([text_block("Recovered.")], "end_turn"),
        ]
    )
    agent = Agent(memory=memory, settings=fake_settings(), client=client)
    result = agent.run("trigger malformed")

    assert result["text"] == "Recovered."
    # The client was called twice: malformed response, then recovery.
    assert client.messages.calls == 2

    history = memory.get_history()
    all_blocks = [
        b
        for msg in history
        if isinstance(msg["content"], list)
        for b in msg["content"]
        if isinstance(b, dict)
    ]
    # No tool_result should have been injected for the filtered call.
    tool_results = [b for b in all_blocks if b.get("type") == "tool_result"]
    assert tool_results == []
    # The malformed tool_use must not appear in history either.
    malformed = [
        b
        for b in all_blocks
        if b.get("type") == "tool_use" and not b.get("name")
    ]
    assert malformed == []


def test_max_tokens_text_is_accumulated(memory):
    # A max_tokens truncation followed by an end_turn continuation should
    # accumulate both text segments into the final answer.
    client = FakeClient(
        [
            make_message([text_block("part one ")], "max_tokens"),
            make_message([text_block("part two")], "end_turn"),
        ]
    )
    agent = Agent(memory=memory, settings=fake_settings(), client=client)
    result = agent.run("write a long answer")

    assert client.messages.calls == 2
    assert "part one" in result["text"]
    assert "part two" in result["text"]


def test_valid_tool_use_always_gets_tool_result(memory):
    # A well-formed tool_use followed by a recovery text turn must produce
    # exactly one tool_result for the single tool_use.
    client = FakeClient(
        [
            make_message(
                [tool_block("t1", "calculator", {"expression": "2+2"})], "tool_use"
            ),
            make_message([text_block("All done.")], "end_turn"),
        ]
    )
    agent = Agent(memory=memory, settings=fake_settings(), client=client)
    result = agent.run("please compute")

    assert result["text"] == "All done."
    assert client.messages.calls == 2

    history = memory.get_history()
    tool_uses = [
        b
        for msg in history
        if isinstance(msg["content"], list)
        for b in msg["content"]
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    tool_results = [
        b
        for msg in history
        if isinstance(msg["content"], list)
        for b in msg["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert len(tool_uses) == 1
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "t1"


def test_input_rejected(memory):
    client = FakeClient([make_message([text_block("unused")], "end_turn")])
    agent = Agent(memory=memory, settings=fake_settings(), client=client)
    # Input with >50% non-printable characters.
    result = agent.run("\x07\x08\x0b\x0c\x0e\x0f\x10\x11")

    assert "rejected" in result["text"].lower()
    assert client.messages.calls == 0
