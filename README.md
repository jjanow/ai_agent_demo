# Aria — a Mini Claude

A production-quality, tool-using AI agent built on the Anthropic API and
orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph). Aria
reasons step by step, calls tools (web search, calculator, file I/O, datetime,
recursive summarization), and answers through a streaming Streamlit chat UI.

## What it does

- Multi-step **agentic loop**: the model reasons, optionally calls one or more
  tools, reads their results, and loops until it has a final answer.
- **Six tools** with full JSON schemas, input validation, structured results,
  and per-tool timeouts: `web_search`, `calculator`, `get_datetime`,
  `read_file`, `write_file`, `summarize_text`.
- **Robust loop** handling: a hard cap on LLM calls per turn, API timeout/rate-
  limit retries with backoff, `max_tokens` continuation, malformed/empty tool
  result handling, and simultaneous (parallel) tool calls.
- **Token-budgeted memory** with JSON persistence per session.
- **Safety guards**: input sanitization, secret redaction, path-traversal
  rejection, and a calculator expression whitelist.
- **Structured logging** to console and a dated file via loguru.
- **Streaming Streamlit UI** with a dark theme, tool-call expanders, and live
  token/iteration metrics.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              Streamlit UI                │
                        │   (main.py: bubbles, tool expanders,     │
                        │    streaming, sidebar metrics)           │
                        └───────────────────┬─────────────────────┘
                                            │ run(user_input)
                                            ▼
        ┌───────────────────────────  LangGraph (agent/loop.py)  ──────────────────────────┐
        │                                                                                   │
        │   START                                                                           │
        │     │                                                                             │
        │     ▼                                                                             │
        │  ┌────────────┐     ┌──────────────┐  tool_use   ┌────────────────────┐           │
        │  │ input_node │────▶│ reasoning_   │────────────▶│ tool_dispatch_node │           │
        │  │ sanitize   │     │ node         │             │ run all tools      │           │
        │  └────────────┘     │ (LLM call)   │             │ (timeouts)         │           │
        │                     └──────┬───────┘             └─────────┬──────────┘           │
        │                            │ text / limit                  │                      │
        │                            │ reached                       ▼                      │
        │                            ▼                     ┌────────────────────┐           │
        │                     ┌────────────┐               │ tool_result_node   │           │
        │                     │ output_node│               │ inject results     │           │
        │                     └──────┬─────┘               └─────────┬──────────┘           │
        │                            │                               │ loop back            │
        │                            ▼                               └──────────────────────┘
        │                           END                                                      │
        └────────────────────────────────────────────────────────────────────────────────────┘
                 │                         │                          │
                 ▼                         ▼                          ▼
        ┌────────────────┐       ┌──────────────────┐       ┌──────────────────┐
        │ Conversation-  │       │   tools.py       │       │  utils/          │
        │ Memory         │       │ web_search,      │       │  logging.py      │
        │ (token budget, │       │ calculator,      │       │  safety.py       │
        │  persistence)  │       │ file I/O, ...    │       │                  │
        └────────────────┘       └──────────────────┘       └──────────────────┘
```

## Project layout

```
.
  main.py              # Streamlit entry point
  config.py            # Env-driven configuration (dotenv)
  conftest.py          # Makes packages importable under pytest
  agent/
    loop.py            # Core LangGraph agent loop
    tools.py           # Tool definitions, schemas, dispatch
    memory.py          # Conversation history management
    prompts.py         # System prompt (Aria)
  utils/
    logging.py         # Loguru setup + structured log helpers
    safety.py          # Input/output and tool guards
  tests/
    test_loop.py
    test_tools.py
    test_memory.py
  .env.example
  requirements.txt
  README.md
```

## Setup

```bash
git clone <repo-url>
cd ai_agent_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your keys
```

`.env` keys:

```
ANTHROPIC_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here      # optional; web_search is disabled without it
ANTHROPIC_MODEL=claude-sonnet-4-6
MAX_ITERATIONS=10
TOKEN_BUDGET=80000
LOG_LEVEL=INFO
```

All configuration is read from the environment — nothing is hardcoded.

## How to run

```bash
streamlit run main.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

## How to run tests

```bash
pytest
```

The tests mock the Anthropic and Tavily APIs, so no API keys or network access
are required.

## Notes & limitations

- `web_search` requires a Tavily API key; it degrades gracefully (returns a
  structured error) when the key is absent.
- Per-tool timeouts run work on a worker thread. Because Python threads cannot
  be force-killed, a timed-out tool returns a timeout error while the abandoned
  work may continue briefly in the background.
- Token counting for the memory budget is a cheap `chars / 4` estimate, not a
  true tokenizer count.
- The calculator uses `eval` against a restricted namespace with no builtins and
  a strict character/name whitelist; it is intended for arithmetic only.
