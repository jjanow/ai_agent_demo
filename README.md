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
- **Safety guards**: input sanitization, secret redaction (including streamed
  output and tool expanders), path-traversal rejection, sensitive-file denylist
  for file tools, and an AST-based calculator with exponent caps.
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
    test_safety.py
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

# Optional operational overrides (defaults shown)
# API_TIMEOUT_S=30
# TOOL_TIMEOUT_S=10
# MAX_INPUT_CHARS=10000
# MAX_OUTPUT_TOKENS=4096
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
- Per-tool timeouts run each tool in an isolated child process (default start
  method `spawn`). On timeout the process is terminated (SIGTERM, then SIGKILL),
  so a runaway tool's CPU/memory work is actually reclaimed rather than leaked.
  The trade-off is a small per-call process-startup cost.
- Token counting for the memory budget is a cheap `chars / 4` estimate, not a
  true tokenizer count.
- The calculator parses expressions with an AST whitelist (no `eval`/`exec`),
  permitting only arithmetic operators and a fixed set of math functions
  (`sqrt`, `sin`, `cos`, `tan`, `log`, `pow`, `abs`) and constants (`pi`, `e`);
  exponents are capped to avoid CPU/memory exhaustion.
- The file tools are restricted to the working directory and additionally deny
  dotfiles/dotdirs (e.g. `.env`, `.git/`) and private-key/cert files. Output and
  tool-result displays are scrubbed for anything resembling an API key.
