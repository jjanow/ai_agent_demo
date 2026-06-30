# Mini-Claude Agent (Aria) — Complete Project Guide

This document explains every source file, every function, and how the pieces fit together. Use it to study how a production-style LLM agent is built: LangGraph for orchestration, Anthropic for the model, tools with timeouts and safety guards, conversation memory with token budgeting, and a Streamlit UI.

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [Technologies & Concepts Explained](#technologies--concepts-explained)
3. [Architecture Overview](#architecture-overview)
4. [End-to-End Request Flow](#end-to-end-request-flow)
5. [Dependencies](#dependencies)
6. [Configuration (`config.py`)](#configuration-configpy)
7. [Entry Point (`main.py`)](#entry-point-mainpy)
8. [Agent Core (`agent/`)](#agent-core-agent)
9. [Utilities (`utils/`)](#utilities-utils)
10. [Test Suite (`tests/`)](#test-suite-tests)
11. [Design Patterns Worth Learning](#design-patterns-worth-learning)
12. [Environment Variables Reference](#environment-variables-reference)

---

## What This Project Is

**Aria** is a small but complete **tool-using chat agent** modeled after Claude. A user sends a message; the agent may call the LLM multiple times in a loop, invoke tools (web search, calculator, file I/O, etc.), feed tool results back to the model, and finally return a text answer.

Key technologies (see [Technologies & Concepts Explained](#technologies--concepts-explained) for full descriptions):

| Layer | Technology | Role |
|-------|------------|------|
| UI | Streamlit | Chat interface, streaming, sidebar metrics |
| Orchestration | LangGraph | State machine for the agent loop |
| LLM | Anthropic Messages API | Reasoning, tool selection, text generation |
| Web search | Tavily | External information retrieval |
| Config | python-dotenv | Environment-based settings |
| Logging | loguru | Console + rotating file logs |
| Testing | pytest | Automated test runner |

The project is intentionally **small enough to read in an afternoon** but includes patterns you would want in a real agent: retries, iteration caps, malformed-tool recovery, secret redaction, path traversal protection, and isolated tool execution with hard timeouts.

---

## Technologies & Concepts Explained

This section defines every major library, service, and idea referenced in this guide. Read it first if any of the names above are unfamiliar.

### Core ideas

#### LLM (large language model)

A neural network trained on vast amounts of text that predicts the next token in a sequence. Given a conversation history, an LLM can answer questions, follow instructions, and (when given tool definitions) decide which external actions to take. In this project, **Claude** — Anthropic's family of models — is the LLM behind Aria.

#### Tool-using agent / agentic loop

A program that repeatedly asks an LLM what to do next. The model may reply with plain text (a final answer) or with structured **tool calls** — requests to run code such as a web search or calculator. The agent executes those tools, appends the results to the conversation, and asks the LLM again. This **reason → act → observe** cycle continues until the model stops with a text answer or a safety limit (iteration cap, timeout) is reached.

#### State machine / directed graph

A way to model a workflow as **nodes** (steps) and **edges** (transitions). Each node reads shared **state**, does one job, and returns updates. **LangGraph** compiles such a graph so routing logic (e.g. "go to tools or finish?") lives in one place instead of a tangled `while` loop. See [LangGraph Topology](#langgraph-topology) for this project's graph.

#### Streaming

Instead of waiting for the full LLM response, the API can emit **chunks** of text as they are generated. The UI updates in real time (`on_text` callbacks in `main.py`). Streaming improves perceived latency; this project also runs `redact_secrets()` on each chunk so API keys cannot flash on screen mid-stream.

#### Token budget

LLMs have context limits measured in **tokens** (roughly word pieces; this project estimates `chars ÷ 4`). `ConversationMemory` trims the oldest messages when history exceeds `TOKEN_BUDGET` so API calls stay within limits and cost stays bounded.

#### System prompt

A fixed instruction block sent to the model on every API call but **not** stored in the user-visible chat history. It defines persona and rules (e.g. "You are Aria; use tools when needed"). Defined in `agent/prompts.py`.

#### Tool-use contract (`tool_use` / `tool_result` blocks)

Anthropic's Messages API represents tool calls as assistant **content blocks** with `type: "tool_use"` (id, name, input). The client must reply with a user message containing matching **`tool_result`** blocks (same id). Skipping or mismatching ids causes API errors; this project filters malformed blocks and batches multiple results into one user message.

#### JSON Schema

A standard for describing the shape of JSON data (field names, types, required fields). Tool `input_schema` values in `get_tool_schemas()` tell the model what arguments each tool accepts.

#### `.env` files

Plain-text files of `KEY=value` lines, usually gitignored, that hold secrets and local overrides (API keys, timeouts). Loaded at startup by **python-dotenv**; see `.env.example` for the template.

---

### Third-party libraries and services

#### Streamlit

A Python framework for building web UIs with only Python — no HTML/CSS/JavaScript required. You run `streamlit run main.py` and get a local web app. This project uses it for the chat layout (`st.chat_message`, `st.chat_input`), sidebar metrics, tool expanders, session state (`st.session_state`), and streaming placeholders. Streamlit reruns the script top-to-bottom on interaction; `_init_state()` guards one-time setup.

#### LangGraph

A library from LangChain for building **stateful, multi-step workflows** as graphs. You define `StateGraph`, add nodes (functions), wire conditional edges, and call `.compile()` to get an invokable graph. Here it orchestrates `input → reasoning → tools → reasoning → … → output`. Alternatives include a hand-written loop; LangGraph makes routing and visualization explicit.

#### Anthropic / Claude / Messages API

**Anthropic** is the company that builds **Claude**. The **`anthropic`** Python package talks to their **Messages API** — the HTTP interface for multi-turn chat, tool definitions, and streaming. `Agent._call_llm()` sends `system` + `messages` + `tools` and receives content blocks plus `stop_reason` (e.g. `end_turn`, `tool_use`, `max_tokens`). Model id defaults to `claude-sonnet-4-6` in `config.py`.

#### Tavily (`tavily-python`)

A search API aimed at AI agents: you send a query, get back ranked web snippets and URLs. The `web_search` tool uses `TavilyClient.search()` and requires `TAVILY_API_KEY`. It is an alternative to wiring raw Google/Bing APIs or scraping.

#### python-dotenv

Loads variables from a `.env` file into `os.environ` so `config.py` can read `ANTHROPIC_API_KEY` and friends without exporting them in your shell. `load_dotenv()` runs once at import time.

#### loguru

A logging library with simple setup, colored console output, and file rotation. `utils/logging.py` configures console (level from `LOG_LEVEL`) and daily rotating files under `logs/`. Used for structured lines around LLM calls, tool calls, and iterations.

#### pytest

The standard Python test runner. Tests live under `tests/`; `conftest.py` adds the project root to `sys.path`. Run with `pytest` or `pytest tests/test_loop.py`. This project mocks the Anthropic client so tests need no network.

---

### Python standard library used here

#### `multiprocessing`

Runs code in **child processes**. `dispatch_tool()` spawns a worker per tool call so a hung or CPU-heavy tool cannot block the main agent; on timeout the child is terminated (`SIGTERM`, then `SIGKILL`). Uses `spawn` start method for compatibility with Streamlit.

#### `ast` (Abstract Syntax Tree)

Python's parser represents expressions as a tree of nodes. `safe_eval_math()` parses calculator input with `ast.parse`, then walks only allowed node types — a safe alternative to `eval()`, which would execute arbitrary code.

#### `dataclass`, `TypedDict`, `uuid`

- **`dataclass`** — boilerplate-free data containers (`Settings`, `GuardResult`).
- **`TypedDict`** — typed dict shapes for graph state (`AgentState`).
- **`uuid`** — generates unique session ids for `ConversationMemory` and JSON files under `sessions/`.

---

### Other terms you will see in this guide

| Term | Meaning |
|------|---------|
| **Aria** | This project's agent persona and app title — a mini Claude-style assistant. |
| **Content blocks** | Structured pieces in an API message: `text`, `tool_use`, or `tool_result`. |
| **`stop_reason`** | Why the model ended a turn: finished (`end_turn`), wants tools (`tool_use`), or hit length (`max_tokens`). |
| **RAG** | Retrieval-augmented generation — fetch documents, inject into context (not implemented here; mentioned as a possible graph extension). |
| **FastAPI / WebSocket / CLI** | Other UIs that could reuse `Agent` via callbacks without Streamlit. |
| **Rate limiting / backoff** | API may return 429; `_call_llm()` retries with increasing delays. |
| **Path traversal** | Attack using `../` to read files outside the project; blocked by `safe_resolve_path()`. |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           main.py (Streamlit UI)                        │
│  - Renders chat, streams text, shows tool expanders                     │
│  - Owns ConversationMemory in st.session_state                          │
│  - Creates Agent per turn with callbacks (on_text, on_tool, on_iteration) │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ agent.run(user_input)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         agent/loop.py (Agent)                           │
│  LangGraph state machine:                                               │
│    input → reasoning → [tools → results → reasoning]* → output → END    │
└───┬─────────────┬──────────────┬──────────────┬────────────────────────┘
    │             │              │              │
    ▼             ▼              ▼              ▼
 memory.py    prompts.py     tools.py      utils/safety.py
              (system        (6 tools +    (input sanitize,
               prompt)        dispatch)      redaction, paths)
    │
    ▼
 sessions/<uuid>.json   (optional persistence on each run)
```

### LangGraph Topology

The agent loop is a **compiled directed graph** over `AgentState`:

```
START
  │
  ▼
input_node          ← sanitize user message, add to memory
  │
  ▼
reasoning_node      ← call Anthropic API with history + tool schemas
  │
  ├── (exceeded / final_text / end_turn) ──► output_node ──► END
  ├── (pending_tool_calls) ──► tool_dispatch_node
  │                                  │
  │                                  ▼
  │                            tool_result_node  ← inject results into memory
  │                                  │
  └◄─────────────────────────────────┘
  └── (max_tokens / tool_use with no valid tools) ──► reasoning_node (continue)
```

---

## End-to-End Request Flow

Here is what happens when a user types a message in the Streamlit app:

1. **`main.py`** appends the user message to `st.session_state.turns` for display.
2. **`main.py`** constructs an `Agent` with the shared `ConversationMemory` and callback hooks.
3. **`Agent.run()`** builds initial `AgentState` and invokes the compiled LangGraph.
4. **`input_node`** runs `sanitize_input()`; on failure, short-circuits to `output_node` with a rejection message.
5. **`reasoning_node`** increments the iteration counter, calls `_call_llm()`, parses content blocks, filters malformed `tool_use` blocks, persists the assistant turn to memory, and updates token totals.
6. **`_route_after_reasoning`** decides the next step:
   - Tools pending → `tool_dispatch_node`
   - `max_tokens` with no tools → loop back to `reasoning_node` (continue generation)
   - `tool_use` but all blocks were malformed → loop back (recovery)
   - Otherwise → `output_node`
7. **`tool_dispatch_node`** calls `dispatch_tool()` for each pending tool (in separate processes with timeout).
8. **`tool_result_node`** adds `tool_result` blocks to memory (batched into one user message when possible).
9. Loop returns to **`reasoning_node`** until the model stops with text or limits are hit.
10. **`output_node`** builds final text (including partial-answer prefix if iteration cap hit), runs `redact_secrets()`.
11. **`Agent.run()`** calls `memory.save()` and returns a summary dict.
12. **`main.py`** updates token metrics, renders tool expanders, and stores the assistant turn.

---

## Dependencies

Package versions from `requirements.txt`. For what each package *is*, see [Technologies & Concepts Explained](#technologies--concepts-explained).

| Package | Version | Used by |
|---------|---------|---------|
| `anthropic` | 0.112.0 | LLM calls in `loop.py`, `summarize_text` tool |
| `langgraph` | 1.2.6 | Agent state graph in `loop.py` |
| `streamlit` | 1.58.0 | `main.py` UI |
| `tavily-python` | 0.7.26 | `web_search` tool |
| `python-dotenv` | 1.2.2 | `config.py` |
| `loguru` | 0.7.3 | `utils/logging.py`, various modules |
| `pytest` | 9.1.1 | Test suite |

### Project modules (not pip packages)

These are folders in this repo, not installable libraries:

| Module | What it is |
|--------|------------|
| **`agent/`** | The agent brain: LangGraph loop (`loop.py`), conversation memory (`memory.py`), tool implementations (`tools.py`), and system prompt (`prompts.py`). |
| **`utils/`** | Shared helpers: safety guards (`safety.py`) and logging setup (`logging.py`). |
| **`tests/`** | pytest tests that mock the LLM and verify loop, memory, tools, and safety behavior. |
| **`config.py`** | Single place for settings loaded from environment variables. |
| **`main.py`** | Streamlit app — the only user-facing entry point. |
| **`sessions/`** | Created at runtime; JSON files storing conversation history per session id. |
| **`logs/`** | Created at runtime; rotating log files from loguru. |

---

## Configuration (`config.py`)

Central configuration loaded from environment variables (optionally via `.env`). No magic numbers are scattered in the codebase — operational limits live here.

### Module-level

| Name | Type | Purpose |
|------|------|---------|
| `load_dotenv()` | call | Runs at import; loads `.env` from project/cwd |

### `_get_int(name, default) -> int`

Reads an environment variable as an integer. Falls back to parsing `default` if the value is missing or invalid.

### `Settings` (frozen dataclass)

Immutable snapshot of all runtime settings. Fields and their env vars:

| Field | Env var | Default | Meaning |
|-------|---------|---------|---------|
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | `""` | Anthropic API key |
| `tavily_api_key` | `TAVILY_API_KEY` | `""` | Tavily search API key |
| `anthropic_model` | `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model ID |
| `max_iterations` | `MAX_ITERATIONS` | `10` | Max LLM calls per user turn |
| `token_budget` | `TOKEN_BUDGET` | `80000` | Conversation history token budget |
| `log_level` | `LOG_LEVEL` | `INFO` | Console log level |
| `api_timeout_s` | `API_TIMEOUT_S` | `30` | Anthropic client timeout (seconds) |
| `tool_timeout_s` | `TOOL_TIMEOUT_S` | `10` | Per-tool execution timeout |
| `max_input_chars` | `MAX_INPUT_CHARS` | `10000` | Max user message length |
| `max_output_tokens` | `MAX_OUTPUT_TOKENS` | `4096` | Max tokens per LLM response |

#### `Settings.require_anthropic() -> str`

Raises `RuntimeError` if `anthropic_api_key` is empty; otherwise returns the key. Useful for strict startup checks (not used by `main.py`, which shows a UI warning instead).

### `get_settings() -> Settings`

Builds a fresh `Settings` instance from the current environment. Called when constructing `Agent` and in various tools.

### `settings`

Module-level convenience instance: `settings = get_settings()`.

---

## Entry Point (`main.py`)

Streamlit chat UI. Run with: `streamlit run main.py`

### Page setup

- `st.set_page_config` — title "Aria — Mini Claude", wide layout.
- Inline CSS — dark theme for app background, sidebar, and chat bubbles.

### `_init_state() -> None`

Initializes Streamlit session state on first load:

| Key | Type | Purpose |
|-----|------|---------|
| `memory` | `ConversationMemory` | Shared conversation history across turns |
| `turns` | `list[dict]` | UI display history: `{role, text, tools}` |
| `total_input_tokens` | `int` | Running input token count |
| `total_output_tokens` | `int` | Running output token count |
| `last_iterations` | `int` | LLM iterations from the last turn |

Also calls `setup_logging()` and `get_settings()`.

### `_redact_obj(obj)`

Recursively walks dicts, lists, and strings, applying `redact_secrets()` to every string. Used before displaying tool inputs/results in the UI (file contents might contain leaked keys).

### `_render_tool_expanders(tools: list[dict]) -> None`

For each tool call in a turn, renders a collapsible Streamlit expander with ✓/✗ status, JSON input, and JSON result (redacted).

### `_render_history() -> None`

Iterates `st.session_state.turns` and renders each as a `st.chat_message` with optional tool expanders.

### `_sidebar() -> None`

Sidebar panel showing:

- Session ID (`memory.session_id`)
- Model name
- Token usage metrics (input, output, last-turn iterations, history estimate)
- **Clear conversation** — resets memory and UI state, reruns app
- **Export conversation** — downloads `memory.export_json()` as a file
- Warning if `ANTHROPIC_API_KEY` is missing

### `main() -> None`

Main UI loop:

1. `_init_state()`, `_sidebar()`, `_render_history()`
2. Waits for `st.chat_input`
3. Immediately shows user message in chat and appends to `turns`
4. If no API key → shows error assistant message, returns
5. Creates assistant chat area with:
   - `st.status` — "Thinking... (iteration N)" updated via `on_iteration`
   - `text_placeholder` — streaming markdown via `on_text`
   - `collected_tools` — populated via `on_tool`
6. Constructs `Agent(memory=..., settings=..., callbacks...)`
7. Calls `agent.run(prompt)`, updates status to "Done in N iteration(s)"
8. Renders final text and tool expanders; updates session metrics

### `if __name__ == "__main__"`

Calls `main()`.

---

## Agent Core (`agent/`)

### `agent/__init__.py`

Empty package marker.

---

### `agent/prompts.py`

Defines the **system prompt** that shapes agent behavior.

#### `SYSTEM_PROMPT` (constant string)

Instructs the model to:

- Act as **Aria** — concise, honest, no fabrication
- Use tools deliberately: `web_search`, `calculator`, `get_datetime`, `read_file`, `write_file`, `summarize_text`
- Think before acting; call multiple tools when needed
- Stop calling tools when the task is complete

#### `get_system_prompt() -> str`

Returns `SYSTEM_PROMPT`. Used by `Agent` and `main.py` when creating `ConversationMemory`.

---

### `agent/memory.py`

Manages conversation history in **Anthropic Messages API format** and enforces a token budget.

#### Constants

| Name | Value | Purpose |
|------|-------|---------|
| `_CHARS_PER_TOKEN` | `4` | Cheap token estimate: `chars // 4` |

#### `ConversationMemory`

##### `__init__(system_prompt="", token_budget=80000, session_id=None, persist_dir="sessions")`

- `system_prompt` — passed separately to the API (not in `messages`)
- `token_budget` — max estimated tokens for history trimming
- `session_id` — UUID string, auto-generated if omitted
- `persist_dir` — directory for JSON session files
- `messages` — list of `{"role", "content"}` dicts

##### `add_user(text: str) -> None`

Appends `{"role": "user", "content": text}` and calls `_enforce_budget()`.

##### `add_assistant(content) -> None`

Appends assistant message. `content` may be a plain string or a list of content blocks (text + tool_use). Trims if over budget.

##### `add_tool_result(tool_use_id, content, is_error=False) -> None`

Adds a `tool_result` block. **Batches** consecutive results into the trailing user message if it already contains only `tool_result` blocks (Anthropic API convention). Non-string `content` is JSON-serialized. Sets `is_error: true` on the block when `is_error=True`.

##### `clear() -> None`

Empties `messages`.

##### `get_history() -> list[dict]`

Shallow copy of `messages` for API calls.

##### `estimate_tokens() -> int`

Sums character counts of system prompt + all message content, divides by 4.

##### `export_json() -> str`

Serializes session to JSON: `session_id`, `system_prompt`, `token_budget`, `messages`.

##### `import_json(data: str) -> None`

Parses JSON and restores fields. Raises `ValueError` on invalid data.

##### `save() -> Path`

Writes `persist_dir/<session_id>.json`. Creates directory if needed.

##### `load(session_id: str) -> bool`

Loads session from disk. Returns `False` if file missing or corrupt.

##### `_is_tool_result_message(msg) -> bool` (static)

True if message is a user message whose content is exclusively `tool_result` blocks.

##### `_content_chars(content) -> int` (classmethod)

Estimates character count for string, list of blocks, or other content.

##### `_enforce_budget() -> None`

When `estimate_tokens() > token_budget`:

1. Pop oldest messages until under budget (always keeps at least one)
2. **Repair** the front: drop leading assistant messages or orphaned `tool_result`-only user messages (Anthropic requires history to start with a valid user message)

---

### `agent/tools.py`

Tool implementations, JSON schemas for the Anthropic API, and **process-isolated dispatch** with timeouts.

#### Result helpers

| Function | Returns | Purpose |
|----------|---------|---------|
| `_ok(result)` | `{success: True, result, error: None}` | Success wrapper |
| `_err(message)` | `{success: False, result: None, error}` | Error wrapper |

#### Individual tools

Every tool **never raises** to the caller — exceptions become `_err(...)`.

##### `web_search(query: str) -> dict`

- Validates non-empty string query
- Requires `TAVILY_API_KEY`
- Uses `TavilyClient.search(max_results=3)`
- Returns top 3 results: `title`, `snippet` (max 500 chars), `url`

##### `calculator(expression: str) -> dict`

- Delegates to `safe_eval_math(expression)` from `utils.safety`
- Returns computed `value` or error message

##### `get_datetime() -> dict`

- Returns current UTC time: `iso`, `utc`, `timezone`, `unix` timestamp

##### `read_file(path: str) -> dict`

- Resolves path via `safe_resolve_path()` (cwd-relative only)
- Reads UTF-8 text, truncates at 100,000 characters
- Returns `path` and `content`

##### `write_file(path, content, confirm_overwrite=False) -> dict`

- Validates `content` is a string
- Resolves path safely
- Refuses overwrite unless `confirm_overwrite=True`
- Creates parent directories, writes UTF-8
- Returns `path` and `bytes_written`

##### `summarize_text(text: str) -> dict`

- Makes a **nested LLM call** to Anthropic (separate from the main agent loop)
- Truncates input to 50,000 chars, asks for ≤5 sentence summary
- Requires `ANTHROPIC_API_KEY`

#### Registry

```python
_TOOL_FUNCS = {
    "web_search": web_search,
    "calculator": calculator,
    "get_datetime": get_datetime,
    "read_file": read_file,
    "write_file": write_file,
    "summarize_text": summarize_text,
}
```

#### `get_tool_schemas() -> list[dict]`

Returns Anthropic-compatible tool definitions: `name`, `description`, `input_schema` (JSON Schema) for each tool. Passed to `client.messages.create(tools=...)`.

#### `_summarize_result(result: dict) -> str`

Short string for logging: `"ok: ..."` or `"error: ..."` (truncated to 200 chars).

#### Process isolation constants

| Name | Value | Purpose |
|------|-------|---------|
| `MP_START_METHOD` | `"spawn"` | Safe default for Streamlit (no inherited locks) |
| `_REAP_GRACE_S` | `1.0` | Wait before SIGKILL on timeout |

#### `_tool_worker(name, tool_input, out_queue) -> None`

Child process entry point. Looks up tool by name in `_TOOL_FUNCS`, calls `func(**tool_input)`, puts result on queue. Catches all exceptions.

#### `_reap(process) -> None`

Terminates a worker: `terminate()` → wait → `kill()` if still alive.

#### `dispatch_tool(name, tool_input, timeout_s=None) -> dict`

Main dispatch path:

1. Fast-path: unknown tool or non-dict input → error without spawning
2. Spawns child process via `multiprocessing.get_context(MP_START_METHOD)`
3. Waits on queue with `timeout_s` (default from settings)
4. On timeout → error, kills process via `_reap`
5. Validates result shape, logs via `log_tool_call`
6. **Never raises**

---

### `agent/loop.py`

The heart of the project: **`Agent`** class and LangGraph state machine.

#### Constants

| Name | Value | Purpose |
|------|-------|---------|
| `PARTIAL_PREFIX` | `"I've exceeded my reasoning limit..."` | Prepended when iteration cap hit |

#### `AgentState` (TypedDict)

Graph state passed between nodes:

| Field | Type | Purpose |
|-------|------|---------|
| `user_input` | `str` | Original user message for this turn |
| `iteration` | `int` | LLM call count this turn |
| `exceeded` | `bool` | Iteration cap or fatal short-circuit |
| `pending_tool_calls` | `list[dict]` | Parsed `tool_use` blocks awaiting dispatch |
| `stop_reason` | `str \| None` | Anthropic stop reason from last call |
| `last_text` | `str` | Accumulated assistant text |
| `final_text` | `str \| None` | Pre-set final answer (errors, rejections) |
| `partial` | `bool` | True if stopped due to iteration cap |
| `input_tokens_total` | `int` | Running input token count |
| `output_tokens_total` | `int` | Running output token count |
| `error` | `str \| None` | Error message if any |
| `_tool_results` | `list[dict]` | Internal: results between dispatch and result nodes |

#### `Agent` class

##### `__init__(memory=None, settings=None, on_text=None, on_tool=None, on_iteration=None, client=None)`

- Loads settings, configures logging
- Creates Anthropic client (or uses injected `client` for testing)
- Creates `ConversationMemory` with system prompt and token budget
- Stores optional callbacks:
  - `on_text(delta)` — streaming text chunks
  - `on_tool(name, input, result)` — after each tool execution
  - `on_iteration(n)` — at start of each reasoning iteration
- Loads tool schemas via `get_tool_schemas()`
- Builds and compiles LangGraph via `_build_graph()`

##### `run(user_input: str) -> dict`

Public API for one user turn:

1. Builds initial `AgentState`
2. Invokes graph with `recursion_limit = max_iterations * 4 + 10`
3. Calls `memory.save()` (warns on failure)
4. Returns:

```python
{
    "text": str,           # final assistant text
    "iterations": int,     # LLM calls made
    "partial": bool,       # True if iteration cap hit
    "input_tokens": int,
    "output_tokens": int,
    "error": str | None,
}
```

##### `_build_graph()`

Constructs `StateGraph(AgentState)` with nodes and edges (see topology above). Returns `graph.compile()`.

##### `input_node(state) -> dict`

1. Runs `sanitize_input(user_input, max_chars=settings.max_input_chars)`
2. On rejection → sets `final_text`, `error`, `stop_reason="input_rejected"`
3. On success → `memory.add_user(guard.value)`, returns `{}`

##### `reasoning_node(state) -> dict`

1. Short-circuits if input was rejected
2. Checks `iteration >= max_iterations` → `{exceeded: True, partial: True}`
3. Increments iteration, fires `on_iteration`
4. Calls `_call_llm()`; on failure → API error final text
5. Parses content blocks via `_blocks_to_dicts()`
6. **Filters malformed `tool_use`** (missing `id` or `name`) before persisting
7. `memory.add_assistant(blocks)`
8. Extracts text and tool_calls; accumulates token usage
9. **Accumulates text** across `max_tokens` continuations (concatenates with previous `last_text`)

##### `tool_dispatch_node(state) -> dict`

For each pending `tool_use`:

1. Validates `name` and `id`
2. Normalizes `input` to dict
3. Calls `dispatch_tool(name, input, timeout_s=settings.tool_timeout_s)`
4. Fires `on_tool` callback
5. Returns `{_tool_results: [...]}`

##### `tool_result_node(state) -> dict`

For each result:

1. Determines payload (result or error)
2. Replaces empty payloads with explicit error
3. `memory.add_tool_result(tool_use_id, content, is_error)`
4. Clears `pending_tool_calls`

##### `output_node(state) -> dict`

1. If `final_text` already set → redact and return
2. If `partial` → prefix with `PARTIAL_PREFIX` + `last_text`
3. Else → `last_text` or fallback message
4. Always runs `redact_secrets()` on output

##### `_route_after_reasoning(state) -> str`

Returns `"tools"`, `"continue"`, or `"output"`:

| Condition | Route |
|-----------|-------|
| `exceeded` or `final_text` set | `"output"` |
| `pending_tool_calls` non-empty | `"tools"` |
| `stop_reason == "max_tokens"` | `"continue"` |
| `stop_reason == "tool_use"` but no pending tools (all filtered) | `"continue"` |
| Otherwise | `"output"` |

##### `_call_llm()`

Calls Anthropic with retry logic:

- Uses streaming (`_stream_call`) when `on_text` is set and client supports it
- Otherwise `client.messages.create(...)`
- Logs via `log_llm_call`
- **RateLimitError**: up to 3 retries, exponential backoff (1s, 2s, 4s...)
- **APITimeoutError / APIConnectionError**: 1 retry

##### `_stream_call(system, history, model, max_tokens)`

Uses `client.messages.stream(...)`, emits text deltas through `on_text` (with per-chunk redaction), returns `stream.get_final_message()`.

##### `_blocks_to_dicts(content) -> list[dict]` (static)

Normalizes SDK content blocks (objects or dicts) into Anthropic-format dicts with `type`, `text`/`id`/`name`/`input`.

---

## Utilities (`utils/`)

### `utils/__init__.py`

Empty package marker.

---

### `utils/safety.py`

Defense-in-depth guards for input, output, file paths, and calculator expressions.

#### Constants

| Name | Purpose |
|------|---------|
| `MAX_INPUT_CHARS` | Default 10,000 char input cap |
| `MAX_EXPONENT` | 1000 — prevents `9**99999` CPU exhaustion |
| `_API_KEY_RE` | Regex for `sk-...` style keys |
| `_TAVILY_KEY_RE` | Regex for `tvly-...` keys |
| `_SENSITIVE_SUFFIXES` | `.pem`, `.key`, `.crt`, etc. |
| `_CALC_ALLOWED_NAMES` | Whitelisted math function/constant names |
| `_CALC_TOKEN_RE`, `_CALC_CHAR_RE` | Expression validation regexes |

#### `GuardResult` (dataclass)

`ok: bool`, `value: str`, `error: str | None`

#### `sanitize_input(text, max_chars=MAX_INPUT_CHARS) -> GuardResult`

1. Rejects `None`
2. Strips null bytes
3. Rejects if over `max_chars`
4. Rejects if >50% non-printable characters
5. Returns cleaned text on success

#### `redact_secrets(text: str) -> str`

Replaces API key patterns with `[REDACTED_API_KEY]`.

#### `safe_resolve_path(path, base_dir=None) -> GuardResult`

Path rules for file tools:

- Reject `..` segments
- Reject absolute paths
- Reject dotfiles/dotdirs and sensitive suffixes
- Resolved path must stay within `base_dir` (default: cwd)

#### `validate_calculator_expression(expression) -> GuardResult`

Character and name whitelist check (used conceptually; `safe_eval_math` does AST validation).

#### `safe_eval_math(expression) -> tuple[bool, value, error]`

**Safe math evaluator** — never uses `eval()`:

1. Parses expression with `ast.parse(mode="eval")`
2. Recursively evaluates only allowed nodes:
   - `Constant` (int/float only)
   - `BinOp` (+, -, *, /, %, //, ** with exponent cap)
   - `UnaryOp` (+, -)
   - `Call` (whitelisted functions: sqrt, sin, cos, tan, log, pow, abs)
   - `Name` (pi, e constants only)
3. Returns `(True, value, None)` or `(False, None, error_message)`

#### `_ExponentTooLarge` (internal exception)

Raised when `**` exponent exceeds `MAX_EXPONENT`.

---

### `utils/logging.py`

Loguru configuration and structured log helpers.

#### `_CONFIGURED` (module flag)

Ensures `setup_logging` only configures sinks once.

#### `setup_logging(log_level="INFO", log_dir="logs")`

1. Removes default loguru handler
2. Adds **console** sink at `log_level` with colored format
3. Adds **file** sink at DEBUG: `logs/agent_{date}.log`, 10 MB rotation, 14-day retention
4. Sets `_CONFIGURED = True`

#### `log_llm_call(model, input_tokens, output_tokens, latency_ms, stop_reason)`

INFO log line for each LLM API call.

#### `log_tool_call(tool_name, tool_input, result_summary, latency_ms, success)`

INFO on success, WARNING on failure. Truncates input/result to 300 chars.

#### `log_iteration(iteration, stop_reason)`

INFO log for each loop iteration.

#### `_truncate(text, limit) -> str`

Truncates with `"... (+N chars)"` suffix.

---

## Test Suite (`tests/`)

### `conftest.py`

Adds project root to `sys.path` so imports work when running pytest.

### `tests/test_loop.py`

Tests the agent loop with **mocked Anthropic clients** (`FakeClient`, `AlwaysToolClient`). Covers:

- Single-turn text response
- Tool call → result → final answer
- Max iteration cap → partial answer with prefix
- Multiple `tool_use` blocks in one response
- Malformed `tool_use` filtering and recovery
- `max_tokens` text accumulation across continuations
- Valid `tool_use` always gets matching `tool_result`
- Input rejection (no LLM call)

### `tests/test_memory.py`

Tests `ConversationMemory`: add/retrieve, tool result batching, token trimming, system prompt preservation, export/import, save/load, clear, conversation-start repair after trim, malformed JSON load.

### `tests/test_tools.py`

Tests each tool and `dispatch_tool`: calculator edge cases, file I/O safety, web search mocking, unknown tool, bad input type, process round-trip, timeout kills runaway tool.

### `tests/test_safety.py`

Tests `redact_secrets`, `safe_resolve_path`, `safe_eval_math` guards.

---

## Design Patterns Worth Learning

### 1. LangGraph as an explicit state machine

Instead of a `while True` loop with nested conditionals, the agent loop is a **graph** with named nodes and conditional edges. Benefits:

- Each step has a single responsibility
- Routing logic is centralized in `_route_after_reasoning`
- Easy to visualize and extend (add a node for human approval, RAG retrieval, etc.)

### 2. Anthropic tool-use contract

Every `tool_use` block in assistant history **must** be followed by a matching `tool_result` in the next user message. The code:

- Filters malformed `tool_use` **before** persisting (prevents 400 errors on next call)
- Batches multiple `tool_result` blocks into one user message
- Repairs memory after trimming so history does not start with orphaned tool results

### 3. Structured tool results

All tools return `{success, result, error}` — the loop never needs try/except around individual tools. Empty results become explicit errors in `tool_result_node`.

### 4. Process-isolated tools with hard kill

`dispatch_tool` runs each tool in a child process. On timeout, the process is terminated (SIGTERM → SIGKILL). This prevents a hung or CPU-heavy tool from blocking the agent forever — important for any agent that runs user-influenced code paths.

### 5. Defense in depth for safety

| Threat | Mitigation |
|--------|------------|
| Huge user input | `sanitize_input` length + garbage check |
| Leaked API keys in output | `redact_secrets` on final text and stream chunks |
| Path traversal | `safe_resolve_path` |
| Calculator code injection | AST whitelist evaluator, no `eval()` |
| Runaway exponentiation | `MAX_EXPONENT` cap |
| Infinite agent loop | `max_iterations` cap with partial answer |

### 6. Callback hooks for UI decoupling

`Agent` accepts `on_text`, `on_tool`, `on_iteration` — the core loop has no Streamlit dependency. The same `Agent` could power a CLI, FastAPI WebSocket, or tests.

### 7. Injectable client for testing

`Agent(..., client=fake_client)` lets tests run the full graph without network calls.

### 8. Token budget with conversation repair

Cheap `chars/4` estimation plus trimming oldest messages is good enough for a demo agent. The **repair** step (don't leave orphaned tool results or leading assistant messages) is the subtle part that keeps the Anthropic API happy after trimming.

---

## Environment Variables Reference

Copy `.env.example` to `.env`:

```bash
ANTHROPIC_API_KEY=your_key_here      # Required for chat
TAVILY_API_KEY=your_key_here         # Required for web_search tool
ANTHROPIC_MODEL=claude-sonnet-4-6    # Model ID
MAX_ITERATIONS=10                    # LLM calls per user turn
TOKEN_BUDGET=80000                   # History size limit (estimated tokens)
LOG_LEVEL=INFO                       # Console log level

# Optional (defaults in parentheses)
API_TIMEOUT_S=30
TOOL_TIMEOUT_S=10
MAX_INPUT_CHARS=10000
MAX_OUTPUT_TOKENS=4096
```

---

## File Index (code only)

| File | Role |
|------|------|
| `main.py` | Streamlit UI entry point |
| `config.py` | Environment-based settings |
| `agent/loop.py` | Agent class + LangGraph loop |
| `agent/memory.py` | Conversation history + persistence |
| `agent/prompts.py` | System prompt for Aria |
| `agent/tools.py` | Tool implementations + dispatch |
| `agent/__init__.py` | Package marker |
| `utils/safety.py` | Input/output/path/math guards |
| `utils/logging.py` | Loguru setup + structured logs |
| `utils/__init__.py` | Package marker |
| `conftest.py` | Pytest path setup |
| `tests/test_loop.py` | Agent loop tests |
| `tests/test_memory.py` | Memory tests |
| `tests/test_tools.py` | Tool + dispatch tests |
| `tests/test_safety.py` | Safety guard tests |

---

## Suggested Study Order

1. **`config.py`** — understand what is configurable
2. **`agent/prompts.py`** — what the model is told to do
3. **`agent/memory.py`** — how conversation state is stored and trimmed
4. **`agent/tools.py`** — how tools work and are isolated
5. **`utils/safety.py`** — guards that protect tools and I/O
6. **`agent/loop.py`** — put it all together in the graph
7. **`main.py`** — how the UI wires callbacks and memory
8. **`tests/`** — concrete examples of expected behavior

After reading in that order, trace a single message through the graph with the diagrams above. Then try adding a new tool: implement the function, add it to `_TOOL_FUNCS` and `get_tool_schemas()`, and mention it in `SYSTEM_PROMPT`.
