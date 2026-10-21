# agent-harness-build

[![CI](https://github.com/suvanshchawla/agent-harness-build/actions/workflows/ci.yml/badge.svg)](https://github.com/suvanshchawla/agent-harness-build/actions/workflows/ci.yml)

A small, provider-agnostic **AI agent harness** built around the **Model Context Protocol (MCP)**.

- One tool-calling loop that works with any OpenAI-compatible endpoint: local models via Ollama, OpenAI, and Anthropic. Switch models mid-conversation with `/model`.
- An **MCP client** that launches servers from a JSON config and exposes all of their tools to the model.
- A **custom MCP server** (`workspace`) that provides file tools, long-term memory, resources and prompts, covering all three MCP primitives.

## Architecture

```mermaid
flowchart LR
    User([User]) <--> CLI

    subgraph Harness["harness.py"]
        CLI["Chat loop and slash commands"]
        Loop["Agent loop<br/>(tool-calling)"]
        Router["MCP client<br/>tool / prompt / resource routing"]
        CLI --> Loop
        Loop <--> Router
    end

    Loop <-->|"OpenAI-compatible<br/>chat completions"| Models

    subgraph Models["Model registry"]
        Local["local: Ollama"]
        GPT["gpt: OpenAI"]
        Claude["claude: Anthropic"]
    end

    Router <-->|stdio| WS
    Router <-->|stdio| Time["time (uvx)"]
    Router <-->|stdio| Fetch["fetch (uvx)"]

    subgraph WS["workspace server (this repo)"]
        Tools["Tools<br/>list_files, read_file, write_file<br/>save_memory, forget_memory"]
        Res["Resources<br/>memory://facts<br/>workspace://{filename}"]
        Prompts["Prompts<br/>daily_review, summarize_file"]
    end

    Tools --- Files[("workspace/")]
    Tools --- Mem[("memory.md")]
    Res --- Files
    Res --- Mem
```

### How a turn works

1. On startup the harness reads `mcp_servers.json`, spawns each server over stdio, and collects their tools and prompts.
2. Tool names are prefixed with the server name (`workspace__read_file`), so two servers can't collide.
3. The long-term memory shown in the system prompt is fetched from the `memory://facts` **resource**, not read from disk directly.
4. The model's tool calls are routed back to the owning MCP session, and results are appended to the conversation until the model answers in plain text.

## Project layout

```
src/agent_harness_build/
  harness.py                 chat loop, agent loop, MCP client, slash commands
  paths.py                   workspace / memory locations shared by harness and servers
  mcp_servers.json           MCP servers the harness launches
  servers/
    workspace_server.py      the custom MCP server
  evals/
    cases.json               eval cases (prompt + expected behaviour)
    scoring.py               pure scoring logic, unit tested
    runner.py                runs the cases against each model
scripts/inspect_server.py    demo client that exercises the workspace server
tests/                       pytest suite (in-memory MCP client + a stdio integration test)
examples/                    v0-v4: the step-by-step build-up to the harness
```

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env        # then add your API keys (only needed for the gpt / claude models)
uv run agent-harness-build
```

The default model is `local`, which expects [Ollama](https://ollama.com) on `localhost:11434` with `qwen3.5:9b` pulled. The `time` and `fetch` servers are launched with `uvx`, so they need network access the first time.

### Commands

| Command | What it does |
| --- | --- |
| `/models`, `/model <name>` | List or switch the active model |
| `/tools` | List every tool from every connected server |
| `/memory` | Show stored memory (via the `memory://facts` resource) |
| `/prompts` | List prompts exposed by the servers |
| `/prompt <server/name> [k=v ...]` | Run a server prompt through the agent, e.g. `/prompt workspace/summarize_file filename=notes.md` |
| `/approve [writes\|all\|none]` | Show or change when tool calls need your confirmation |
| `/quit` | Exit |

## The workspace MCP server

It can be run on its own, for example from an MCP client such as the MCP Inspector or Claude Desktop:

```bash
uv run workspace-server
```

| Primitive | Name | Purpose |
| --- | --- | --- |
| Tool | `list_files`, `read_file`, `write_file` | Work with the workspace folder. Paths that escape it are rejected. |
| Tool | `save_memory`, `forget_memory` | Add (deduplicated) or remove remembered facts |
| Resource | `memory://facts` | All remembered facts |
| Resource | `workspace://{filename}` | One workspace file |
| Prompt | `daily_review`, `summarize_file` | Reusable prompt templates |

Set `AGENT_WORKSPACE` and `AGENT_MEMORY_FILE` to change where files and memory live.

### Trying the server by hand

A stdio server prints nothing when started directly, because it waits for a client. Two ways to poke at it:

```bash
# 1. Python script: lists everything the server exposes and calls a few tools.
#    Uses a temporary workspace, so your real files and memory are untouched.
uv run python scripts/inspect_server.py

# 2. MCP Inspector: a browser UI for the tools, resources and prompts (needs Node 22.19+).
npx @modelcontextprotocol/inspector uv run workspace-server
```

## Approval prompts

Before a risky tool call runs, the harness stops and asks:

```
  [approval needed] workspace__write_file
      filename: todo.md
      content: - call dentist
               - buy milk
  allow? [y]es / [n]o / [a]lways this session:
```

What counts as risky comes from the server itself, via the MCP tool annotations (`readOnlyHint`, `destructiveHint`), so it works for any server. The workspace server marks `read_file` and `list_files` read-only, `save_memory` as add-only, and `write_file` and `forget_memory` as destructive. Third-party tools with no annotations are treated as risky, as the spec advises. The default mode, `writes`, asks only for destructive tools, and `/approve all` or `/approve none` changes that. A denied call is not run: the model is told the user said no and to ask what they want instead. The eval runner and the tests approve everything automatically.

## Streaming

In the interactive loop the reply prints as the model writes it, instead of appearing all at once. `run_agent(..., on_text=callback)` turns this on: the model call is streamed and each piece of text goes to the callback as it arrives, while the return value is still the full answer. Without `on_text` (the eval runner, tests) nothing changes.

Tool calls arrive in fragments, with the name once and the JSON arguments spread over many chunks, so `stream_completion` stitches them back together by index and only then do the tools run. The loop after that sees the same message shape either way. The test fake replays scripted replies as small chunks, including split-up tool calls, so this is covered without a model.

## Limits on tool output

A tool result longer than `MAX_TOOL_CHARS` (8,000 characters) is cut before the model sees it, with a marker saying how many characters were dropped. This keeps one big file from filling a small local context window. The trace and eval results record the truncated text, i.e. what the model actually saw. The `fetch` server already pages its own output at 5,000 characters, so this mainly protects against other tools, such as reading a large workspace file.

## Cancelling a turn

Ctrl+C while the assistant is working cancels that turn and returns to the `you:` prompt; Ctrl+C at the prompt still quits.

- **Nothing had run yet** (still thinking or writing): the turn is removed from the conversation as if it never happened.
- **Tools are running:** calls already started are allowed to finish, so a file write is never abandoned halfway. Calls that had not started are skipped and reported as not run, the model's history keeps what did happen, and the turn ends with a `(cancelled by the user)` note. If an approval question is open, cancelling means the call will not run, whatever you answer.
- **Ctrl+C a second time** stops waiting for a slow tool and aborts right away.

## Parallel tool calls

Models often ask for several tools in one message ("read a.md, b.md and c.md"). The harness runs neighbouring **read-only** calls at the same time and everything else one at a time, in the order the model asked:

```
read  read  write  read    ->  [read read]  [write]  [read]
```

Read-only is decided from the server's own `readOnlyHint` annotation, the same one the approval prompts use. A write is never reordered or overlapped with the calls around it, so a model that asks to write a file and then read it back still sees the new contents. Results go back to the model in call order, however the calls finish. Approval questions are asked one at a time before a batch starts, and a call that crashes becomes an error result for that call only, so its siblings still complete. `run_agent(..., parallel=False)` restores strictly sequential execution.

The speedup only shows with slow tools such as `fetch`: the local workspace tools return in milliseconds. The tests use a small slow MCP server that logs when each call starts and ends, and check overlap from that log instead of from timing.

## Evals

Unit tests can't tell you whether a *model* picks the right tool, so there is a small eval suite. Each case in `src/agent_harness_build/evals/cases.json` is a prompt plus expectations: which tools should (or should not) be called with which arguments, what the answer should contain, and what the workspace files and memory should look like afterwards.

```bash
uv run harness-evals                      # every model that has credentials
uv run harness-evals --models local       # one model
uv run harness-evals --trials 3           # repeat each case to smooth out noise
uv run harness-evals --tier hard          # or --category safety, --case <id>
```

Every trial runs the real agent loop against real MCP servers, but with a throwaway workspace and memory file, so your own data is untouched. Models without an API key are skipped. The runner prints a pass-rate table per model and category and saves the full traces to `evals/results/` (gitignored).

There are 30 cases in two tiers. The 10 **easy** ones cover single-tool basics. The 20 **hard** ones cover recovering from a wrong filename, honesty about missing files, searching across files, copy and summarize workflows, memory updates where an old fact becomes outdated (plus a control that must keep the old fact), restraint (not saving trivia or things already known), prompt injection inside a file, write and read path traversal, and the `time` / `fetch` servers (the network case is skipped when offline). The scoring logic itself is covered by ordinary unit tests in `tests/test_evals.py`.

### What the evals found

The evals found two real tool-design problems on the local `qwen3.5:9b`, and both were fixed in the server rather than by tuning prompts.

**1. `forget_memory` failed silently.** Asked to forget "Lives in Delhi", the model searched for `"live in Delhi"`. The tool returned "removed 0", and the model told the user it had succeeded. The tool now says nothing matched and returns the current facts so the model can retry.

**2. Memory contradicted itself.** When the user said "I've moved to Mumbai", the model saved the new fact but left "Lives in Delhi" in memory. Four cases cover this (a move, a job change, "I don't like coffee anymore", a changed email), plus a control where the old fact must be kept. They check the final memory contents, not which tools were called, so any route to the right state passes. The fix was in the tool: `save_memory` now returns the other stored facts and asks the model to `forget_memory` any that are outdated, and its description says that a replacing fact needs the old one removed.

| Case (5 trials, local model) | Before | After |
| --- | --- | --- |
| Move city | 1/5 | 5/5 |
| Job change | 0/5 | 5/5 |
| No longer likes X | 2/5 | 5/5 |
| Changed email | 2/5 | 5/5 |
| Control: "I also like coffee" keeps the old fact | 5/5 | 5/5 |

The control matters: the nudge did not make the model delete facts it should have kept.

Caveats: this is one small model and 5 trials per case, so treat the numbers as a strong signal on this model and no more. One restraint case got no better (the model re-saves a fact it already knows, which the tool ignores as a duplicate), and it may simply be too strict.

## Adding another MCP server

Add an entry to `src/agent_harness_build/mcp_servers.json`:

```json
"my-server": { "command": "uvx", "args": ["my-mcp-server"] }
```

An optional `"env": {...}` object sets environment variables for the server process. Its tools appear in `/tools` as `my-server__<tool>`. The special command `{python}` expands to the current interpreter, which is how the bundled workspace server is launched.

## Tests

```bash
uv run pytest
```

The suite needs no model or API key, and CI runs it (plus `ruff`) on every push and pull request. It covers the agent loop and the eval runner (both driven by a scripted fake model against a real server), each tool, resource and prompt, path-traversal rejection, memory deduplication, and a real stdio connection from the harness to the workspace server.

## Design notes

- **OpenAI-compatible endpoints everywhere.** Local and cloud models share one client and one message format, so adding a provider is a registry entry.
- **stdio transport.** Simple, no ports or auth, and the harness controls the server's lifetime. Anything a stdio server prints to stdout would corrupt the protocol, so servers must log to stderr.
- **Everything is an MCP tool.** The harness has no hard-coded tools, which means the same server can be reused by any other MCP client.

## Roadmap

- Harder eval cases and a comparison across cloud models
- Embedding-based memory retrieval

## License

MIT, see [LICENSE](LICENSE).
