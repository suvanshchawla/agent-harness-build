import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ToolAnnotations
from openai import AsyncOpenAI
from pydantic import BaseModel

from agent_harness_build.paths import MCP_CONFIG, ROOT, WORKSPACE

load_dotenv(ROOT.parents[1] / ".env")  # project-root .env, untracked


# ---------------------------------------------------------------------------
# Pillar 3: the model registry. Every entry is just an OpenAI-compatible
# endpoint. Cloud or local, frontier or free - same client, same harness.
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    base_url: str
    api_key: str = "none"
    model: str


MODELS: dict[str, ModelConfig] = {
    "local": ModelConfig(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3.5:9b",
    ),
    "gpt": ModelConfig(
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model="gpt-5.2",
    ),
    "claude": ModelConfig(
        # Anthropic's OpenAI-compatible endpoint - same client works.
        base_url="https://api.anthropic.com/v1/",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model="claude-sonnet-5-5",
    ),
}

model_name = "local"  # switch at runtime with /model


SYSTEM_PROMPT = """You are a helpful personal assistant running inside a \
custom harness. Be concise.

The user's workspace folder holds their personal files: notes, todo lists, \
ideas. You have tools to list, read, and write those files. Never claim you \
lack access to the user's files or tasks - use your tools to look.

Here is what you remember about the user from previous sessions:
{memory}

When you learn a new lasting fact about the user, save it with the save_memory tool."""

# the running conversation - shared by every model we switch to
messages: list = [{"role": "system", "content": SYSTEM_PROMPT.format(memory="(not loaded)")}]


# ---------------------------------------------------------------------------
# Pillar 1: MCP. Every tool - including our own workspace/memory ones - comes
# from an MCP server listed in mcp_servers.json. Tool names are prefixed with
# the server name ("workspace__read_file") so two servers can't collide.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = []
tool_routes: dict[str, tuple[ClientSession, str]] = {}  # exposed name -> (session, real name)
prompt_routes: dict[str, tuple[ClientSession, str]] = {}  # "server/prompt" -> (session, name)
sessions: dict[str, ClientSession] = {}  # server name -> session
tool_annotations: dict[str, ToolAnnotations | None] = {}  # exposed name -> the server's risk hints


async def connect_mcp(stack: AsyncExitStack, config_path: Path | None = None):
    """Launch every server in the config (default mcp_servers.json) and register its tools and prompts."""
    config = json.loads((config_path or MCP_CONFIG).read_text(encoding="utf-8"))
    for name, spec in config["mcpServers"].items():
        command = sys.executable if spec["command"] == "{python}" else spec["command"]
        params = StdioServerParameters(command=command, args=spec["args"], env=spec.get("env"))
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools = (await session.list_tools()).tools
            prompts = (await session.list_prompts()).prompts
        except Exception as e:
            print(f"  [mcp] '{name}' failed to start: {e}")
            continue

        sessions[name] = session
        for tool in tools:
            exposed = f"{name}__{tool.name}"
            tool_routes[exposed] = (session, tool.name)
            tool_annotations[exposed] = tool.annotations
            TOOL_SCHEMAS.append(
                {
                    "type": "function",
                    "function": {
                        "name": exposed,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
            )
        for prompt in prompts:
            prompt_routes[f"{name}/{prompt.name}"] = (session, prompt.name)
        print(f"  [mcp] connected '{name}': {[t.name for t in tools]}")


async def read_memory() -> str:
    """Fetch the memory://facts resource from whichever server provides it."""
    for session in sessions.values():
        try:
            result = await session.read_resource("memory://facts")
        except Exception:
            continue
        return "".join(c.text for c in result.contents if hasattr(c, "text")).strip() or "(nothing saved yet)"
    return "(memory unavailable)"


async def call_tool(name: str, args: dict) -> str:
    if name not in tool_routes:
        return f"error: unknown tool {name}"
    session, real_name = tool_routes[name]
    result = await session.call_tool(real_name, args)
    text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
    return f"error: {text}" if result.is_error else text


# ---------------------------------------------------------------------------
# Approval: ask the user before risky tool calls
# ---------------------------------------------------------------------------

Approver = Callable[[str, dict], Awaitable[bool]]

APPROVAL_MODES = ("writes", "all", "none")
approval_mode = "writes"  # writes: confirm destructive tools | all: confirm every call | none: never ask
always_allowed: set[str] = set()  # tools the user answered "always" for, this session only


def needs_approval(name: str) -> bool:
    """Decide from the server's own annotations whether this call needs the user's OK.

    In "writes" mode, read-only tools never ask. Tools that only ever add (destructiveHint
    false) don't ask either. Anything else asks, including tools with no annotations, because
    the MCP spec says to assume an unannotated tool may be destructive.
    """
    if approval_mode == "none" or name in always_allowed:
        return False
    if approval_mode == "all":
        return True
    ann = tool_annotations.get(name)
    if ann is None:
        return True
    if ann.read_only_hint:
        return False
    return ann.destructive_hint is not False


async def ask_user(name: str, args: dict) -> bool:
    """Interactive approver: show the call, read y / n / a(lways) from the terminal."""
    print(f"\n  [approval needed] {name}")
    for key, value in args.items():
        shown = str(value).replace("\n", "\n" + " " * (len(key) + 8))
        print(f"      {key}: {shown[:300]}")
    answer = await asyncio.to_thread(input, "  allow? [y]es / [n]o / [a]lways this session: ")
    answer = answer.strip().lower()
    if answer in ("a", "always"):
        always_allowed.add(name)
        return True
    return answer in ("y", "yes")


DENIED = (
    "error: the user denied this action, so it was NOT performed. "
    "Do not retry it. Tell the user it was not done and ask what they would like instead."
)


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

MAX_STEPS = 10  # model round-trips per user message before we give up


async def run_agent(
    user_message: str,
    history: list | None = None,
    model: str | None = None,
    trace: list | None = None,
    verbose: bool = True,
    client: AsyncOpenAI | None = None,
    approve: Approver | None = None,
) -> str:
    """Run one user turn to completion.

    history/model default to the interactive session; the eval runner passes its
    own. client defaults to one built from the model registry (tests inject a fake).
    If trace is given, every tool call is appended to it as
    {"tool", "args", "result"}, plus "denied": True when the user said no.

    approve is asked before any call that needs_approval() flags. Leaving it as None
    approves everything, which is what the eval runner and tests want.
    """
    history = messages if history is None else history
    cfg = MODELS[model or model_name]
    client = client or AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    history.append({"role": "user", "content": user_message})
    for _ in range(MAX_STEPS):
        response = await client.chat.completions.create(model=cfg.model, messages=history, tools=TOOL_SCHEMAS)
        message = response.choices[0].message
        if not message.tool_calls:
            history.append({"role": "assistant", "content": message.content})
            return message.content or ""

        history.append(message)
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            if verbose:
                print(f"  [tool] {call.function.name}({args})")
            denied = (
                approve is not None
                and needs_approval(call.function.name)
                and not await approve(call.function.name, args)
            )
            result = DENIED if denied else await call_tool(call.function.name, args)
            if trace is not None:
                entry = {"tool": call.function.name, "args": args, "result": result}
                trace.append({**entry, "denied": True} if denied else entry)
            history.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return f"(stopped after {MAX_STEPS} steps without a final answer)"


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


async def handle_command(line: str) -> bool:
    """Returns True if the line was a command."""
    global model_name, approval_mode
    if not line.startswith("/"):
        return False
    cmd, _, arg = line.partition(" ")
    if cmd == "/models":
        for name, cfg in MODELS.items():
            marker = "*" if name == model_name else " "
            print(f" {marker} {name:<12} {cfg.model}  ({cfg.base_url})")
    elif cmd == "/model":
        if arg in MODELS:
            model_name = arg
            print(f"  switched to {arg} ({MODELS[arg].model})")
        else:
            print(f"  unknown model '{arg}' - try /models")
    elif cmd == "/tools":
        for schema in TOOL_SCHEMAS:
            fn = schema["function"]
            print(f"  {fn['name']}: {' '.join(fn['description'].split())[:60]}")
    elif cmd == "/approve":
        if arg in APPROVAL_MODES:
            approval_mode = arg
            print(f"  approval mode: {arg}")
        else:
            print(f"  approval mode is '{approval_mode}' - choose from {', '.join(APPROVAL_MODES)}")
    elif cmd == "/memory":
        print(await read_memory())
    elif cmd == "/prompts":
        for key in prompt_routes:
            print(f"  {key}")
    elif cmd == "/prompt":
        # /prompt workspace/summarize_file filename=notes.md
        key, *pairs = arg.split()
        if key not in prompt_routes:
            print(f"  unknown prompt '{key}' - try /prompts")
        else:
            session, name = prompt_routes[key]
            result = await session.get_prompt(name, dict(p.split("=", 1) for p in pairs))
            text = "\n".join(m.content.text for m in result.messages if hasattr(m.content, "text"))
            print("\nassistant:", await run_agent(text, approve=ask_user), "\n")
    elif cmd == "/quit":
        raise SystemExit
    else:
        print(
            "  commands: /models /model <name> /tools /memory /prompts /prompt <server/name> [k=v ...]\n"
            "            /approve [writes|all|none] /quit"
        )
    return True


async def main():
    WORKSPACE.mkdir(exist_ok=True)
    async with AsyncExitStack() as stack:
        print("starting harness...")
        await connect_mcp(stack)
        messages[0]["content"] = SYSTEM_PROMPT.format(memory=await read_memory())
        print(
            f"\nharness ready - model: {model_name} "
            f"({MODELS[model_name].model}), "
            f"{len(TOOL_SCHEMAS)} tools, memory loaded"
        )
        print("type /models, /model <name>, /tools, /memory, /prompts, or just talk\n")
        try:
            while True:
                line = (await asyncio.to_thread(input, "you: ")).strip()
                if not line or await handle_command(line):
                    continue
                print("\nassistant:", await run_agent(line, approve=ask_user), "\n")
        except (EOFError, KeyboardInterrupt, SystemExit):
            print("\nbye!")


if __name__ == "__main__":
    asyncio.run(main())
