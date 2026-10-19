"""Connect to the workspace MCP server over stdio and exercise it.

Prints everything the server exposes, then calls a few tools. It runs against a
temporary workspace and memory file, so your real data is never touched.

    uv run python scripts/inspect_server.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from mcp import StdioServerParameters
from mcp.client import Client


def text_of(result) -> str:
    return "\n".join(c.text for c in result.content if hasattr(c, "text"))


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agent_harness_build.servers.workspace_server"],
            env={
                "AGENT_WORKSPACE": str(Path(tmp) / "workspace"),
                "AGENT_MEMORY_FILE": str(Path(tmp) / "memory.md"),
            },
        )
        async with Client(params) as client:
            print("== Tools")
            for tool in (await client.list_tools()).tools:
                print(f"  {tool.name}: {' '.join((tool.description or '').split())}")

            print("\n== Resources")
            for res in (await client.list_resources()).resources:
                print(f"  {res.uri}")
            for tpl in (await client.list_resource_templates()).resource_templates:
                print(f"  {tpl.uri_template}  (template)")

            print("\n== Prompts")
            for prompt in (await client.list_prompts()).prompts:
                args = ", ".join(a.name for a in prompt.arguments or [])
                print(f"  {prompt.name}({args})")

            print("\n== Try it")
            calls = [
                ("write_file", {"filename": "todo.md", "content": "- buy milk\n"}),
                ("list_files", {}),
                ("read_file", {"filename": "todo.md"}),
                ("read_file", {"filename": "../../etc/passwd"}),
                ("save_memory", {"fact": "Likes tea"}),
                ("save_memory", {"fact": "likes tea"}),
            ]
            for name, args in calls:
                result = await client.call_tool(name, args)
                print(f"  {name}({args}) -> {text_of(result)!r}")

            facts = (await client.read_resource("memory://facts")).contents[0].text
            print(f"\n  memory://facts -> {facts!r}")
            file = (await client.read_resource("workspace://todo.md")).contents[0].text
            print(f"  workspace://todo.md -> {file!r}")

            prompt = await client.get_prompt("summarize_file", {"filename": "todo.md"})
            print(f"  summarize_file prompt -> {prompt.messages[0].content.text!r}")


if __name__ == "__main__":
    asyncio.run(main())
