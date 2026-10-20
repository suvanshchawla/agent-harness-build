"""A test-only MCP server with slow tools that log when each call starts and ends.

The log lets tests check whether calls really overlapped, without depending on timing.
Set AGENT_LOG to the file to append "start|end <tool> <label> <time>" lines to.
"""

import asyncio
import os
import time

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

mcp = MCPServer("slow")


def log(event: str, tool: str, label: str) -> None:
    with open(os.environ["AGENT_LOG"], "a") as f:
        f.write(f"{event} {tool} {label} {time.time()}\n")


async def work(tool: str, label: str, delay: float) -> str:
    log("start", tool, label)
    await asyncio.sleep(delay)
    log("end", tool, label)
    return f"{tool}:{label}"


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def read(label: str, delay: float = 0.3) -> str:
    """A slow read-only call."""
    return await work("read", label, delay)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True))
async def write(label: str, delay: float = 0.3) -> str:
    """A slow destructive call."""
    return await work("write", label, delay)


if __name__ == "__main__":
    mcp.run(transport="stdio")
