import json
from contextlib import AsyncExitStack, asynccontextmanager

import pytest

from agent_harness_build import harness


@pytest.fixture
def connected_harness(tmp_path, monkeypatch):
    """Factory for an async context manager: the harness wired to a real workspace server.

    The server uses a throwaway workspace and memory file. Enter it with `async with` inside the
    test, since MCP clients must be entered and exited in the same task.
    """
    workspace, memory = tmp_path / "workspace", tmp_path / "memory.md"
    config = tmp_path / "servers.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "workspace": {
                        "command": "{python}",
                        "args": ["-m", "agent_harness_build.servers.workspace_server"],
                        "env": {"AGENT_WORKSPACE": str(workspace), "AGENT_MEMORY_FILE": str(memory)},
                    }
                }
            }
        )
    )

    @asynccontextmanager
    async def connect():
        for registry in (harness.TOOL_SCHEMAS, harness.tool_routes, harness.prompt_routes, harness.sessions):
            registry.clear()
        async with AsyncExitStack() as stack:
            await harness.connect_mcp(stack, config)
            yield workspace, memory

    return connect
