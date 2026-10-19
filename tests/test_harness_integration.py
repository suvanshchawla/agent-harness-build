"""Spawns the real workspace server over stdio and drives it through the harness."""

import json
from contextlib import AsyncExitStack

from agent_harness_build import harness


async def test_harness_connects_to_workspace_server(tmp_path, monkeypatch):
    config = tmp_path / "servers.json"
    config.write_text(json.dumps({"mcpServers": {"workspace": {
        "command": "{python}",
        "args": ["-m", "agent_harness_build.servers.workspace_server"],
    }}}))
    monkeypatch.setattr(harness, "MCP_CONFIG", config)
    for registry in (harness.TOOL_SCHEMAS, harness.tool_routes, harness.prompt_routes, harness.sessions):
        registry.clear()

    async with AsyncExitStack() as stack:
        await harness.connect_mcp(stack)
        assert "workspace__read_file" in harness.tool_routes
        assert "workspace/daily_review" in harness.prompt_routes

        assert "outside the workspace" in await harness.call_tool(
            "workspace__read_file", {"filename": "../x"}
        )
        assert "unknown tool" in await harness.call_tool("nope", {})
        assert (await harness.read_memory()) != "(memory unavailable)"
