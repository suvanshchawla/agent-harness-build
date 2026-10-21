import json
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import pytest

from agent_harness_build import harness

SLOW_SERVER = Path(__file__).parent / "slow_server.py"


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


@pytest.fixture
def slow_harness(tmp_path):
    """Factory for a context manager that connects the harness to the slow server; yields a reader
    for the call log: label -> (start, end)."""
    log = tmp_path / "calls.log"
    config = tmp_path / "servers.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "slow": {
                        "command": "{python}",
                        "args": [str(SLOW_SERVER)],
                        "env": {"AGENT_LOG": str(log)},
                    }
                }
            }
        )
    )

    def intervals() -> dict[str, tuple[float, float]]:
        starts, ends = {}, {}
        for line in log.read_text().splitlines():
            event, _tool, label, stamp = line.split()
            (starts if event == "start" else ends)[label] = float(stamp)
        return {label: (starts[label], ends[label]) for label in starts}

    intervals.log = log  # tests that need to know a call has started poll this file

    @asynccontextmanager
    async def connect():
        for registry in (harness.TOOL_SCHEMAS, harness.tool_routes, harness.prompt_routes, harness.sessions):
            registry.clear()
        harness.tool_annotations.clear()
        async with AsyncExitStack() as stack:
            await harness.connect_mcp(stack, config)
            yield intervals

    return connect
