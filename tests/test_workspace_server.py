import pytest
from mcp.client import Client

from agent_harness_build import paths
from agent_harness_build.servers.workspace_server import mcp


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr(paths, "MEMORY_FILE", tmp_path / "memory.md")
    return tmp_path


@pytest.fixture
def connect(isolated):
    """Client(...) must be entered and exited in the same task, so tests open it themselves."""
    return lambda: Client(mcp)


def text(result) -> str:
    return "\n".join(c.text for c in result.content if hasattr(c, "text"))


async def test_lists_all_primitives(connect):
    async with connect() as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert tools == {"list_files", "read_file", "write_file", "save_memory", "forget_memory"}
        prompts = {p.name for p in (await client.list_prompts()).prompts}
        assert prompts == {"daily_review", "summarize_file"}
        uris = {r.uri for r in (await client.list_resources()).resources}
        assert "memory://facts" in uris
        templates = {t.uri_template for t in (await client.list_resource_templates()).resource_templates}
        assert "workspace://{filename}" in templates


async def test_write_read_list(connect):
    async with connect() as client:
        assert text(await client.call_tool("list_files", {})) == "(empty)"
        await client.call_tool("write_file", {"filename": "todo.md", "content": "buy milk"})
        assert text(await client.call_tool("read_file", {"filename": "todo.md"})) == "buy milk"
        assert text(await client.call_tool("list_files", {})) == "todo.md"


async def test_missing_file(connect):
    async with connect() as client:
        assert "no file named" in text(await client.call_tool("read_file", {"filename": "nope.md"}))


@pytest.mark.parametrize("name", ["../secret.txt", "../../etc/passwd", "/etc/passwd"])
async def test_path_traversal_rejected(connect, name):
    async with connect() as client:
        assert "outside the workspace" in text(await client.call_tool("read_file", {"filename": name}))
        assert "outside the workspace" in text(
            await client.call_tool("write_file", {"filename": name, "content": "x"})
        )


async def test_memory_save_dedupe_forget_and_resource(connect):
    async with connect() as client:
        await client.call_tool("save_memory", {"fact": "Likes tea"})
        assert "already known" in text(await client.call_tool("save_memory", {"fact": "likes tea"}))
        await client.call_tool("save_memory", {"fact": "Lives in Delhi"})

        facts = (await client.read_resource("memory://facts")).contents[0].text
        assert facts == "- Likes tea\n- Lives in Delhi\n"

        assert "removed 1" in text(await client.call_tool("forget_memory", {"fragment": "TEA"}))
        facts = (await client.read_resource("memory://facts")).contents[0].text
        assert facts == "- Lives in Delhi\n"


async def test_workspace_resource_and_prompts(connect):
    async with connect() as client:
        await client.call_tool("write_file", {"filename": "a.md", "content": "hello"})
        assert (await client.read_resource("workspace://a.md")).contents[0].text == "hello"

        prompt = await client.get_prompt("summarize_file", {"filename": "a.md"})
        assert "a.md" in prompt.messages[0].content.text
