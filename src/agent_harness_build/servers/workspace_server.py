"""MCP server exposing the user's workspace folder and long-term memory.

Tools     - list_files, read_file, write_file, save_memory, forget_memory
Resources - memory://facts, workspace://{filename}
Prompts   - daily_review, summarize_file

Runs over stdio, so nothing here may print to stdout.
"""

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from agent_harness_build import paths

mcp = MCPServer(
    "workspace",
    instructions="Read and write the user's workspace files and manage their long-term memory.",
)


def _safe_path(filename: str) -> Path | None:
    """Resolve filename inside the workspace; None if it would escape it."""
    root = paths.WORKSPACE.resolve()
    path = (root / filename).resolve()
    return path if path.is_relative_to(root) else None


def _read_memory() -> str:
    if paths.MEMORY_FILE.is_file():
        return paths.MEMORY_FILE.read_text(encoding="utf-8")
    return "(nothing saved yet)"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_files() -> str:
    """List the files in the user's workspace folder."""
    paths.WORKSPACE.mkdir(parents=True, exist_ok=True)
    return "\n".join(sorted(p.name for p in paths.WORKSPACE.iterdir())) or "(empty)"


@mcp.tool()
def read_file(filename: str) -> str:
    """Read one file from the user's workspace folder."""
    path = _safe_path(filename)
    if path is None:
        return "error: path is outside the workspace"
    if not path.is_file():
        return f"error: no file named {filename}"
    return path.read_text(encoding="utf-8")


@mcp.tool()
def write_file(filename: str, content: str) -> str:
    """Write (or overwrite) a file in the user's workspace folder."""
    path = _safe_path(filename)
    if path is None:
        return "error: path is outside the workspace"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {filename}"


@mcp.tool()
def save_memory(fact: str) -> str:
    """Save one short fact about the user to long-term memory.

    Use whenever you learn something lasting: their name, preferences,
    projects, recurring tasks. Duplicates are ignored. If the new fact REPLACES
    an older one (moved city, new job, changed email, no longer likes
    something), you must also remove the outdated fact with forget_memory,
    otherwise memory will contradict itself.
    """
    fact = fact.strip()
    others = [line for line in _read_memory().splitlines() if line.strip().startswith("-")]
    if fact.lower() in [line.removeprefix("-").strip().lower() for line in others]:
        return f"already known: {fact}"
    paths.MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with paths.MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    reply = f"saved: {fact}"
    if others:
        reply += (
            "\nOther facts already in memory:\n" + "\n".join(others)
            + "\nIf any of these are now outdated or contradicted by the new fact, "
            "remove them now with forget_memory."
        )
    return reply


@mcp.tool()
def forget_memory(fragment: str) -> str:
    """Remove every remembered fact containing the given text (case-insensitive).

    The text must appear verbatim in the fact, so use a short distinctive word
    (e.g. "Delhi"), not a paraphrase. If nothing matches, the current facts are
    returned so you can retry.
    """
    if not paths.MEMORY_FILE.is_file() or not fragment.strip():
        return "nothing removed: memory is empty"
    lines = paths.MEMORY_FILE.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if fragment.lower() not in line.lower()]
    removed = len(lines) - len(kept)
    if removed == 0:
        return f"nothing removed: no fact contains {fragment!r}. Current facts:\n" + "\n".join(lines)
    paths.MEMORY_FILE.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    return f"removed {removed} fact(s)"


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("memory://facts", mime_type="text/markdown")
def memory_facts() -> str:
    """Everything remembered about the user."""
    return _read_memory()


@mcp.resource("workspace://{filename}")
def workspace_file(filename: str) -> str:
    """One file from the user's workspace folder."""
    return read_file(filename)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@mcp.prompt()
def daily_review() -> str:
    """Review the user's notes and todos and propose today's priorities."""
    return (
        "List the files in my workspace, read my todo lists and notes, then give me "
        "a short review: what is most urgent, what can wait, and one thing I might "
        "be forgetting. Be concise."
    )


@mcp.prompt()
def summarize_file(filename: str) -> str:
    """Summarize one workspace file."""
    return f"Read the workspace file '{filename}' and summarize it in three bullet points."


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
