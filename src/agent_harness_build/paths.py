"""Filesystem locations shared by the harness and the MCP servers.

Both sides import this module so they can't disagree about where the
workspace and memory file live. Env vars override the defaults (used by tests).
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent

WORKSPACE = Path(os.getenv("AGENT_WORKSPACE", ROOT / "workspace"))
MEMORY_FILE = Path(os.getenv("AGENT_MEMORY_FILE", ROOT / "memory.md"))
MCP_CONFIG = ROOT / "mcp_servers.json"
