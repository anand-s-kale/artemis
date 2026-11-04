"""ARTEMIS_MCP_TOOLS allowlist applied to the FastMCP server at startup."""

import pytest
from mcp.server.fastmcp import FastMCP

from mcp_server.base import mcp as global_mcp
from mcp_server.server import _allowlist_from_env, apply_tool_allowlist

pytestmark = pytest.mark.asyncio

ALL_TOOLS = {
    "mobile_run_task",
    "mobile_manage_task",
    "mobile_get_device_state",
    "mobile_inspect_trace",
    "mobile_diagnose",
    "mobile_observe",
    "mobile_act",
    "mobile_session_end",
}


def _fresh_server() -> FastMCP:
    """A throwaway server with the real tools, so the shared instance stays intact."""
    server = FastMCP("allowlist-test")
    for tool in global_mcp._tool_manager.list_tools():
        server.add_tool(tool.fn, name=tool.name)
    return server


async def test_allowlist_keeps_only_named_tools(monkeypatch):
    monkeypatch.setenv("ARTEMIS_MCP_TOOLS", "mobile_observe,mobile_act,mobile_session_end")
    server = _fresh_server()
    apply_tool_allowlist(server, _allowlist_from_env())
    assert {t.name for t in await server.list_tools()} == {
        "mobile_observe",
        "mobile_act",
        "mobile_session_end",
    }


async def test_no_allowlist_keeps_all_eight(monkeypatch):
    monkeypatch.delenv("ARTEMIS_MCP_TOOLS", raising=False)
    server = _fresh_server()
    apply_tool_allowlist(server, _allowlist_from_env())
    assert {t.name for t in await server.list_tools()} == ALL_TOOLS


async def test_unknown_name_in_allowlist_fails_at_startup():
    server = _fresh_server()
    with pytest.raises(ValueError, match="mobile_bogus"):
        apply_tool_allowlist(server, {"mobile_observe", "mobile_bogus"})
