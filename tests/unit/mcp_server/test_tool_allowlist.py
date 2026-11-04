# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
