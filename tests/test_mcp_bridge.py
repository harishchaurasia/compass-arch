"""Integration test: the sync bridge drives the real fs MCP server over stdio.

Spawns the server as a subprocess, so it exercises the whole MCP path
(JSON-RPC over stdio, persistent session, content-block flattening).
"""
import sys

import pytest

from compass.mcp.bridge import MCPToolServer


@pytest.fixture
def server():
    srv = MCPToolServer(sys.executable, ["-m", "compass.mcp.fs_server"])
    yield srv
    srv.close()


def test_agent_tools_exclude_control_tools(server):
    names = {t.name for t in server.agent_tools()}
    assert names == {"list_dir", "read_file", "find_files",
                     "write_file", "delete_file", "move_file"}
    assert not any(n.startswith("_") for n in names)


def test_state_persists_across_calls_within_session(server):
    server.reset({"/a.txt": "A", "/b.txt": "B"})
    tools = {t.name: t for t in server.agent_tools()}

    assert tools["read_file"].invoke({"path": "/a.txt"}) == "A"
    tools["write_file"].invoke({"path": "/a.txt", "content": "A2"})  # overwrite
    tools["delete_file"].invoke({"path": "/b.txt"})

    world = server.dump()
    assert world == {"/a.txt": "A2"}  # state shared across the whole session


def test_reset_clears_previous_world(server):
    server.reset({"/x.txt": "1"})
    server.reset({"/y.txt": "2"})
    assert server.dump() == {"/y.txt": "2"}
