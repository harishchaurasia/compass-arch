"""End-to-end test of the MCP runner + grader against the real fs MCP server.

Uses a stub agent that performs scripted tool calls, so the runner / grading /
bridge integration is exercised deterministically without an LLM. (The full
Compass graph's execution path is covered by test_agent_compass.)
"""
import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from compass.mcp.bridge import MCPToolServer
from eval.mcp_runner import run_mcp_trial

ROOT = Path(__file__).parent.parent
SEED = json.loads((ROOT / "tasks" / "custom_mcp" / "seed.json").read_text())
TASKS = {t["id"]: t for t in json.loads((ROOT / "tasks" / "custom_mcp" / "tasks.json").read_text())}


class StubAgent:
    """Fake vanilla agent: replays scripted (tool, args) calls, returns final state."""

    def __init__(self, server: MCPToolServer, actions: list[tuple[str, dict]]):
        self._tools = {t.name: t for t in server.agent_tools()}
        self._actions = actions

    def invoke(self, state: dict) -> dict:
        for name, args in self._actions:
            self._tools[name].invoke(args)
        return {
            "messages": state["messages"] + [AIMessage(content="done")],
            "steps": len(self._actions),
            "abstained": False,
        }


@pytest.fixture
def server():
    srv = MCPToolServer(sys.executable, ["-m", "compass.mcp.fs_server"])
    yield srv
    srv.close()


def test_correct_deletion_succeeds_without_compound_failure(server):
    task = TASKS["delete-payments-backup"]
    agent = StubAgent(server, [("delete_file", {"path": "/services/payments/config.yaml.bak"})])
    r = run_mcp_trial(task, agent, "vanilla", "stub", server, SEED)

    assert r.success is True
    assert r.mutated_order_ids == ["/services/payments/config.yaml.bak"]
    assert r.task_id == "mcp_fs_delete-payments-backup"


def test_deleting_the_live_config_is_a_compound_failure(server):
    task = TASKS["delete-payments-backup"]
    # the cascading trap: delete the LIVE config instead of the backup
    agent = StubAgent(server, [("delete_file", {"path": "/services/payments/config.yaml"})])
    r = run_mcp_trial(task, agent, "vanilla", "stub", server, SEED)

    assert r.success is False
    assert r.mutated_order_ids == ["/services/payments/config.yaml"]  # destructive + wrong


def test_read_only_task_needs_no_mutation(server):
    task = TASKS["read-auth-port"]
    agent = StubAgent(server, [("read_file", {"path": "/services/auth/config.yaml"})])
    # stub returns "done", which lacks the required "9000" output -> not success,
    # but crucially it mutated nothing, so it is NOT a compound failure
    r = run_mcp_trial(task, agent, "vanilla", "stub", server, SEED)

    assert r.mutated_order_ids == []
    assert r.success is False  # "done" has no "9000"


def test_read_only_answer_succeeds_when_output_present(server):
    task = TASKS["read-auth-port"]

    class AnsweringAgent(StubAgent):
        def invoke(self, state):
            return {"messages": state["messages"] + [AIMessage(content="port 9000")],
                    "steps": 1, "abstained": False}

    r = run_mcp_trial(task, AnsweringAgent(server, []), "vanilla", "stub", server, SEED)
    assert r.success is True
    assert r.mutated_order_ids == []
