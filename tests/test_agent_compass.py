"""Tests for the Compass calibrated agent.

FakeCompassModel supports with_structured_output() and returns CompassStep
objects in sequence — no real LLM calls.
"""
import copy
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

import compass.tools.retail_db as db
from compass.agent_compass import CompassState, CompassStep, build_compass_agent

# ── fake model ────────────────────────────────────────────────────────────────

class FakeCompassModel:
    """Yields CompassStep objects from with_structured_output().invoke()."""

    def __init__(self, steps: list[CompassStep]):
        self._iter = iter(steps)

    def with_structured_output(self, schema):
        return self  # invoke() returns CompassStep, not AIMessage

    def invoke(self, messages) -> CompassStep:
        return next(self._iter)


# ── test tool ─────────────────────────────────────────────────────────────────

@tool
def delete_record(record_id: str) -> str:
    """Delete a record permanently."""
    return f"Record {record_id} deleted."


@tool
def read_record(record_id: str) -> str:
    """Read a record (read-only, safe)."""
    return f"Record {record_id}: some data."


# ── tests ─────────────────────────────────────────────────────────────────────

def test_low_risk_tool_executes():
    """Low risk action should always be executed regardless of confidence."""
    steps = [
        CompassStep(
            reasoning="Safe read operation.",
            action={"tool": "read_record", "args": {"record_id": "r1"}},
            confidence=0.3,  # low confidence, but low risk → still execute
            risk_level="low",
        ),
        CompassStep(
            reasoning="Done.",
            action={"final_answer": "Read complete."},
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [read_record, delete_record])
    state = agent.invoke({"messages": [HumanMessage(content="Read r1")], "steps": [], "abstained": False})

    assert state["abstained"] is False
    assert "Read complete." in state["messages"][-1].content


def test_high_risk_low_confidence_abstains():
    """High risk + low confidence → agent must abstain without calling the tool."""
    called = []

    @tool
    def risky_delete(record_id: str) -> str:
        """Delete permanently."""
        called.append(record_id)
        return "deleted"

    steps = [
        CompassStep(
            reasoning="About to delete something important.",
            action={"tool": "risky_delete", "args": {"record_id": "r99"}},
            confidence=0.5,  # below T_HIGH=0.8 → abstain
            risk_level="high",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [risky_delete])
    state = agent.invoke({"messages": [HumanMessage(content="Delete r99")], "steps": [], "abstained": False})

    assert state["abstained"] is True
    assert called == []  # tool was NEVER called


def test_medium_risk_low_confidence_self_verifies_then_executes():
    """Medium risk + low confidence → self-verify, then if confidence improves → execute."""
    steps = [
        # step 1: low confidence → self_verify
        CompassStep(
            reasoning="Not sure about this.",
            action={"tool": "read_record", "args": {"record_id": "r2"}},
            confidence=0.4,  # below T_MED=0.6 → self_verify
            risk_level="medium",
        ),
        # step 2: after seeing self-verify prompt, confidence is higher → execute
        CompassStep(
            reasoning="Re-read context, now confident.",
            action={"tool": "read_record", "args": {"record_id": "r2"}},
            confidence=0.75,  # above T_MED=0.6 → execute
            risk_level="medium",
        ),
        CompassStep(
            reasoning="Done.",
            action={"final_answer": "Read r2 successfully."},
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [read_record])
    state = agent.invoke({"messages": [HumanMessage(content="Read r2")], "steps": [], "abstained": False})

    assert state["abstained"] is False
    assert "Read r2 successfully." in state["messages"][-1].content
    # self-verify prompt should appear somewhere in message history
    all_content = " ".join(m.content for m in state["messages"] if hasattr(m, "content"))
    assert "verify" in all_content.lower() or "confidence" in all_content.lower()


def test_final_answer_ends_without_tool():
    """If first step is a final_answer, agent should return it without calling any tool."""
    steps = [
        CompassStep(
            reasoning="I already know the answer.",
            action={"final_answer": "The order status is pending."},
            confidence=0.95,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [read_record])
    state = agent.invoke({"messages": [HumanMessage(content="What is the order status?")], "steps": [], "abstained": False})

    assert "pending" in state["messages"][-1].content
    assert state["abstained"] is False
