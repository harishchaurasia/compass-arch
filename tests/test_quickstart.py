"""Tests for the one-call public API (compass.run).

Uses a fake agent (an object with .invoke) so the mapping from LangGraph state to
GateResult is tested without any model or API key.
"""

from langchain_core.messages import AIMessage

from compass import GateResult, run
from compass.agent_compass import CompassAction, CompassStep


def _step(tool: str, confidence: float, risk: str = "high") -> CompassStep:
    return CompassStep(
        reasoning="r",
        action=CompassAction(tool=tool, args={}),
        confidence=confidence,
        risk_level=risk,
    )


class _FakeAgent:
    def __init__(self, final_state):
        self._final_state = final_state
        self.seen_state = None

    def invoke(self, state):
        self.seen_state = state
        return self._final_state


def test_run_maps_state_to_result():
    agent = _FakeAgent(
        {
            "messages": [AIMessage(content="all done")],
            "steps": [_step("delete_file", 0.9), _step("list_files", 0.8, "low")],
            "abstained": False,
        }
    )
    result = run(agent, "do the thing")

    assert isinstance(result, GateResult)
    assert result.final_message == "all done"
    assert result.abstained is False
    assert result.steps == 2
    assert result.confidences == [0.9, 0.8]
    # success_probs are the calibrated gate values, one per step
    assert len(result.success_probs) == 2
    # per-step effective risk is surfaced for inspection
    assert result.risk_levels == ["high", "low"]


def test_run_reports_abstention():
    agent = _FakeAgent(
        {
            "messages": [AIMessage(content="ABSTAINING: ...")],
            "steps": [_step("delete_file", 1.0)],
            "abstained": True,
        }
    )
    assert run(agent, "delete everything").abstained is True


def test_run_initializes_compass_state_keys():
    agent = _FakeAgent(
        {"messages": [AIMessage(content="x")], "steps": [], "abstained": False}
    )
    run(agent, "hi")
    # the helper must seed the internal keys the graph expects
    for key in (
        "messages",
        "steps",
        "abstained",
        "self_verify_count",
        "verified_action",
    ):
        assert key in agent.seen_state
