"""Tests for trajectory feature extraction.

TrajectoryFeatures captures behavioural signals from the history of CompassStep
objects — things like how many steps have been taken, whether the agent is
looping, and how long since it last tried something new.
"""
from compass.agent_compass import CompassAction, CompassStep
from compass.trajectory import extract_features


def _tool_step(tool: str, confidence: float = 0.9, risk: str = "medium") -> CompassStep:
    return CompassStep(
        reasoning="doing stuff",
        action=CompassAction(tool=tool, args={}),
        confidence=confidence,
        risk_level=risk,
    )


def _final_step(confidence: float = 1.0) -> CompassStep:
    return CompassStep(
        reasoning="done",
        action=CompassAction(final_answer="all done"),
        confidence=confidence,
        risk_level="low",
    )


# ── step count ────────────────────────────────────────────────────────────────

def test_step_count_single():
    assert extract_features([_tool_step("find_user")]).step_count == 1


def test_step_count_many():
    assert extract_features([_tool_step("a")] * 7).step_count == 7


def test_step_count_includes_final_answer():
    steps = [_tool_step("find_user"), _final_step()]
    assert extract_features(steps).step_count == 2


# ── unique tools ──────────────────────────────────────────────────────────────

def test_unique_tools_all_different():
    steps = [_tool_step("a"), _tool_step("b"), _tool_step("c")]
    assert extract_features(steps).unique_tools_used == 3


def test_unique_tools_repeats_counted_once():
    steps = [_tool_step("a"), _tool_step("a"), _tool_step("b")]
    assert extract_features(steps).unique_tools_used == 2


def test_unique_tools_final_answer_not_counted():
    steps = [_tool_step("a"), _final_step()]
    assert extract_features(steps).unique_tools_used == 1


# ── oscillation ───────────────────────────────────────────────────────────────

def test_no_oscillation_too_few_tool_steps():
    """Fewer than 4 tool steps — not enough data to call it a loop."""
    steps = [_tool_step("a"), _tool_step("b"), _tool_step("a")]
    assert extract_features(steps).is_oscillating is False


def test_no_oscillation_varied_tools():
    steps = [_tool_step("a"), _tool_step("b"), _tool_step("c"), _tool_step("d")]
    assert extract_features(steps).is_oscillating is False


def test_oscillation_two_tools_alternating():
    """Last 4 tool steps cycle between only 2 tools."""
    steps = [
        _tool_step("find_user"),
        _tool_step("get_order"),
        _tool_step("find_user"),
        _tool_step("get_order"),
    ]
    assert extract_features(steps).is_oscillating is True


def test_oscillation_same_tool_repeated():
    steps = [_tool_step("get_order")] * 4
    assert extract_features(steps).is_oscillating is True


def test_oscillation_window_is_last_four_tool_steps():
    """Early variety then a loop — only the last 4 tool steps matter."""
    steps = [
        _tool_step("a"), _tool_step("b"), _tool_step("c"),  # varied start
        _tool_step("x"), _tool_step("y"), _tool_step("x"), _tool_step("y"),
    ]
    assert extract_features(steps).is_oscillating is True


def test_oscillation_ignores_final_answer_steps():
    """final_answer steps don't count toward the tool-step window."""
    steps = [
        _tool_step("a"), _tool_step("b"), _tool_step("a"),
        _final_step(),  # not a tool call — 3 tool steps total, below threshold
    ]
    assert extract_features(steps).is_oscillating is False


# ── steps since new tool ──────────────────────────────────────────────────────

def test_steps_since_new_tool_zero_on_first_step():
    assert extract_features([_tool_step("find_user")]).steps_since_new_tool == 0


def test_steps_since_new_tool_zero_when_last_step_is_new():
    steps = [_tool_step("a"), _tool_step("b"), _tool_step("c")]
    assert extract_features(steps).steps_since_new_tool == 0


def test_steps_since_new_tool_counts_repeats_at_end():
    # "a" new, "b" new, "b" repeat, "b" repeat → 2 trailing repeats
    steps = [_tool_step("a"), _tool_step("b"), _tool_step("b"), _tool_step("b")]
    assert extract_features(steps).steps_since_new_tool == 2


def test_steps_since_new_tool_final_answer_excluded():
    """final_answer step is not a tool call, so it doesn't increment the counter."""
    steps = [_tool_step("a"), _tool_step("a"), _final_step()]
    # 2 tool steps: first is "a" (new), second is "a" (repeat) → 1 since new
    assert extract_features(steps).steps_since_new_tool == 1
