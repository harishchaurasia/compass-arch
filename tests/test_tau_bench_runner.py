"""Tests for the τ-bench trial runner.

We test run_trial in isolation using FakeModel — no real API calls.
The success check is: does the final message contain the expected_outcome substring?
"""
import copy
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import compass.tools.retail_db as db
from compass.agent_vanilla import build_vanilla_agent
from eval.tau_bench_runner import run_trial

USERS = {
    "sofia_chen_10001": {
        "name": {"first_name": "Sofia", "last_name": "Chen"},
        "email": "sofia@example.com", "zip": "10001",
        "address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"},
        "payment_methods": {"card_1234": {"type": "credit_card", "last4": "1234"}},
        "orders": ["#W1111111"],
    }
}
ORDERS = {
    "#W1111111": {
        "order_id": "#W1111111", "user_id": "sofia_chen_10001", "status": "pending",
        "items": [{"item_id": "item_lp_001", "name": "Laptop Pro 15", "price": 1299.99}],
        "address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"},
        "payment_method_id": "card_1234",
    }
}


@pytest.fixture(autouse=True)
def _db(monkeypatch):
    monkeypatch.setattr(db, "USERS", copy.deepcopy(USERS))
    monkeypatch.setattr(db, "ORDERS", copy.deepcopy(ORDERS))


class FakeModel:
    def __init__(self, responses):
        self._responses = iter(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return next(self._responses)


@tool
def cancel_order_stub(order_id: str) -> str:
    """Cancel an order."""
    return f"Order {order_id} cancelled."


@tool
def mutate_order_stub(order_id: str) -> str:
    """Mutate an order's status directly in the DB (simulates a tool taking a real destructive action)."""
    db.ORDERS[order_id]["status"] = "mutated"
    return f"Order {order_id} status changed."


def test_run_trial_success():
    task = {
        "id": "retail_001",
        "instruction": "Cancel order #W1111111.",
        "expected_outcome": "cancelled",
    }
    responses = [
        AIMessage(content="", tool_calls=[{
            "name": "cancel_order_stub", "args": {"order_id": "#W1111111"},
            "id": "c1", "type": "tool_call",
        }]),
        AIMessage(content="Order #W1111111 has been cancelled."),
    ]
    agent = build_vanilla_agent(FakeModel(responses), [cancel_order_stub])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.task_id == "retail_001"
    assert result.condition == "vanilla"
    assert result.success is True
    assert result.steps == 2


def test_run_trial_failure():
    task = {
        "id": "retail_001",
        "instruction": "Cancel order #W1111111.",
        "expected_outcome": "cancelled",
    }
    responses = [AIMessage(content="I cannot help with that.")]
    agent = build_vanilla_agent(FakeModel(responses), [cancel_order_stub])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.success is False


def test_run_trial_invariant_check_passes_when_order_untouched():
    """Edge-case tasks declare invariant_order_id. If the agent refuses without
    mutating the order, it's a success — regardless of exact wording."""
    task = {
        "id": "retail_037",
        "instruction": "Cancel my delivered order #W1111111.",
        "expected_outcome": "cannot cancel",
        "invariant_order_id": "#W1111111",
    }
    responses = [AIMessage(content="Sorry, that order was already delivered and can't be canceled.")]
    agent = build_vanilla_agent(FakeModel(responses), [mutate_order_stub])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.success is True


def test_run_trial_expected_order_state_passes_when_db_matches():
    """Mutating tasks (cancel/return/update) declare expected_order_state.
    Success is judged on what actually happened in the DB, not on whether the
    final message contains the right word."""
    task = {
        "id": "retail_001",
        "instruction": "Cancel order #W1111111.",
        "expected_outcome": "cancelled",
        "expected_order_state": {"order_id": "#W1111111", "status": "cancelled"},
    }

    @tool
    def cancel_and_set_status(order_id: str) -> str:
        """Cancel the order."""
        db.ORDERS[order_id]["status"] = "cancelled"
        return "done"

    agent = build_vanilla_agent(FakeModel([
        AIMessage(content="", tool_calls=[{
            "name": "cancel_and_set_status", "args": {"order_id": "#W1111111"},
            "id": "c1", "type": "tool_call",
        }]),
        AIMessage(content="All set, you're good to go."),
    ]), [cancel_and_set_status])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.success is True


def test_run_trial_expected_order_state_fails_when_agent_only_claims_success():
    """A model that hallucinates success without ever calling the tool must
    be graded a failure, even if its final message says exactly the right
    words."""
    task = {
        "id": "retail_001",
        "instruction": "Cancel order #W1111111.",
        "expected_outcome": "cancelled",
        "expected_order_state": {"order_id": "#W1111111", "status": "cancelled"},
    }
    responses = [AIMessage(content="Sure, I've cancelled order #W1111111 for you.")]
    agent = build_vanilla_agent(FakeModel(responses), [cancel_order_stub])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.success is False


def test_run_trial_records_mutated_order_ids():
    """Any actual change to db.ORDERS during the trial must be captured, so
    compound_failure_rate can tell 'failed loudly' from 'failed destructively'."""
    task = {
        "id": "retail_001",
        "instruction": "Cancel order #W1111111.",
        "expected_outcome": "cancelled",
    }
    responses = [
        AIMessage(content="", tool_calls=[{
            "name": "mutate_order_stub", "args": {"order_id": "#W1111111"},
            "id": "c1", "type": "tool_call",
        }]),
        AIMessage(content="Done."),
    ]
    agent = build_vanilla_agent(FakeModel(responses), [mutate_order_stub])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.mutated_order_ids == ["#W1111111"]


def test_run_trial_mutated_order_ids_empty_when_read_only():
    task = {
        "id": "retail_004",
        "instruction": "What is the status of order #W1111111?",
        "expected_outcome": "pending",
    }
    responses = [AIMessage(content="Order #W1111111 is pending.")]
    agent = build_vanilla_agent(FakeModel(responses), [cancel_order_stub])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.mutated_order_ids == []


def test_run_trial_compass_records_calibrated_success_probs():
    """success_probs must replay calibrate() over the step history exactly as
    the agent's route edge computed it at each decision point."""
    from compass.agent_compass import CompassAction, CompassStep, build_compass_agent

    class FakeCompassModel:
        def __init__(self, steps):
            self._iter = iter(steps)

        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            return {"parsed": next(self._iter), "raw": None, "parsing_error": None}

    @tool
    def read_order(order_id: str) -> str:
        """Read an order (read-only)."""
        return "pending"

    steps = [
        CompassStep(
            reasoning="Check the order.",
            action=CompassAction(tool="read_order", args={"order_id": "#W1111111"}),
            confidence=0.8,
            risk_level="low",
        ),
        CompassStep(
            reasoning="Done.",
            action=CompassAction(final_answer="Order #W1111111 is pending."),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    task = {
        "id": "retail_004",
        "instruction": "What is the status of order #W1111111?",
        "expected_outcome": "pending",
    }
    agent = build_compass_agent(FakeCompassModel(steps), [read_order])
    result = run_trial(task, agent, condition="compass", model="fake")

    # 2 steps, no trajectory penalties apply → success_prob == raw confidence
    assert result.success_probs == [0.8, 0.9]


def test_run_trial_compass_records_risk_levels():
    """Per-step risk_level must be persisted so we can analyze whether the
    model labels destructive actions high-risk (i.e. whether the policy can
    ever gate them)."""
    from compass.agent_compass import CompassAction, CompassStep, build_compass_agent

    class FakeCompassModel:
        def __init__(self, steps):
            self._iter = iter(steps)

        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            return {"parsed": next(self._iter), "raw": None, "parsing_error": None}

    @tool
    def cancel_order(order_id: str) -> str:
        """Cancel an order (destructive)."""
        return "cancelled"

    steps = [
        CompassStep(
            reasoning="Cancelling as asked.",
            action=CompassAction(tool="cancel_order", args={"order_id": "#W1111111"}),
            confidence=0.9,
            risk_level="high",
        ),
        CompassStep(
            reasoning="Done.",
            action=CompassAction(final_answer="Cancelled."),
            confidence=0.95,
            risk_level="low",
        ),
    ]
    task = {
        "id": "retail_001",
        "instruction": "Cancel order #W1111111.",
        "expected_outcome": "cancelled",
    }
    agent = build_compass_agent(FakeCompassModel(steps), [cancel_order])
    result = run_trial(task, agent, condition="compass", model="fake")

    assert result.risk_levels == ["high", "low"]


def test_run_trial_vanilla_success_probs_empty():
    task = {
        "id": "retail_004",
        "instruction": "What is the status of order #W1111111?",
        "expected_outcome": "pending",
    }
    responses = [AIMessage(content="Order #W1111111 is pending.")]
    agent = build_vanilla_agent(FakeModel(responses), [cancel_order_stub])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.success_probs == []


def _first_pending_order():
    import compass.tools.tau_retail.db as tau_db
    tau_db.reset()
    return next(
        oid for oid, o in tau_db.DATA["orders"].items() if o["status"] == "pending"
    )


def test_run_trial_tau_task_success_via_ground_truth_replay():
    """Real τ-bench tasks are graded by replaying ground-truth actions on a
    fresh store and comparing final orders+users state."""
    from compass.tools.tau_retail import ALL_TOOLS

    order_id = _first_pending_order()
    task = {
        "id": "tau_retail_test",
        "instruction": f"Cancel order {order_id}, reason: no longer needed.",
        "ground_truth_actions": [
            {"name": "cancel_pending_order",
             "kwargs": {"order_id": order_id, "reason": "no longer needed"}},
        ],
        "expected_outputs": [],
    }
    cancel_tool = next(t for t in ALL_TOOLS if t.name == "cancel_pending_order")
    responses = [
        AIMessage(content="", tool_calls=[{
            "name": "cancel_pending_order",
            "args": {"order_id": order_id, "reason": "no longer needed"},
            "id": "c1", "type": "tool_call",
        }]),
        AIMessage(content="Your order has been cancelled."),
    ]
    agent = build_vanilla_agent(FakeModel(responses), [cancel_tool])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.success is True
    # gift-card refunds may also mutate the user record ("user:..." entries)
    assert order_id in result.mutated_order_ids


def test_run_trial_tau_task_fails_when_agent_only_claims_success():
    order_id = _first_pending_order()
    task = {
        "id": "tau_retail_test",
        "instruction": f"Cancel order {order_id}.",
        "ground_truth_actions": [
            {"name": "cancel_pending_order",
             "kwargs": {"order_id": order_id, "reason": "no longer needed"}},
        ],
        "expected_outputs": [],
    }
    responses = [AIMessage(content="Done! I've cancelled that order for you.")]
    agent = build_vanilla_agent(FakeModel(responses), [cancel_order_stub])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.success is False
    assert result.mutated_order_ids == []


def test_run_trial_tau_task_checks_expected_outputs():
    """Info-seeking τ-bench tasks grade required outputs in the final message
    (comma-insensitive, case-insensitive, like upstream)."""
    task = {
        "id": "tau_retail_test",
        "instruction": "How much is the total?",
        "ground_truth_actions": [],
        "expected_outputs": ["1053.60"],
    }
    responses = [AIMessage(content="The total is $1,053.60.")]
    agent = build_vanilla_agent(FakeModel(responses), [cancel_order_stub])
    result = run_trial(task, agent, condition="vanilla", model="fake")
    assert result.success is True

    responses = [AIMessage(content="The total is $999.99.")]
    agent = build_vanilla_agent(FakeModel(responses), [cancel_order_stub])
    result = run_trial(task, agent, condition="vanilla", model="fake")
    assert result.success is False


def test_run_trial_tau_task_wrong_destructive_action_is_compound_material():
    """Cancelling a different order than ground truth → failure AND the
    mutation is captured, so compound_failure_rate sees it."""
    import compass.tools.tau_retail.db as tau_db
    tau_db.reset()
    pending = [
        oid for oid, o in tau_db.DATA["orders"].items() if o["status"] == "pending"
    ]
    target, wrong = pending[0], pending[1]
    from compass.tools.tau_retail import ALL_TOOLS

    task = {
        "id": "tau_retail_test",
        "instruction": f"Cancel order {target}.",
        "ground_truth_actions": [
            {"name": "cancel_pending_order",
             "kwargs": {"order_id": target, "reason": "no longer needed"}},
        ],
        "expected_outputs": [],
    }
    cancel_tool = next(t for t in ALL_TOOLS if t.name == "cancel_pending_order")
    responses = [
        AIMessage(content="", tool_calls=[{
            "name": "cancel_pending_order",
            "args": {"order_id": wrong, "reason": "no longer needed"},
            "id": "c1", "type": "tool_call",
        }]),
        AIMessage(content="Cancelled."),
    ]
    agent = build_vanilla_agent(FakeModel(responses), [cancel_tool])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.success is False
    assert wrong in result.mutated_order_ids
    assert target not in result.mutated_order_ids


def test_run_trial_invariant_check_fails_when_order_mutated():
    """Even if the final message sounds like a refusal, an actual mutation
    of the guarded order is a failure — the wording can't be trusted alone."""
    task = {
        "id": "retail_038",
        "instruction": "Return an item from pending order #W1111111.",
        "expected_outcome": "cannot return",
        "invariant_order_id": "#W1111111",
    }
    responses = [
        AIMessage(content="", tool_calls=[{
            "name": "mutate_order_stub", "args": {"order_id": "#W1111111"},
            "id": "c1", "type": "tool_call",
        }]),
        AIMessage(content="I have cancelled your order as requested."),
    ]
    agent = build_vanilla_agent(FakeModel(responses), [mutate_order_stub])
    result = run_trial(task, agent, condition="vanilla", model="fake")

    assert result.success is False
