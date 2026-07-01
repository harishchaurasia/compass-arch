"""Tests for the τ-bench trial runner.

We test run_trial in isolation using FakeModel — no real API calls.
The success check is: does the final message contain the expected_outcome substring?
"""
import copy
import pytest
from langchain_core.messages import AIMessage, HumanMessage
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
