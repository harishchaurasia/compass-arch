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
