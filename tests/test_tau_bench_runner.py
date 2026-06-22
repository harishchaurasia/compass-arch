"""End-to-end test: vanilla agent runs a τ-bench task using retail tools.

We still use FakeModel so no API calls are made — but now the fake model
produces realistic tool-call sequences matching our 3 tasks.
"""
import json
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from compass.agent_vanilla import build_vanilla_agent
from compass.tools.retail import (
    cancel_pending_order,
    find_user_id_by_name_zip,
    get_order_details,
)
import compass.tools.retail_db as db
import copy

USERS = {
    "sofia_chen_10001": {
        "name": {"first_name": "Sofia", "last_name": "Chen"},
        "email": "sofia@example.com",
        "zip": "10001",
        "address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"},
        "payment_methods": {"card_1234": {"type": "credit_card", "last4": "1234"}},
        "orders": ["#W1111111"],
    }
}
ORDERS = {
    "#W1111111": {
        "order_id": "#W1111111",
        "user_id": "sofia_chen_10001",
        "status": "pending",
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


def test_vanilla_agent_cancels_order_end_to_end():
    """Agent should find user, check order, cancel it — using real retail tools."""
    responses = [
        # step 1: find the user
        AIMessage(content="", tool_calls=[{
            "name": "find_user_id_by_name_zip",
            "args": {"first_name": "Sofia", "last_name": "Chen", "zip": "10001"},
            "id": "call_1", "type": "tool_call",
        }]),
        # step 2: check order details
        AIMessage(content="", tool_calls=[{
            "name": "get_order_details",
            "args": {"order_id": "#W1111111"},
            "id": "call_2", "type": "tool_call",
        }]),
        # step 3: cancel it
        AIMessage(content="", tool_calls=[{
            "name": "cancel_pending_order",
            "args": {"order_id": "#W1111111"},
            "id": "call_3", "type": "tool_call",
        }]),
        # step 4: final answer
        AIMessage(content="Order #W1111111 has been cancelled successfully."),
    ]

    tools = [find_user_id_by_name_zip, get_order_details, cancel_pending_order]
    agent = build_vanilla_agent(FakeModel(responses), tools)

    task_instruction = "Cancel my pending laptop order #W1111111. I'm Sofia Chen, zip 10001."
    state = agent.invoke({"messages": [HumanMessage(content=task_instruction)], "steps": 0})

    # order was actually cancelled in the db
    assert db.ORDERS["#W1111111"]["status"] == "cancelled"
    # agent gave a final answer
    assert "cancelled" in state["messages"][-1].content.lower()
    # tools were actually called (ToolMessages in history)
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 3
