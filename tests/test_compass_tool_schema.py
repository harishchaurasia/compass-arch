"""Verify the Compass system prompt exposes exact tool parameter names,
and that tools handle common model mis-formatting gracefully.
"""
from langchain_core.tools import tool
import compass.tools.retail_db as db
import copy
import pytest

from compass.agent_compass import _system_prompt
from compass.tools.retail import get_order_details

ORDERS = {
    "#W1111111": {
        "order_id": "#W1111111", "user_id": "u1", "status": "pending",
        "items": [{"item_id": "i1", "name": "Laptop", "price": 999.0}],
        "address": {}, "payment_method_id": "card_1",
    }
}


@pytest.fixture(autouse=True)
def _db(monkeypatch):
    monkeypatch.setattr(db, "ORDERS", copy.deepcopy(ORDERS))
    monkeypatch.setattr(db, "USERS", {})


# ── system prompt includes exact param names ──────────────────────────────────

def test_system_prompt_includes_exact_param_names():
    @tool
    def lookup(user_id: str, zip: str) -> str:
        """Find a user."""
        return "ok"

    prompt = _system_prompt([lookup])
    assert "user_id" in prompt.content
    assert "zip" in prompt.content
    # must NOT say zip_code — that's the wrong name
    assert "zip_code" not in prompt.content


def test_system_prompt_lists_all_tools():
    @tool
    def tool_a(x: str) -> str:
        """Tool A."""
        return x

    @tool
    def tool_b(y: int) -> str:
        """Tool B."""
        return str(y)

    prompt = _system_prompt([tool_a, tool_b])
    assert "tool_a" in prompt.content
    assert "tool_b" in prompt.content


# ── get_order_details normalizes missing # prefix ────────────────────────────

def test_get_order_details_with_hash_prefix():
    result = get_order_details.invoke({"order_id": "#W1111111"})
    assert "Laptop" in result


def test_get_order_details_without_hash_prefix():
    """Model sometimes omits # — tool should still work."""
    result = get_order_details.invoke({"order_id": "W1111111"})
    assert "Laptop" in result
