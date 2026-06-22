"""Tests for retail LangChain tools.

All tests use the _db fixture to inject fresh in-memory data before each test,
so mutations from one test (e.g. cancel_pending_order) don't bleed into the next.
"""
import pytest
import compass.tools.retail_db as db
from compass.tools.retail import (
    cancel_pending_order,
    find_user_id_by_name_zip,
    find_user_id_by_email,
    get_order_details,
    get_user_details,
    modify_pending_order_address,
    return_delivered_order_items,
)

# ── shared test data ──────────────────────────────────────────────────────────

USERS = {
    "sofia_chen_10001": {
        "name": {"first_name": "Sofia", "last_name": "Chen"},
        "email": "sofia@example.com",
        "zip": "10001",
        "address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"},
        "payment_methods": {"card_1234": {"type": "credit_card", "last4": "1234"}},
        "orders": ["#W1111111", "#W1111112"],
    },
    "marcus_johnson_90210": {
        "name": {"first_name": "Marcus", "last_name": "Johnson"},
        "email": "marcus@example.com",
        "zip": "90210",
        "address": {"street": "456 Oak Ave", "city": "Beverly Hills", "state": "CA", "zip": "90210"},
        "payment_methods": {"card_5678": {"type": "credit_card", "last4": "5678"}},
        "orders": ["#W2222222"],
    },
}

ORDERS = {
    "#W1111111": {
        "order_id": "#W1111111",
        "user_id": "sofia_chen_10001",
        "status": "pending",
        "items": [{"item_id": "item_lp_001", "name": "Laptop Pro 15", "price": 1299.99}],
        "address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"},
        "payment_method_id": "card_1234",
    },
    "#W1111112": {
        "order_id": "#W1111112",
        "user_id": "sofia_chen_10001",
        "status": "pending",
        "items": [{"item_id": "item_hp_001", "name": "Wireless Headphones", "price": 199.99}],
        "address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"},
        "payment_method_id": "card_1234",
    },
    "#W2222222": {
        "order_id": "#W2222222",
        "user_id": "marcus_johnson_90210",
        "status": "delivered",
        "items": [{"item_id": "item_hp_002", "name": "Noise-Cancelling Headphones", "price": 349.99}],
        "address": {"street": "456 Oak Ave", "city": "Beverly Hills", "state": "CA", "zip": "90210"},
        "payment_method_id": "card_5678",
    },
}


@pytest.fixture(autouse=True)
def _db(monkeypatch):
    """Replace module-level dicts with fresh copies before every test."""
    import copy
    monkeypatch.setattr(db, "USERS", copy.deepcopy(USERS))
    monkeypatch.setattr(db, "ORDERS", copy.deepcopy(ORDERS))


# ── find_user_id_by_name_zip ──────────────────────────────────────────────────

def test_find_user_by_name_zip_returns_id():
    result = find_user_id_by_name_zip.invoke({"first_name": "Sofia", "last_name": "Chen", "zip": "10001"})
    assert result == "sofia_chen_10001"

def test_find_user_by_name_zip_not_found():
    result = find_user_id_by_name_zip.invoke({"first_name": "Nobody", "last_name": "Here", "zip": "00000"})
    assert "not found" in result.lower()

# ── find_user_id_by_email ────────────────────────────────────────────────────

def test_find_user_by_email_returns_id():
    result = find_user_id_by_email.invoke({"email": "marcus@example.com"})
    assert result == "marcus_johnson_90210"

def test_find_user_by_email_not_found():
    result = find_user_id_by_email.invoke({"email": "ghost@nowhere.com"})
    assert "not found" in result.lower()

# ── get_user_details ──────────────────────────────────────────────────────────

def test_get_user_details_returns_profile():
    result = get_user_details.invoke({"user_id": "sofia_chen_10001"})
    assert "Sofia" in result
    assert "10001" in result

def test_get_user_details_invalid_id():
    result = get_user_details.invoke({"user_id": "ghost_user"})
    assert "not found" in result.lower()

# ── get_order_details ─────────────────────────────────────────────────────────

def test_get_order_details_returns_order():
    result = get_order_details.invoke({"order_id": "#W1111111"})
    assert "pending" in result.lower()
    assert "Laptop" in result

def test_get_order_details_invalid_order():
    result = get_order_details.invoke({"order_id": "#W9999999"})
    assert "not found" in result.lower()

# ── cancel_pending_order ──────────────────────────────────────────────────────

def test_cancel_pending_order_succeeds():
    result = cancel_pending_order.invoke({"order_id": "#W1111111"})
    assert "cancelled" in result.lower()
    assert db.ORDERS["#W1111111"]["status"] == "cancelled"

def test_cancel_delivered_order_fails():
    result = cancel_pending_order.invoke({"order_id": "#W2222222"})
    assert "cannot" in result.lower() or "error" in result.lower()
    assert db.ORDERS["#W2222222"]["status"] == "delivered"  # unchanged

# ── return_delivered_order_items ──────────────────────────────────────────────

def test_return_delivered_items_succeeds():
    result = return_delivered_order_items.invoke({
        "order_id": "#W2222222",
        "item_ids": ["item_hp_002"],
        "payment_method_id": "card_5678",
    })
    assert "return" in result.lower()

def test_return_pending_order_fails():
    result = return_delivered_order_items.invoke({
        "order_id": "#W1111111",
        "item_ids": ["item_lp_001"],
        "payment_method_id": "card_1234",
    })
    assert "cannot" in result.lower() or "error" in result.lower()

# ── modify_pending_order_address ──────────────────────────────────────────────

def test_modify_address_on_pending_order():
    new_addr = {"street": "789 New Rd", "city": "Brooklyn", "state": "NY", "zip": "11201"}
    result = modify_pending_order_address.invoke({"order_id": "#W1111112", "address": new_addr})
    assert "updated" in result.lower()
    assert db.ORDERS["#W1111112"]["address"]["street"] == "789 New Rd"

def test_modify_address_on_delivered_order_fails():
    new_addr = {"street": "789 New Rd", "city": "Brooklyn", "state": "NY", "zip": "11201"}
    result = modify_pending_order_address.invoke({"order_id": "#W2222222", "address": new_addr})
    assert "cannot" in result.lower() or "error" in result.lower()
