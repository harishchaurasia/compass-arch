"""Retail domain tools for τ-bench tasks."""
import json
from langchain_core.tools import tool
import compass.tools.retail_db as db


def _json(obj: dict) -> str:
    return json.dumps(obj, indent=2)


@tool
def find_user_id_by_name_zip(first_name: str, last_name: str, zip: str) -> str:
    """Find a user's ID by their first name, last name, and zip code."""
    for uid, u in db.USERS.items():
        n = u["name"]
        if (n["first_name"].lower() == first_name.lower()
                and n["last_name"].lower() == last_name.lower()
                and u["zip"] == zip):
            return uid
    return f"User not found: {first_name} {last_name} {zip}"


@tool
def find_user_id_by_email(email: str) -> str:
    """Find a user's ID by their email address."""
    for uid, u in db.USERS.items():
        if u["email"].lower() == email.lower():
            return uid
    return f"User not found: {email}"


@tool
def get_user_details(user_id: str) -> str:
    """Get full profile details for a user including address, payment methods, and orders."""
    if user_id not in db.USERS:
        return f"User not found: {user_id}"
    return _json(db.USERS[user_id])


@tool
def get_order_details(order_id: str) -> str:
    """Get full details for an order including items, status, and address."""
    if order_id not in db.ORDERS:
        return f"Order not found: {order_id}"
    return _json(db.ORDERS[order_id])


@tool
def cancel_pending_order(order_id: str) -> str:
    """Cancel a pending order. Only works if the order status is 'pending'."""
    if order_id not in db.ORDERS:
        return f"Order not found: {order_id}"
    order = db.ORDERS[order_id]
    if order["status"] != "pending":
        return f"Cannot cancel order {order_id}: status is '{order['status']}', not 'pending'."
    order["status"] = "cancelled"
    return f"Order {order_id} has been cancelled. Refund will be issued to the original payment method."


@tool
def return_delivered_order_items(order_id: str, item_ids: list[str], payment_method_id: str) -> str:
    """Return one or more items from a delivered order."""
    if order_id not in db.ORDERS:
        return f"Order not found: {order_id}"
    order = db.ORDERS[order_id]
    if order["status"] != "delivered":
        return f"Cannot return items from order {order_id}: status is '{order['status']}', not 'delivered'."
    return f"Return processed for order {order_id}, items {item_ids}. Refund to payment method {payment_method_id}."


@tool
def modify_pending_order_address(order_id: str, address: dict) -> str:
    """Update the shipping address on a pending order."""
    if order_id not in db.ORDERS:
        return f"Order not found: {order_id}"
    order = db.ORDERS[order_id]
    if order["status"] != "pending":
        return f"Cannot modify address on order {order_id}: status is '{order['status']}', not 'pending'."
    order["address"] = address
    return f"Shipping address for order {order_id} updated to: {_json(address)}"
