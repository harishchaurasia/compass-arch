"""Tests for the vendored τ-bench retail environment adapter.

The fidelity contract: every one of the 115 vendored tasks' ground-truth
action sequences must replay through our adapted tools without error.
"""
import json
from pathlib import Path

import pytest

import compass.tools.tau_retail.db as tau_db
from compass.tools.tau_retail import ALL_TOOLS, TOOL_CLASSES, TOOL_RISK, replay_actions

TASKS = json.loads(
    (Path(__file__).parent.parent / "tasks" / "tau_bench" / "tasks_real.json").read_text()
)


@pytest.fixture(autouse=True)
def _fresh_db():
    tau_db.reset()


# ── data store ────────────────────────────────────────────────────────────────

def test_reset_loads_real_data():
    assert len(tau_db.DATA["users"]) == 500
    assert len(tau_db.DATA["orders"]) == 1000
    assert len(tau_db.DATA["products"]) == 50


def test_reset_restores_mutations():
    order_id = next(iter(tau_db.DATA["orders"]))
    tau_db.DATA["orders"][order_id]["status"] = "clobbered"
    tau_db.reset()
    assert tau_db.DATA["orders"][order_id]["status"] != "clobbered"


# ── adapter ───────────────────────────────────────────────────────────────────

def test_all_sixteen_tools_adapted():
    names = {t.name for t in ALL_TOOLS}
    assert len(ALL_TOOLS) == 16
    assert "cancel_pending_order" in names
    assert "exchange_delivered_order_items" in names
    assert "think" in names


def test_adapted_tool_invokes_against_store():
    user_id = next(iter(tau_db.DATA["users"]))
    get_user = next(t for t in ALL_TOOLS if t.name == "get_user_details")
    result = get_user.invoke({"user_id": user_id})
    assert json.loads(result)  # valid JSON payload, not an error string


def test_adapted_mutating_tool_changes_store():
    order_id = next(
        oid for oid, o in tau_db.DATA["orders"].items() if o["status"] == "pending"
    )
    cancel = next(t for t in ALL_TOOLS if t.name == "cancel_pending_order")
    cancel.invoke({"order_id": order_id, "reason": "no longer needed"})
    assert tau_db.DATA["orders"][order_id]["status"] == "cancelled"


def test_tool_risk_covers_every_tool():
    assert set(TOOL_RISK) == {t.name for t in ALL_TOOLS}
    for name in (
        "cancel_pending_order", "exchange_delivered_order_items",
        "modify_pending_order_items", "modify_pending_order_payment",
        "modify_pending_order_address", "modify_user_address",
        "return_delivered_order_items",
    ):
        assert TOOL_RISK[name] == "high", name
    assert TOOL_RISK["get_order_details"] == "low"


# ── ground-truth fidelity ─────────────────────────────────────────────────────

def test_every_ground_truth_sequence_replays_cleanly():
    """All 115 vendored tasks must replay their ground-truth actions without
    raising. 'Error:' string returns are legitimate — τ-bench ground truths
    deliberately include rejected calls (e.g. a refund to a disallowed payment
    method followed by escalation), and upstream grading replays them the same
    way. The pinned error signature guards against adapter drift: if our port
    diverges from the vendored revision, this count changes."""
    assert len(TASKS) == 115
    errors = []
    for task in TASKS:
        tau_db.reset()
        for action in task["ground_truth_actions"]:
            result = TOOL_CLASSES[action["name"]].invoke(
                tau_db.DATA, **action["kwargs"]
            )
            if isinstance(result, str) and result.startswith("Error:"):
                errors.append(task["id"])
    # Upstream-faithful signature at the pinned revision: 24 errored ground-truth
    # actions spread over 19 tasks.
    assert len(errors) == 24
    assert len(set(errors)) == 19


def test_replay_actions_produces_expected_state():
    """replay_actions is what grading uses to compute the expected final DB."""
    task = next(
        t for t in TASKS
        if any(a["name"] == "cancel_pending_order" for a in t["ground_truth_actions"])
    )
    tau_db.reset()
    replay_actions(task["ground_truth_actions"], tau_db.DATA)
    cancel = next(
        a for a in task["ground_truth_actions"] if a["name"] == "cancel_pending_order"
    )
    assert tau_db.DATA["orders"][cancel["kwargs"]["order_id"]]["status"] == "cancelled"
