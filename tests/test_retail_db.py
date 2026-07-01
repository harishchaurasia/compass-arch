"""Tests for the retail in-memory DB reset helper.

Without an explicit reset between trials, a mutation from one trial (e.g.
cancelling an order) leaks into the next trial that touches the same order
id, silently corrupting any eval run that reuses order ids across tasks or
conditions within a single process.
"""
import compass.tools.retail_db as db


def test_reset_db_restores_mutated_order():
    db.ORDERS["#W1111111"]["status"] = "mutated"

    db.reset_db()

    assert db.ORDERS["#W1111111"]["status"] == "pending"
