"""In-memory retail database. Tools read/write these dicts directly.
Tests replace them via monkeypatch fixtures before each test.
"""
import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent.parent / "tasks" / "tau_bench" / "data"


def _load(filename: str) -> dict:
    path = _DATA_DIR / filename
    if path.exists():
        return json.loads(path.read_text())
    return {}


USERS: dict = _load("users.json")
ORDERS: dict = _load("orders.json")
PRODUCTS: dict = _load("products.json")


def reset_db() -> None:
    """Reload USERS/ORDERS/PRODUCTS fresh from disk, discarding mutations.

    Call before each eval trial. Tools mutate these dicts directly with no
    per-trial isolation, so without this, a mutation from one trial (e.g.
    cancelling an order) leaks into any later trial that reuses the same
    order id, corrupting the comparison between conditions.
    """
    global USERS, ORDERS, PRODUCTS
    USERS = _load("users.json")
    ORDERS = _load("orders.json")
    PRODUCTS = _load("products.json")
