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
