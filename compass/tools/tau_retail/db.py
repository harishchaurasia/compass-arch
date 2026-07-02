"""Mutable data store for the vendored τ-bench retail environment.

Vendored tools operate on a plain dict of {"users", "orders", "products"};
DATA is that dict. reset() restores it to the canonical vendored state —
call it before every trial so mutations never leak across trials.
"""
import copy
import json
from pathlib import Path

_DATA_DIR = Path(__file__).parents[3] / "tasks" / "tau_bench" / "real_data"

DATA: dict = {"users": {}, "orders": {}, "products": {}}
_canonical: dict | None = None


def canonical() -> dict:
    """The pristine vendored dataset (do not mutate the returned dict)."""
    global _canonical
    if _canonical is None:
        _canonical = {
            key: json.loads((_DATA_DIR / f"{key}.json").read_text())
            for key in ("users", "orders", "products")
        }
    return _canonical


def reset() -> None:
    for key, value in canonical().items():
        DATA[key] = copy.deepcopy(value)
