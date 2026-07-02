"""Vendor the τ-bench retail environment (Sierra Research, MIT license).

Downloads, at a pinned revision:
  - retail DB data (users/orders/products) + policy wiki  → tasks/tau_bench/real_data/
  - tool implementations (verbatim, import line rewritten) → compass/tools/tau_retail/vendor/
  - tasks_test.py, converted to JSON                       → tasks/tau_bench/tasks_real.json

Usage:
    uv run python scripts/vendor_tau_bench.py

Re-running is idempotent: everything is overwritten from the pinned revision.
"""
import json
import sys
import types
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REV = "59a200c6d575d595120f1cb70fea53cef0632f6b"  # tau-bench main, 2026-03-18
BASE = f"https://raw.githubusercontent.com/sierra-research/tau-bench/{REV}/tau_bench/envs/retail"

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "tasks" / "tau_bench" / "real_data"
VENDOR_DIR = ROOT / "compass" / "tools" / "tau_retail" / "vendor"

TOOL_FILES = [
    "calculate.py",
    "cancel_pending_order.py",
    "exchange_delivered_order_items.py",
    "find_user_id_by_email.py",
    "find_user_id_by_name_zip.py",
    "get_order_details.py",
    "get_product_details.py",
    "get_user_details.py",
    "list_all_product_types.py",
    "modify_pending_order_address.py",
    "modify_pending_order_items.py",
    "modify_pending_order_payment.py",
    "modify_user_address.py",
    "return_delivered_order_items.py",
    "think.py",
    "transfer_to_human_agents.py",
]

TOOL_BASE_STUB = '''"""Minimal stand-in for tau_bench.envs.tool.Tool (vendored tools subclass it)."""


class Tool:
    @staticmethod
    def invoke(data, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_info():
        raise NotImplementedError
'''

ATTRIBUTION = """# Vendored from τ-bench

Source: https://github.com/sierra-research/tau-bench @ {rev}
License: MIT (Copyright Sierra)

Files in this directory (and the JSONs in tasks/tau_bench/real_data/) are
vendored verbatim except for one rewritten import line per tool module.
Regenerate with: uv run python scripts/vendor_tau_bench.py
"""


def fetch(path: str) -> bytes:
    with urllib.request.urlopen(f"{BASE}/{path}") as resp:
        return resp.read()


def vendor_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("users.json", "orders.json", "products.json"):
        (DATA_DIR / name).write_bytes(fetch(f"data/{name}"))
        print(f"  data/{name}")
    (DATA_DIR / "wiki.md").write_bytes(fetch("wiki.md"))
    print("  wiki.md")


def vendor_tools() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    (VENDOR_DIR / "tool.py").write_text(TOOL_BASE_STUB)
    (VENDOR_DIR / "__init__.py").write_text("")
    (VENDOR_DIR / "README.md").write_text(ATTRIBUTION.format(rev=REV))
    for name in TOOL_FILES:
        src = fetch(f"tools/{name}").decode()
        src = src.replace(
            "from tau_bench.envs.tool import Tool",
            "from compass.tools.tau_retail.vendor.tool import Tool",
        )
        (VENDOR_DIR / name).write_text(src)
        print(f"  tools/{name}")


def convert_tasks() -> None:
    """Exec tasks_test.py with stub Task/Action types and dump plain JSON."""

    @dataclass
    class Action:
        name: str
        kwargs: dict = field(default_factory=dict)

    @dataclass
    class Task:
        annotator: str = ""
        user_id: str = ""
        instruction: str = ""
        actions: list = field(default_factory=list)
        outputs: list = field(default_factory=list)

    stub = types.ModuleType("tau_bench.types")
    stub.Action, stub.Task = Action, Task
    pkg = types.ModuleType("tau_bench")
    pkg.types = stub
    sys.modules["tau_bench"] = pkg
    sys.modules["tau_bench.types"] = stub

    namespace: dict = {}
    exec(fetch("tasks_test.py").decode(), namespace)
    tasks = namespace["TASKS_TEST"]

    converted = [
        {
            "id": f"tau_retail_{i:03d}",
            "domain": "retail",
            "user_id": t.user_id,
            "instruction": t.instruction,
            "ground_truth_actions": [
                {"name": a.name, "kwargs": a.kwargs} for a in t.actions
            ],
            "expected_outputs": list(t.outputs),
        }
        for i, t in enumerate(tasks)
    ]
    out = ROOT / "tasks" / "tau_bench" / "tasks_real.json"
    out.write_text(json.dumps(converted, indent=2))
    print(f"  {len(converted)} tasks → {out.relative_to(ROOT)}")


def main() -> None:
    print(f"Vendoring τ-bench retail @ {REV[:12]}")
    vendor_data()
    vendor_tools()
    convert_tasks()


if __name__ == "__main__":
    main()
