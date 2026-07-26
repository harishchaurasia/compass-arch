"""Vendor τ-bench environments (Sierra Research, MIT license).

Downloads, at a pinned revision, for each domain:
  - DB data (users/orders/products or users/flights/reservations) + policy wiki
  - tool implementations (verbatim, import line rewritten)
  - tasks_test.py, converted to JSON

Usage:
    uv run python scripts/vendor_tau_bench.py            # retail + airline
    uv run python scripts/vendor_tau_bench.py --domain airline

Re-running is idempotent: everything is overwritten from the pinned revision.
"""
import argparse
import json
import sys
import types
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REV = "59a200c6d575d595120f1cb70fea53cef0632f6b"  # tau-bench main, 2026-03-18
REPO = f"https://raw.githubusercontent.com/sierra-research/tau-bench/{REV}/tau_bench/envs"

# Windows consoles default stdout to cp1252, which can't encode the τ glyph the
# progress prints use; force UTF-8 so vendoring works identically on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).parent.parent

RETAIL_TOOLS = [
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

AIRLINE_TOOLS = [
    "book_reservation.py",
    "calculate.py",
    "cancel_reservation.py",
    "get_reservation_details.py",
    "get_user_details.py",
    "list_all_airports.py",
    "search_direct_flight.py",
    "search_onestop_flight.py",
    "send_certificate.py",
    "think.py",
    "transfer_to_human_agents.py",
    "update_reservation_baggages.py",
    "update_reservation_flights.py",
    "update_reservation_passengers.py",
]

DOMAINS = {
    "retail": {
        "data_dir": ROOT / "tasks" / "tau_bench" / "real_data",
        "vendor_dir": ROOT / "compass" / "tools" / "tau_retail" / "vendor",
        "vendor_pkg": "compass.tools.tau_retail.vendor.tool",
        "data_files": ("users.json", "orders.json", "products.json"),
        "tool_files": RETAIL_TOOLS,
        "tasks_var": "TASKS_TEST",
        "id_prefix": "tau_retail",
        "out_tasks": ROOT / "tasks" / "tau_bench" / "tasks_real.json",
    },
    "airline": {
        "data_dir": ROOT / "tasks" / "tau_bench" / "airline_data",
        "vendor_dir": ROOT / "compass" / "tools" / "tau_airline" / "vendor",
        "vendor_pkg": "compass.tools.tau_airline.vendor.tool",
        "data_files": ("users.json", "flights.json", "reservations.json"),
        "tool_files": AIRLINE_TOOLS,
        "tasks_var": "TASKS",
        "id_prefix": "tau_airline",
        "out_tasks": ROOT / "tasks" / "tau_bench" / "airline_tasks.json",
    },
}

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

Files in this directory (and the JSONs in the sibling data dir) are vendored
verbatim except for one rewritten import line per tool module.
Regenerate with: uv run python scripts/vendor_tau_bench.py
"""


def fetch(base: str, path: str) -> bytes:
    with urllib.request.urlopen(f"{base}/{path}") as resp:
        return resp.read()


def vendor_data(cfg: dict, base: str) -> None:
    cfg["data_dir"].mkdir(parents=True, exist_ok=True)
    for name in cfg["data_files"]:
        (cfg["data_dir"] / name).write_bytes(fetch(base, f"data/{name}"))
        print(f"  data/{name}")
    (cfg["data_dir"] / "wiki.md").write_bytes(fetch(base, "wiki.md"))
    print("  wiki.md")


def vendor_tools(cfg: dict, base: str) -> None:
    cfg["vendor_dir"].mkdir(parents=True, exist_ok=True)
    (cfg["vendor_dir"] / "tool.py").write_text(TOOL_BASE_STUB, encoding="utf-8")
    (cfg["vendor_dir"] / "__init__.py").write_text("", encoding="utf-8")
    (cfg["vendor_dir"] / "README.md").write_text(ATTRIBUTION.format(rev=REV), encoding="utf-8")
    for name in cfg["tool_files"]:
        src = fetch(base, f"tools/{name}").decode()
        src = src.replace(
            "from tau_bench.envs.tool import Tool",
            f"from {cfg['vendor_pkg']} import Tool",
        )
        (cfg["vendor_dir"] / name).write_text(src, encoding="utf-8")
        print(f"  tools/{name}")


def convert_tasks(cfg: dict, base: str) -> None:
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
    exec(fetch(base, "tasks_test.py").decode(), namespace)
    tasks = namespace[cfg["tasks_var"]]

    domain = cfg["id_prefix"].split("_")[1]
    converted = [
        {
            "id": f"{cfg['id_prefix']}_{i:03d}",
            "domain": domain,
            "user_id": t.user_id,
            "instruction": t.instruction,
            "ground_truth_actions": [
                {"name": a.name, "kwargs": a.kwargs} for a in t.actions
            ],
            "expected_outputs": list(t.outputs),
        }
        for i, t in enumerate(tasks)
    ]
    cfg["out_tasks"].write_text(json.dumps(converted, indent=2), encoding="utf-8")
    print(f"  {len(converted)} tasks → {cfg['out_tasks'].relative_to(ROOT)}")


def vendor_domain(name: str) -> None:
    cfg = DOMAINS[name]
    base = f"{REPO}/{name}"
    print(f"Vendoring τ-bench {name} @ {REV[:12]}")
    vendor_data(cfg, base)
    vendor_tools(cfg, base)
    convert_tasks(cfg, base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=[*DOMAINS, "all"], default="all")
    args = parser.parse_args()
    names = list(DOMAINS) if args.domain == "all" else [args.domain]
    for name in names:
        vendor_domain(name)


if __name__ == "__main__":
    main()
