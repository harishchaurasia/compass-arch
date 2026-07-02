"""LangChain adapters over the vendored τ-bench retail tools.

Each vendored Sierra tool class (invoke(data, ...) + get_info() JSON schema)
is wrapped into a StructuredTool bound to the module-level data store in
db.DATA. Vendoring + adapting keeps the environment faithful by construction;
we never re-implement tool semantics.
"""
import copy
import importlib

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

import compass.tools.tau_retail.db as db
from compass.tools.tau_retail.vendor.tool import Tool

_TOOL_MODULES = [
    "calculate",
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "think",
    "transfer_to_human_agents",
]

# Static risk classes for the policy's risk floor: everything that mutates
# the store is high, pure lookups/reasoning are low.
TOOL_RISK = {
    "calculate": "low",
    "cancel_pending_order": "high",
    "exchange_delivered_order_items": "high",
    "find_user_id_by_email": "low",
    "find_user_id_by_name_zip": "low",
    "get_order_details": "low",
    "get_product_details": "low",
    "get_user_details": "low",
    "list_all_product_types": "low",
    "modify_pending_order_address": "high",
    "modify_pending_order_items": "high",
    "modify_pending_order_payment": "high",
    "modify_user_address": "high",
    "return_delivered_order_items": "high",
    "think": "low",
    "transfer_to_human_agents": "low",
}

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _py_type(prop: dict):
    if prop.get("type") == "array" and prop.get("items", {}).get("type") == "string":
        return list[str]
    return _JSON_TO_PY.get(prop.get("type", "string"), str)


def _args_model(name: str, params: dict):
    required = set(params.get("required", []))
    fields = {}
    for pname, prop in params.get("properties", {}).items():
        default = ... if pname in required else None
        fields[pname] = (
            _py_type(prop),
            Field(default, description=prop.get("description", "")),
        )
    return create_model(f"{name}_args", **fields)


def _tool_class(module_name: str) -> type[Tool]:
    module = importlib.import_module(f"compass.tools.tau_retail.vendor.{module_name}")
    for obj in vars(module).values():
        if isinstance(obj, type) and issubclass(obj, Tool) and obj is not Tool:
            return obj
    raise ImportError(f"No Tool subclass found in vendored module {module_name!r}")


def _adapt(cls: type[Tool]) -> StructuredTool:
    info = cls.get_info()["function"]

    def _run(**kwargs) -> str:
        return cls.invoke(db.DATA, **kwargs)

    return StructuredTool.from_function(
        func=_run,
        name=info["name"],
        description=info["description"],
        args_schema=_args_model(info["name"], info["parameters"]),
    )


TOOL_CLASSES: dict[str, type[Tool]] = {
    name: _tool_class(name) for name in _TOOL_MODULES
}
ALL_TOOLS: list[StructuredTool] = [_adapt(cls) for cls in TOOL_CLASSES.values()]


def replay_actions(actions: list[dict], data: dict) -> None:
    """Apply a task's ground-truth action sequence to `data` in place.
    Grading uses this to compute the expected final DB state. 'Error:'
    returns are kept, matching upstream: some ground truths deliberately
    include rejected calls."""
    for action in actions:
        TOOL_CLASSES[action["name"]].invoke(data, **action["kwargs"])


def expected_state(actions: list[dict]) -> dict:
    """Final DB state after replaying `actions` on the pristine dataset."""
    scratch = {k: copy.deepcopy(v) for k, v in db.canonical().items()}
    replay_actions(actions, scratch)
    return scratch
