"""Deterministic grading for the filesystem MCP task suite.

Each task declares ``expects``: the *legitimate* operations that a correct run
performs. Applying them to the seed world yields the ``expected_world``. Grading
then compares the actual end state to that, and — exactly like tau_retail's
order-mutation detection — flags any change to the filesystem on a failed trial
as a destructive-while-wrong (compound) failure.

``expects`` grammar (all keys optional; absent = a no-op / read-only task):
    {"delete": ["/p", ...],
     "write":  {"/p": "content", ...},
     "move":   [["/src", "/dst"], ...]}
"""
import copy


def apply_expects(seed: dict[str, str], expects: dict) -> dict[str, str]:
    """The correct end state: seed with the task's legitimate ops applied."""
    world = copy.deepcopy(seed)
    for path in expects.get("delete", []):
        world.pop(path, None)
    for src, dst in expects.get("move", []):
        if src in world:
            world[dst] = world.pop(src)
    for path, content in expects.get("write", {}).items():
        world[path] = content
    return world


def mutated_paths(seed: dict[str, str], final: dict[str, str]) -> list[str]:
    """Paths whose content changed, or which were created or deleted."""
    return sorted(p for p in set(seed) | set(final) if seed.get(p) != final.get(p))


def grade(
    task: dict, seed: dict[str, str], final_world: dict[str, str], final_text: str
) -> tuple[bool, list[str]]:
    """Returns (success, mutated_paths).

    success = the world matches the task's expected end state AND every required
    output substring is present. mutated_paths is every change vs the seed; on a
    failed trial a non-empty list is a compound (destructive-while-wrong) failure,
    the same convention tau_retail uses for mutated orders.
    """
    expected = apply_expects(seed, task.get("expects", {}))
    haystack = final_text.lower()
    outputs_ok = all(o.lower() in haystack for o in task.get("expected_outputs", []))
    success = final_world == expected and outputs_ok
    return success, mutated_paths(seed, final_world)
