"""In-memory filesystem backend for the MCP task suite.

Pure state + operations, no MCP dependency, so it unit-tests directly. The
filesystem is a flat ``{path: content}`` map; directories are implicit in the
paths. Operations mirror the *shape* of the official filesystem MCP server
(list / read / find + write / delete / move) so the same tasks can later run
against the real server, but here the state is controlled and resettable and
every mutation is observable — which is what makes deterministic grading of
"destructive action while wrong" possible.

Risk classes (for the policy's risk floor): reads are ``low``; anything that
overwrites, deletes, or moves a file is ``high`` because it is irreversible.
"""
import copy

# Static risk class per model-facing tool. The control tools (_reset/_dump) are
# never exposed to the agent, so they need no risk class.
TOOL_RISK: dict[str, str] = {
    "list_dir": "low",
    "read_file": "low",
    "find_files": "low",
    "write_file": "high",
    "delete_file": "high",
    "move_file": "high",
}

# Module-level world state. The server process owns one of these; the runner
# drives it through the _reset / _dump control tools.
WORLD: dict[str, str] = {}


class FSError(Exception):
    """Returned to the agent as an error string, not raised across MCP."""


def reset(seed: dict[str, str]) -> None:
    """Replace the world with a deep copy of ``seed`` (path -> content)."""
    global WORLD
    WORLD = copy.deepcopy(seed)


def dump() -> dict[str, str]:
    """Full current state, for the runner to snapshot / diff for grading."""
    return copy.deepcopy(WORLD)


def _norm(path: str) -> str:
    """Normalize to a leading-slash, no-trailing-slash absolute path."""
    p = "/" + path.strip().strip("/")
    return p


def list_dir(path: str) -> list[str]:
    """Immediate children (files and subdirs) of a directory path."""
    prefix = _norm(path).rstrip("/") + "/"
    if prefix == "//":
        prefix = "/"
    children: set[str] = set()
    for p in WORLD:
        if p.startswith(prefix) and p != prefix.rstrip("/"):
            rest = p[len(prefix):]
            children.add(rest.split("/", 1)[0])
    return sorted(children)


def read_file(path: str) -> str:
    p = _norm(path)
    if p not in WORLD:
        raise FSError(f"no such file: {p}")
    return WORLD[p]


def find_files(query: str) -> list[str]:
    """Paths whose name or content contains ``query`` (case-insensitive)."""
    q = query.lower()
    return sorted(p for p, c in WORLD.items() if q in p.lower() or q in c.lower())


def write_file(path: str, content: str) -> str:
    """Create or OVERWRITE a file. Overwriting loses the old content — high risk."""
    p = _norm(path)
    existed = p in WORLD
    WORLD[p] = content
    return f"{'overwrote' if existed else 'wrote'} {p} ({len(content)} bytes)"


def delete_file(path: str) -> str:
    """Delete a file. Irreversible — high risk."""
    p = _norm(path)
    if p not in WORLD:
        raise FSError(f"no such file: {p}")
    del WORLD[p]
    return f"deleted {p}"


def move_file(src: str, dst: str) -> str:
    """Move/rename a file. Clobbers ``dst`` if it exists — high risk."""
    s, d = _norm(src), _norm(dst)
    if s not in WORLD:
        raise FSError(f"no such file: {s}")
    clobbered = d in WORLD
    WORLD[d] = WORLD.pop(s)
    return f"moved {s} -> {d}" + (" (clobbered existing)" if clobbered else "")
