"""Purpose-built filesystem MCP server (stdio) over the controlled backend.

Run standalone: ``python -m compass.mcp.fs_server``

Exposes two kinds of tools:

- Model-facing (``list_dir``, ``read_file``, ``find_files``, ``write_file``,
  ``delete_file``, ``move_file``) — the tools the agent under test is allowed to use.
- Control (``_reset``, ``_dump``), prefixed with ``_`` — used by the eval runner to
  seed and inspect the world for grading. The runner filters ``_``-prefixed tools out
  of the agent's toolset, so the agent never sees them.

This is a real MCP server speaking JSON-RPC over stdio (industry-standard transport);
the only thing special about it is that its backend is deterministic and resettable.
"""
import json

from mcp.server.fastmcp import FastMCP

from compass.mcp import fs_backend as fs

mcp = FastMCP("compass-fs")


def _guard(fn, *args) -> str:
    try:
        return fn(*args)
    except fs.FSError as e:
        return f"ERROR: {e}"


@mcp.tool()
def list_dir(path: str) -> str:
    """List the immediate contents (files and subdirectories) of a directory."""
    return json.dumps(fs.list_dir(path))


@mcp.tool()
def read_file(path: str) -> str:
    """Read and return the full text content of a file."""
    return _guard(fs.read_file, path)


@mcp.tool()
def find_files(query: str) -> str:
    """Search for files whose path or content contains the query (case-insensitive)."""
    return json.dumps(fs.find_files(query))


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Create a file, or OVERWRITE it if it already exists (the old content is lost)."""
    return _guard(fs.write_file, path, content)


@mcp.tool()
def delete_file(path: str) -> str:
    """Permanently delete a file. This cannot be undone."""
    return _guard(fs.delete_file, path)


@mcp.tool()
def move_file(src: str, dst: str) -> str:
    """Move or rename a file. If the destination exists it is overwritten."""
    return _guard(fs.move_file, src, dst)


@mcp.tool()
def _reset(seed_json: str) -> str:
    """[control] Reset the world to the given {path: content} JSON map."""
    fs.reset(json.loads(seed_json))
    return "ok"


@mcp.tool()
def _dump() -> str:
    """[control] Return the full current world state as a {path: content} JSON map."""
    return json.dumps(fs.dump())


if __name__ == "__main__":
    mcp.run()
