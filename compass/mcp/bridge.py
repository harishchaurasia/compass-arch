"""Synchronous bridge from a stdio MCP server to the (sync) Compass graph.

langchain-mcp-adapters is async, and `MultiServerMCPClient.get_tools()` opens a
fresh session per call — so tool state would not persist across an agent's
sequential calls. The Compass agent graph, meanwhile, runs synchronously
(`agent.invoke`). This bridge reconciles both:

- one persistent MCP session (shared state within a trial) is opened inside a
  single long-lived task on a dedicated event loop in a background thread;
- that task serves tool calls off a queue, so the session's async-context is
  entered and exited in the *same* task (anyio requires this);
- each MCP tool is wrapped as a *sync* LangChain StructuredTool that submits its
  coroutine to that task and blocks for the result;
- control tools (``_``-prefixed) are kept out of the agent's toolset and exposed
  as `reset()` / `dump()` for the eval runner.

So the eval genuinely runs over real MCP stdio, while the graph sees ordinary
synchronous tools.
"""
import asyncio
import json
import threading
from concurrent.futures import Future
from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

_STOP = object()


def _text(result: Any) -> str:
    """Flatten an MCP tool result (content-block list) to a plain string."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(getattr(block, "text", str(block)))
        return "".join(parts)
    return str(result)


class MCPToolServer:
    """Owns a background event loop + one persistent session to an MCP server."""

    def __init__(self, command: str, args: list[str], server_name: str = "fs"):
        self._name = server_name
        self._client = MultiServerMCPClient(
            {server_name: {"command": command, "args": args, "transport": "stdio"}}
        )
        self._raw: list[BaseTool] = []
        self._queue: asyncio.Queue = None  # type: ignore[assignment]
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        self._ready.wait(timeout=30)

    async def _serve(self) -> None:
        """Single task: open the session, then serve queued calls until stopped,
        so the session context is entered and exited in one task."""
        self._queue = asyncio.Queue()
        async with self._client.session(self._name) as session:
            self._raw = await load_mcp_tools(session)
            self._ready.set()
            while True:
                factory, fut = await self._queue.get()
                if factory is _STOP:
                    fut.set_result(None)
                    return
                try:
                    fut.set_result(await factory())
                except Exception as e:  # surface tool/transport errors to caller
                    fut.set_exception(e)

    def _submit(self, factory: Callable) -> Any:
        fut: Future = Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (factory, fut))
        return fut.result()

    def _call(self, name: str, args: dict) -> str:
        tool = next(t for t in self._raw if t.name == name)
        return _text(self._submit(lambda: tool.ainvoke(args)))

    def agent_tools(self) -> list[StructuredTool]:
        """Sync StructuredTools for every model-facing (non-``_``) MCP tool,
        preserving name / description / args schema."""
        return [self._wrap(raw) for raw in self._raw if not raw.name.startswith("_")]

    def _wrap(self, raw: BaseTool) -> StructuredTool:
        name = raw.name

        def _sync(**kwargs) -> str:
            return self._call(name, kwargs)

        return StructuredTool(
            name=name,
            description=raw.description,
            args_schema=raw.args_schema,
            func=_sync,
        )

    # --- control surface for the runner (agent never sees these) ---
    def reset(self, seed: dict[str, str]) -> None:
        self._call("_reset", {"seed_json": json.dumps(seed)})

    def dump(self) -> dict[str, str]:
        return json.loads(self._call("_dump", {}))

    def close(self) -> None:
        fut: Future = Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (_STOP, fut))
        try:
            fut.result(timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
