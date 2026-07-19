"""Demo: drive real, off-the-shelf MCP servers with the Compass bridge.

The eval suite grades on a purpose-built server so results are deterministic, but
the *same* bridge (compass.mcp.bridge.MCPToolServer) speaks plain MCP and works
against any real server. This script proves it end-to-end:

    uv run python scripts/mcp_real_servers.py            # official filesystem MCP
    uv run python scripts/mcp_real_servers.py --github   # GitHub MCP (needs a token)

Filesystem: uses the official `@modelcontextprotocol/server-filesystem` over a
scratch dir (via npx; Node required). GitHub: uses `@modelcontextprotocol/server-github`
and only runs if GITHUB_PERSONAL_ACCESS_TOKEN is set.

It lists the server's tools, assigns Compass risk classes by name (writes/deletes/
moves = high), and — because the bridge yields ordinary sync LangChain tools — those
tools drop straight into build_compass_agent, so Compass can gate real MCP actions
exactly like it gates the retail and purpose-built suites.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from compass.mcp.bridge import MCPToolServer

# Name-based risk heuristic for arbitrary MCP servers: any tool that can change
# state is high risk; everything else is low. Compass's policy gates on this.
_HIGH_RISK_HINTS = ("write", "delete", "remove", "move", "edit", "create",
                    "update", "put", "push", "merge", "close", "rename")


def risk_for(tool_name: str) -> str:
    n = tool_name.lower()
    return "high" if any(h in n for h in _HIGH_RISK_HINTS) else "low"


def demo_filesystem() -> None:
    scratch = Path(tempfile.mkdtemp(prefix="compass-mcp-"))
    (scratch / "config.yaml").write_text("service: demo\nreplicas: 3\n")
    print(f"scratch dir: {scratch}")

    srv = MCPToolServer(
        "npx", ["-y", "@modelcontextprotocol/server-filesystem", str(scratch)]
    )
    try:
        tools = srv.agent_tools()
        print(f"\nofficial filesystem MCP exposes {len(tools)} tools:")
        for t in sorted(tools, key=lambda t: t.name):
            print(f"  [{risk_for(t.name):>4}] {t.name}")
        read = next(t for t in tools if t.name == "read_file")
        print("\nread_file via the official server:")
        print("  " + read.invoke({"path": str(scratch / "config.yaml")}).replace("\n", " "))
        print("\nThese are plain sync LangChain tools + a risk map — exactly what "
              "build_compass_agent(model, tools, tool_risk=...) consumes.")
    finally:
        srv.close()


def demo_github() -> None:
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        print("GITHUB_PERSONAL_ACCESS_TOKEN not set — skipping the GitHub MCP demo.")
        print("To run it: set the token and re-run with --github. Config used:")
        print('  command="npx", args=["-y", "@modelcontextprotocol/server-github"],')
        print('  env={"GITHUB_PERSONAL_ACCESS_TOKEN": "<token>"}')
        return
    srv = MCPToolServer(
        "npx", ["-y", "@modelcontextprotocol/server-github"],
        server_name="github", env={"GITHUB_PERSONAL_ACCESS_TOKEN": token},
    )
    try:
        tools = srv.agent_tools()
        print(f"\nGitHub MCP exposes {len(tools)} tools (risk-classed):")
        for t in sorted(tools, key=lambda t: t.name):
            print(f"  [{risk_for(t.name):>4}] {t.name}")
        print("\nHigh-risk tools (close_issue, merge_pull_request, push_files, ...) are "
              "exactly the destructive actions Compass would gate before executing.")
    finally:
        srv.close()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    load_dotenv()  # reads GITHUB_PERSONAL_ACCESS_TOKEN from .env (gitignored)
    parser = argparse.ArgumentParser()
    parser.add_argument("--github", action="store_true", help="also run the GitHub MCP demo")
    args = parser.parse_args()

    demo_filesystem()
    if args.github:
        demo_github()


if __name__ == "__main__":
    main()
