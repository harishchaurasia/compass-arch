"""Gate your own agent with Compass in ~30 lines.

This is the whole integration story: define your tools, tag which ones are
destructive, build a Compass agent, and run it. Compass decides execute /
self-verify / abstain before each action - the high-risk ones only fire when its
calibrated confidence clears the bar.

Run it:
    uv run python examples/gate_your_agent.py                      # openai gpt-4o-mini (needs OPENAI_API_KEY)
    uv run python examples/gate_your_agent.py --provider ollama --model qwen2.5:7b   # local, no key

Try changing the request at the bottom to something under-specified and watch the
high-risk `delete_file` get gated instead of executed.
"""

import argparse

from dotenv import load_dotenv
from langchain_core.tools import tool

from compass import build_compass_agent, get_model, run

# --- 1. Your tools (any LangChain tools work). A tiny in-memory FS stands in. ---
FILES = {"report.txt": "...", "report.bak": "...", "notes.md": "..."}


@tool
def list_files() -> str:
    """List the files in the workspace."""
    return ", ".join(sorted(FILES))


@tool
def delete_file(name: str) -> str:
    """Permanently delete a file by name. This cannot be undone."""
    if name not in FILES:
        return f"Error: no such file {name!r}"
    del FILES[name]
    return f"Deleted {name}"


# --- 2. Tag which tools are destructive. Anything that mutates state is 'high'. ---
TOOL_RISK = {"list_files": "low", "delete_file": "high"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    load_dotenv()

    # --- 3. Build a gated agent and run it. That's the whole integration. ---
    agent = build_compass_agent(
        get_model(args.provider, args.model, temperature=0),
        tools=[list_files, delete_file],
        tool_risk=TOOL_RISK,
    )

    request = "Delete the old backup of the report. I think it's one of the .bak files."
    result = run(agent, request)

    print(f"\nRequest : {request}")
    print(f"Steps   : {result.steps}")
    print(f"Conf    : {[round(c, 2) for c in result.confidences]}")
    print(f"Gated p : {[round(p, 2) for p in result.success_probs]}")
    print(f"Abstain : {result.abstained}")
    print(f"Files   : {sorted(FILES)}   (delete_file fires only if the gate let it)")
    print(f"\nAgent   : {result.final_message}\n")


if __name__ == "__main__":
    main()
