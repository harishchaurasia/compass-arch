"""Custom MCP task suite runner (~12 tasks, cascading-failure design)."""
from pathlib import Path

RESULTS_DB = Path(__file__).parent.parent / "results" / "trials.db"


def run_mcp_trial(task: dict, agent, condition: str, model: str) -> dict:
    raise NotImplementedError


def run_mcp_suite(tasks: list[dict], agents: dict, db_path: Path = RESULTS_DB) -> None:
    raise NotImplementedError
